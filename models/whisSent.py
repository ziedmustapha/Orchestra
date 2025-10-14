# models/whisSent.py - Whisper ASR + FR Emotion Classification
# Author: Zied Mustapha
import os
import io
import gc
import sys
import time
import uuid
import base64
import atexit
import signal
import logging
import tempfile
import requests
import multiprocessing as mp
from multiprocessing import Queue
from typing import Any, Dict, Optional, Tuple
from logging_config import setup_logging_for_process, set_correlation_id, clear_correlation_id, get_correlation_id

import torch
from transformers import pipeline

# Configure logging for this module
logger = logging.getLogger(__name__)

# Environment for stability
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

# Models
ASR_MODEL_ID = "openai/whisper-large-v3"
EMO_MODEL_ID = "Lajavaness/wav2vec2-lg-xlsr-fr-speech-emotion-recognition"


def _write_temp_audio_file(audio_spec: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Materialize incoming audio to a temporary file path.
    Supports formats:
      - {"data": <base64>, "format": "wav|mp3|flac|m4a|ogg"}
      - {"url": "http://..."}
      - {"path": "/abs/path/file.wav"}
    Returns: (path, cleanup_token) where cleanup_token!=None means we created a tmp file.
    """
    if not isinstance(audio_spec, dict):
        raise ValueError("audio must be a dict with one of keys: data|url|path")

    # From absolute path
    if 'path' in audio_spec and isinstance(audio_spec['path'], str):
        if not os.path.exists(audio_spec['path']):
            raise FileNotFoundError(f"Audio path not found: {audio_spec['path']}")
        return audio_spec['path'], None

    # From URL
    if 'url' in audio_spec and isinstance(audio_spec['url'], str):
        url = audio_spec['url']
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        ext = None
        # Try to guess extension from headers or url
        ctype = resp.headers.get('content-type', '')
        if 'wav' in ctype or url.lower().endswith('.wav'): ext = '.wav'
        elif 'flac' in ctype or url.lower().endswith('.flac'): ext = '.flac'
        elif 'mp3' in ctype or url.lower().endswith('.mp3'): ext = '.mp3'
        elif 'm4a' in ctype or url.lower().endswith('.m4a'): ext = '.m4a'
        elif 'ogg' in ctype or url.lower().endswith('.ogg'): ext = '.ogg'
        else: ext = '.wav'
        tmp = tempfile.NamedTemporaryFile(prefix='whissent_', suffix=ext, delete=False)
        tmp.write(resp.content)
        tmp.flush(); tmp.close()
        return tmp.name, tmp.name

    # From base64
    if 'data' in audio_spec and isinstance(audio_spec['data'], str):
        b64 = audio_spec['data']
        if b64.startswith('data:audio'):
            header, b64 = b64.split(',', 1)
            # try detect ext from header
            if 'wav' in header: ext = '.wav'
            elif 'flac' in header: ext = '.flac'
            elif 'mp3' in header: ext = '.mp3'
            elif 'm4a' in header: ext = '.m4a'
            elif 'ogg' in header: ext = '.ogg'
            else: ext = '.wav'
        else:
            # fallback to provided format
            ext = '.' + str(audio_spec.get('format', 'wav')).lower().strip('.')
        raw = base64.b64decode(b64)
        tmp = tempfile.NamedTemporaryFile(prefix='whissent_', suffix=ext, delete=False)
        tmp.write(raw)
        tmp.flush(); tmp.close()
        return tmp.name, tmp.name

    raise ValueError("audio must include one of: 'path', 'url', or 'data'")


def concurrent_whissent_worker(task_queue: Queue, result_queue: Queue, worker_id: int, gpu_id: str, process_id_str: str):
    """Subprocess that hosts the Whisper ASR and Emotion pipelines on a specific GPU."""
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    # Structured JSON logging per sub-process
    os.environ.setdefault("ROLE", "worker_subproc")
    os.environ.setdefault("SERVICE_NAME", "orchestra")
    setup_logging_for_process(f"workerproc_{worker_id}_whissent_{process_id_str}.jsonl")
    wlog = logging.getLogger(f"whissent_worker_proc_{process_id_str}")

    wlog.info(f"WhisSent worker {worker_id} (PID {os.getpid()}) starting on GPU {gpu_id}")

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if (device == 0) else torch.float32

    asr_pipe = None
    emo_pipe = None

    try:
        t0 = time.time()
        # Initialize pipelines
        wlog.info("Loading Whisper ASR pipeline...")
        asr_pipe = pipeline(
            task="automatic-speech-recognition",
            model=ASR_MODEL_ID,
            device=device,
            model_kwargs={"torch_dtype": dtype},
            chunk_length_s=30.0,
        )
        wlog.info("Loading FR Emotion classification pipeline...")
        emo_pipe = pipeline(
            task="audio-classification",
            model=EMO_MODEL_ID,
            device=device,
        )
        wlog.info(f"Pipelines loaded in {time.time()-t0:.2f}s")
    except Exception as e:
        wlog.error(f"Failed to load pipelines: {e}", exc_info=True)
        result_queue.put({"type":"ERROR_INIT", "error": f"Pipeline load failed: {e}"})
        return

    # Signal ready
    result_queue.put({"type":"READY"})
    wlog.info("WhisSent worker READY")

    while True:
        try:
            task = task_queue.get()
            if task is None:
                wlog.info("Shutdown signal received")
                break
            task_id = task.get("task_id", f"task-{uuid.uuid4().hex[:8]}")
            payload = task.get("payload", {})
            corr_id = task.get("correlation_id")
            if corr_id:
                try:
                    set_correlation_id(corr_id)
                except Exception:
                    pass

            # Unpack payload
            audio_spec = payload.get("audio")
            return_timestamps = bool(payload.get("return_timestamps", True))
            asr_task = payload.get("task", "transcribe")  # 'transcribe' or 'translate'
            language = payload.get("language", None)        # e.g., 'fr', 'en'
            top_k = int(payload.get("emotion_top_k", 3))

            wlog.info(f"Processing task {task_id} (timestamps={return_timestamps}, task={asr_task}, lang={language}, top_k={top_k})")

            tmp_path = None
            try:
                # 1) Materialize audio
                audio_path, cleanup_token = _write_temp_audio_file(audio_spec)
                tmp_path = cleanup_token

                # 2) Run ASR (Whisper)
                gen_kwargs = {}
                if asr_task in ("transcribe", "translate"):
                    gen_kwargs["task"] = asr_task
                if language:
                    gen_kwargs["language"] = language

                asr_t0 = time.time()
                asr_out = asr_pipe(
                    audio_path,
                    return_timestamps=return_timestamps,
                    generate_kwargs=gen_kwargs,
                )
                asr_dt = time.time() - asr_t0

                # Normalize ASR result
                transcript = asr_out["text"] if isinstance(asr_out, dict) else str(asr_out)
                chunks = asr_out.get("chunks", []) if isinstance(asr_out, dict) else []

                # 3) Emotion classification (overall clip)
                emo_t0 = time.time()
                emo_raw = emo_pipe(audio_path, top_k=max(1, top_k))
                emo_dt = time.time() - emo_t0

                # Normalize emotion output (list of {label, score})
                emotions = []
                if isinstance(emo_raw, list):
                    for item in emo_raw:
                        if isinstance(item, dict) and "label" in item and "score" in item:
                            emotions.append({"label": item["label"], "score": float(item["score"])})
                else:
                    # single result
                    if isinstance(emo_raw, dict) and "label" in emo_raw and "score" in emo_raw:
                        emotions.append({"label": emo_raw["label"], "score": float(emo_raw["score"])})

                result = {
                    "transcript": transcript,
                    "chunks": chunks,  # each with {"text", "timestamp": [start, end]}
                    "emotion": emotions[0]["label"] if emotions else None,
                    "emotion_scores": emotions,
                    "metrics": {
                        "asr_seconds": round(asr_dt, 3),
                        "emotion_seconds": round(emo_dt, 3),
                    }
                }

                result_queue.put({"type":"RESULT", "task_id": task_id, "result": result})
                wlog.info(f"Task {task_id} done. ASR {asr_dt:.2f}s, EMO {emo_dt:.2f}s")

            except Exception as e:
                wlog.error(f"Error processing task {task_id}: {e}", exc_info=True)
                result_queue.put({"type":"ERROR", "task_id": task_id, "error": str(e)})
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                try:
                    clear_correlation_id()
                except Exception:
                    pass

        except Exception as e:
            wlog.error(f"Worker loop error: {e}")
            continue

    # Cleanup
    try:
        del asr_pipe
        del emo_pipe
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    wlog.info("WhisSent worker cleanup completed")


class WhisSentWorkerState:
    def __init__(self, worker_id_gunicorn: int, assigned_gpu: str):
        self.gunicorn_worker_id = worker_id_gunicorn
        self.gpu_id_for_worker = assigned_gpu
        self.unique_process_id_str = str(uuid.uuid4())[:8]
        self.model_task_queue: Optional[Queue] = None
        self.model_result_queue: Optional[Queue] = None
        self.model_worker_process_handle = None
        self.wrapper: Optional[WhisSentModelWrapper] = None
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Creating WhisSent WorkerState on GPU {self.gpu_id_for_worker}")

    def initialize(self):
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Initializing WhisSent worker process...")
        try:
            mp_ctx = mp.get_context('spawn')
            self.model_task_queue = mp_ctx.Queue()
            self.model_result_queue = mp_ctx.Queue()
            self.model_worker_process_handle = mp_ctx.Process(
                target=concurrent_whissent_worker,
                args=(self.model_task_queue, self.model_result_queue, self.gunicorn_worker_id, self.gpu_id_for_worker, self.unique_process_id_str),
                daemon=False,
                name=f"whissent_model_proc_{self.unique_process_id_str}"
            )
            self.model_worker_process_handle.start()

            # Wait for READY
            ready = False
            t0 = time.time()
            while time.time()-t0 < 600:  # 10 minutes
                try:
                    msg = self.model_result_queue.get(timeout=5)
                except mp.queues.Empty:
                    continue
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") == "READY":
                    ready = True
                    break
                if msg.get("type") == "ERROR_INIT":
                    raise RuntimeError(msg.get("error", "Unknown init error"))
            if not ready:
                raise RuntimeError("WhisSent worker failed to signal readiness in time")

            self.wrapper = WhisSentModelWrapper(self.model_task_queue, self.model_result_queue, self.gunicorn_worker_id, self.unique_process_id_str)
            logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: WhisSent worker initialized successfully")
        except Exception:
            logger.exception("Failed to initialize WhisSent worker")
            self.cleanup()
            raise

    def cleanup(self):
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Cleaning up WhisSent worker...")
        try:
            if self.model_task_queue and self.model_worker_process_handle and self.model_worker_process_handle.is_alive():
                try:
                    self.model_task_queue.put(None)
                except Exception:
                    pass
                self.model_worker_process_handle.join(timeout=30)
                if self.model_worker_process_handle.is_alive():
                    self.model_worker_process_handle.terminate()
                    self.model_worker_process_handle.join(timeout=10)
        except Exception:
            logger.exception("Error during WhisSent worker cleanup")
        self.model_task_queue = None
        self.model_result_queue = None
        self.model_worker_process_handle = None
        self.wrapper = None


class WhisSentModelWrapper:
    def __init__(self, task_q: Queue, result_q: Queue, gunicorn_id: int, sub_proc_id: str):
        self.task_queue = task_q
        self.result_queue = result_q
        self.gunicorn_worker_id = gunicorn_id
        self.sub_process_id = sub_proc_id
        self.request_counter = 0
        self._lock = mp.Lock()

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self.request_counter += 1
            task_id = f"gw{self.gunicorn_worker_id}-sp{self.sub_process_id}-req{self.request_counter}"
        try:
            corr_id = get_correlation_id()
        except Exception:
            corr_id = None
        self.task_queue.put({"task_id": task_id, "payload": payload, "correlation_id": corr_id})

        t0 = time.time()
        while True:
            try:
                msg = self.result_queue.get(timeout=5)
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") == "RESULT" and msg.get("task_id") == task_id:
                    return msg.get("result", {})
                if msg.get("type") == "ERROR" and msg.get("task_id") == task_id:
                    raise RuntimeError(msg.get("error", "Unknown error"))
            except mp.queues.Empty:
                if time.time()-t0 > 36000:  # 10 hours safety
                    raise TimeoutError("WhisSent task timed out")
                continue


# Global instance
_whissent_worker_state_instance: Optional[WhisSentWorkerState] = None


def initialize_whissent():
    global _whissent_worker_state_instance
    if _whissent_worker_state_instance is not None:
        logger.info("WhisSent already initialized")
        return
    try:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    worker_id = int(os.environ.get("WORKER_ID", "0"))
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    _whissent_worker_state_instance = WhisSentWorkerState(worker_id, gpu_id)
    _whissent_worker_state_instance.initialize()


def infer_whissent(
    session_id: str,
    audio: Dict[str, Any],
    return_timestamps: bool = True,
    task: str = "transcribe",
    language: Optional[str] = None,
    emotion_top_k: int = 3,
) -> Tuple[Dict[str, Any], int]:
    if _whissent_worker_state_instance is None or _whissent_worker_state_instance.wrapper is None:
        raise RuntimeError("WhisSent model is not available.")

    payload = {
        "audio": audio,
        "return_timestamps": return_timestamps,
        "task": task,
        "language": language,
        "emotion_top_k": emotion_top_k,
    }
    result = _whissent_worker_state_instance.wrapper.invoke(payload)
    # For consistency, also mirror transcript as 'response'
    if isinstance(result, dict) and 'transcript' in result:
        result.setdefault('response', result['transcript'])
    return result, int(_whissent_worker_state_instance.gpu_id_for_worker)


def _cleanup_whissent_on_exit():
    global _whissent_worker_state_instance
    if _whissent_worker_state_instance is not None:
        _whissent_worker_state_instance.cleanup()
        _whissent_worker_state_instance = None

atexit.register(_cleanup_whissent_on_exit)


def _signal_handler(signum, frame):
    logger.info(f"WhisSent worker received signal {signum}, cleaning up...")
    _cleanup_whissent_on_exit()
    sys.exit(0)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
