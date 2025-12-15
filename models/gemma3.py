# gemma3.py - vLLM AsyncLLMEngine version with batching
# Author: Zied Mustapha
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from transformers import AutoTokenizer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import torch
import os
import asyncio
import multiprocessing as mp
from multiprocessing import Process, Queue
import queue
import time
import logging
import threading
import uuid
import gc
import atexit
from logging_config import setup_logging_for_process, set_correlation_id, clear_correlation_id, get_correlation_id

# For image processing
from PIL import Image
import requests
import io
from pdf2image import convert_from_path, convert_from_bytes
from typing import Optional, Union, List, Tuple, Dict, Any
import base64
import signal
import sys

# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_NAME = "unsloth/gemma-3-12b-it"  # Correct model name for vLLM
DEFAULT_MAX_NEW_TOKENS = 200000
DEFAULT_MAX_NEW_TOKENS_MULTI = 400000

# --- CRITICAL: Set environment variables for stable multi-process inference ---
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
# Force V0 engine - V1 engine (vLLM 0.8.x default) has stricter memory checks
os.environ['VLLM_USE_V1'] = '0'

# --- Dynamic GPU memory allocation ---
# GPU_MEMORY_OVERRIDE can be set to force a specific value (for testing)
def get_dynamic_gpu_memory_utilization():
    """Calculate gpu_memory_utilization based on how many models share this GPU."""
    # Check for explicit override (for sweet spot testing)
    override = os.environ.get("GPU_MEMORY_OVERRIDE")
    if override:
        try:
            val = float(override)
            logger.info(f"Using GPU_MEMORY_OVERRIDE={val}")
            return max(0.1, min(0.95, val))
        except ValueError:
            pass
    
    models_on_gpu = int(os.environ.get("MODELS_ON_GPU", "1"))
    available_fraction = 0.95
    utilization = available_fraction / models_on_gpu
    return max(0.1, min(0.95, utilization))


def get_vllm_engine_config() -> Dict[str, Any]:
    """
    Get vLLM engine configuration from environment variables.
    Allows tuning without code changes for H100 optimization.
    
    Note: Gemma3 (12B) is a DENSE model (not MoE), so no MoE config needed.
    
    Environment variables:
        GEMMA_MAX_MODEL_LEN: Maximum sequence length (default: 10000)
        GEMMA_MAX_NUM_BATCHED_TOKENS: Max tokens per scheduler step (default: 16384)
        GEMMA_MAX_NUM_SEQS: Max concurrent sequences (default: 256)
        GEMMA_ENABLE_CHUNKED_PREFILL: Enable chunked prefill (default: true)
        GEMMA_ENABLE_PREFIX_CACHING: Enable prefix/prompt caching (default: false)
    """
    config = {
        "max_model_len": int(os.environ.get("GEMMA_MAX_MODEL_LEN", "10000")),
        "max_num_batched_tokens": int(os.environ.get("GEMMA_MAX_NUM_BATCHED_TOKENS", "16384")),
        "max_num_seqs": int(os.environ.get("GEMMA_MAX_NUM_SEQS", "256")),
        "enable_chunked_prefill": os.environ.get("GEMMA_ENABLE_CHUNKED_PREFILL", "true").lower() in ("true", "1", "yes"),
        "enable_prefix_caching": os.environ.get("GEMMA_ENABLE_PREFIX_CACHING", "false").lower() in ("true", "1", "yes"),
        "coalesce_ms": float(os.environ.get("GEMMA_COALESCE_MS", "5")),
    }
    return config


def _blocking_queue_get(q: Queue, timeout: float = 0.1):
    """Blocking get with timeout for async integration."""
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return "EMPTY_SENTINEL"


# --- Session histories (in-memory, per-worker, for TEXT-ONLY chats) ---
session_histories: dict[str, list[dict]] = {}

def get_history(sid: str) -> list[dict]:
    """Retrieves or creates a session history for text-only chats."""
    return session_histories.setdefault(sid, [])

# --- Utilities ---
def _ensure_int_tokens(val: Any,
                       default_val: int,
                       logger_instance: Optional[logging.Logger] = None,
                       name: str = "max_new_tokens") -> int:
    """Safely coerce token count to a positive int.

    Falls back to default_val on invalid input and logs a warning.
    """
    log = logger_instance if logger_instance is not None else logger
    try:
        if val is None:
            return int(default_val)
        # Prevent True/False being treated as 1/0
        if isinstance(val, bool):
            raise ValueError("boolean not allowed")
        coerced = int(val)
        if coerced < 1:
            raise ValueError("must be >= 1")
        return coerced
    except Exception as e:
        try:
            log.warning(f"Invalid {name}='{val}' ({e}); using default {default_val}.")
        except Exception:
            pass
        return int(default_val)

# --- Image Processing Utility (unchanged) ---
def process_image_internal(image_input: Union[str, Image.Image, bytes, Dict[str, Any]], 
                          worker_logger: logging.Logger) -> Union[Image.Image, List[Image.Image]]:
    """Process different types of image inputs, including PDFs."""
    dpi = 600  # High DPI for all images
    
    # Add this helper function at the beginning
    def enhance_image(img: Image.Image) -> Image.Image:
        """Enhance image contrast for better OCR."""
        from PIL import ImageEnhance, ImageOps
        
        # Convert to RGB if not already
        img = img.convert("RGB")
        
        # Convert to grayscale for better text recognition
        img_gray = ImageOps.grayscale(img)
        
        # Auto-contrast
        img_gray = ImageOps.autocontrast(img_gray)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img_gray)
        img_enhanced = enhancer.enhance(1.5)
        
        # Convert back to RGB for the model
        return img_enhanced.convert("RGB")
    
    try:
        # Handle dictionary input with base64 data
        if isinstance(image_input, dict):
            if 'data' in image_input:
                worker_logger.info(f"Processing base64 image from dictionary...")
                try:
                    image_bytes = base64.b64decode(image_input['data'])
                    if image_bytes.startswith(b'%PDF'):
                        worker_logger.info(f"Processing PDF from base64 dictionary at {dpi} DPI.")
                        images = convert_from_bytes(image_bytes, dpi=dpi)
                        # ENHANCE HERE for PDFs
                        return [enhance_image(img) for img in images]
                    else:
                        image = Image.open(io.BytesIO(image_bytes))
                        # ENHANCE HERE for regular images
                        return enhance_image(image)
                except Exception as e:
                    worker_logger.error(f"Error decoding base64 from dictionary: {e}")
                    raise ValueError(f"Invalid base64 data in dictionary: {e}")
            else:
                raise ValueError("Dictionary image input must have 'data' key with base64 content")
                
        elif isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                response = requests.get(image_input, timeout=10)
                response.raise_for_status()
                content = response.content
                is_pdf_content = response.headers.get('content-type') == 'application/pdf'
                is_pdf_extension = image_input.lower().endswith('.pdf')

                if is_pdf_content or is_pdf_extension:
                    worker_logger.info(f"Processing PDF from URL: {image_input[:80]}... at {dpi} DPI")
                    images = convert_from_bytes(content, dpi=dpi)
                    # ENHANCE HERE for PDFs
                    return [enhance_image(img) for img in images]
                else:
                    image = Image.open(io.BytesIO(content))
                    # ENHANCE HERE for regular images
                    return enhance_image(image)
                    
            elif image_input.startswith('data:image'):
                worker_logger.info(f"Processing Base64 image data...")
                header, encoded = image_input.split(',', 1)
                image_bytes = io.BytesIO(base64.b64decode(encoded))
                if 'application/pdf' in header:
                    images = convert_from_bytes(image_bytes.getvalue(), dpi=dpi)
                    # ENHANCE HERE for PDFs
                    return [enhance_image(img) for img in images]
                else:
                    image = Image.open(image_bytes)
                    # ENHANCE HERE for regular images
                    return enhance_image(image)
                    
            else:  # Local file path
                if image_input.lower().endswith('.pdf'):
                    worker_logger.info(f"Processing PDF from local path: {image_input} at {dpi} DPI")
                    images = convert_from_path(image_input, dpi=dpi, poppler_path=os.getenv("POPPLER_PATH"))
                    # ENHANCE HERE for PDFs
                    return [enhance_image(img) for img in images]
                else:
                    image = Image.open(image_input)
                    # ENHANCE HERE for regular images
                    return enhance_image(image)
                    
        elif isinstance(image_input, bytes):
            if image_input.startswith(b'%PDF'):
                worker_logger.info(f"Processing PDF from raw bytes at {dpi} DPI.")
                images = convert_from_bytes(image_input, dpi=dpi)
                # ENHANCE HERE for PDFs
                return [enhance_image(img) for img in images]
            else:
                image = Image.open(io.BytesIO(image_input))
                # ENHANCE HERE for regular images
                return enhance_image(image)
                
        elif isinstance(image_input, Image.Image):
            # ENHANCE HERE for PIL Image objects
            return enhance_image(image_input)
            
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
    except Exception as e:
        worker_logger.error(f"Error processing image input '{str(image_input)[:100]}...': {e}", exc_info=True)
        raise
    
# --- Async request processing (like qwen3.py) ---
async def process_single_request(
    engine: AsyncLLMEngine,
    task: Dict[str, Any],
    result_queue: Queue,
    worker_id: int,
    wlog: logging.Logger
):
    """Process a single request - vLLM batches these internally at the token level."""
    task_id = task.get("task_id", str(uuid.uuid4()))
    correlation_id = task.get("correlation_id")
    task_type = task.get("task_type", "text")
    task_payload = task.get("payload", {})
    req_max_new_tokens = task.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    
    if correlation_id:
        try:
            set_correlation_id(correlation_id)
        except Exception:
            pass
    
    try:
        # Build prompt based on task type
        if task_type == "multimodal":
            user_text = task_payload.get("user_text", "")
            image_inputs_raw = task_payload.get("images", [])
            pdf_page_to_use = task_payload.get("pdf_page_to_use")
            
            # Process images
            processed_images = []
            if image_inputs_raw:
                for idx, img_item_raw in enumerate(image_inputs_raw):
                    try:
                        processed_img_or_list = process_image_internal(img_item_raw, wlog)
                        if isinstance(processed_img_or_list, list):
                            if processed_img_or_list:
                                page_idx = pdf_page_to_use if pdf_page_to_use is not None and 0 <= pdf_page_to_use < len(processed_img_or_list) else 0
                                processed_images.append(processed_img_or_list[page_idx])
                        else:
                            processed_images.append(processed_img_or_list)
                    except Exception as e:
                        wlog.error(f"[{task_id}] Failed to process image {idx}: {e}")
            
            # Build chat template
            user_content = [{"type": "image"} for _ in processed_images]
            user_content.append({"type": "text", "text": user_text})
            messages = [{"role": "user", "content": user_content}]
            
            # Get tokenizer from engine
            tokenizer = await engine.get_tokenizer()
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": processed_images[0] if len(processed_images) == 1 else processed_images}
            }
        else:
            # Text task
            langchain_messages = task_payload.get("langchain_messages", [])
            chat_history = []
            for msg in langchain_messages:
                if isinstance(msg, HumanMessage):
                    role = "user"
                elif isinstance(msg, AIMessage):
                    role = "model"
                elif isinstance(msg, SystemMessage):
                    role = "system"
                else:
                    continue
                
                # Gemma3 requires alternating roles - merge consecutive same-role messages
                if chat_history and chat_history[-1]["role"] == role:
                    chat_history[-1]["content"] += "\n" + msg.content
                else:
                    chat_history.append({"role": role, "content": msg.content})
            
            # Ensure we have at least one user message
            if not chat_history or chat_history[-1]["role"] != "user":
                wlog.warning(f"[{task_id}] Chat history doesn't end with user message, adding empty user turn")
                if not any(m["role"] == "user" for m in chat_history):
                    chat_history.append({"role": "user", "content": "Hello"})
            
            tokenizer = await engine.get_tokenizer()
            prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
            inputs = prompt
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=req_max_new_tokens,
            repetition_penalty=1.15,
        )
        
        request_id = f"gemma-{task_id}"
        gen_start = time.time()
        
        # Collect output via async generator
        final_output = None
        async for output in engine.generate(inputs, sampling_params, request_id):
            final_output = output
        
        gen_duration = time.time() - gen_start
        
        if final_output and final_output.outputs:
            response_text = final_output.outputs[0].text.strip()
            num_tokens = len(final_output.outputs[0].token_ids)
            tps = num_tokens / gen_duration if gen_duration > 0 else 0
            wlog.info(f"[{task_id}] Completed in {gen_duration:.2f}s, {num_tokens} tokens ({tps:.1f} tok/s)")
            result_queue.put((task_id, response_text, None, req_max_new_tokens, task_type))
        else:
            result_queue.put((task_id, None, "No output generated", req_max_new_tokens, task_type))
            
    except Exception as e:
        wlog.error(f"[{task_id}] Error: {e}", exc_info=True)
        result_queue.put((task_id, None, str(e), req_max_new_tokens, task_type))
    finally:
        try:
            clear_correlation_id()
        except Exception:
            pass


async def request_dispatcher(
    task_queue: Queue,
    engine: AsyncLLMEngine,
    result_queue: Queue,
    worker_id: int,
    wlog: logging.Logger,
    coalesce_ms: float = 5.0
):
    """Dispatch requests to vLLM with coalescing for better batching."""
    loop = asyncio.get_event_loop()
    active_tasks: set = set()
    
    wlog.info(f"Request dispatcher started - burst-aware mode (coalesce={coalesce_ms}ms)")
    
    def cleanup_done_tasks():
        done = {t for t in active_tasks if t.done()}
        for t in done:
            try:
                t.result()
            except Exception as e:
                wlog.error(f"Task exception: {e}")
        active_tasks.difference_update(done)
    
    while True:
        try:
            cleanup_done_tasks()
            
            # Wait for at least one task
            task = await loop.run_in_executor(None, _blocking_queue_get, task_queue, 0.05)
            
            if task == "EMPTY_SENTINEL":
                continue
            
            if task is None:
                wlog.info("Shutdown signal received")
                break
            
            # Coalescing window
            if coalesce_ms > 0:
                await asyncio.sleep(coalesce_ms / 1000.0)
            
            # Collect batch
            batch = [task]
            while len(batch) < 512:
                try:
                    t = task_queue.get_nowait()
                    if t is None:
                        task_queue.put(None)
                        break
                    batch.append(t)
                except queue.Empty:
                    break
            
            # Submit all to vLLM
            for t in batch:
                coro = process_single_request(engine, t, result_queue, worker_id, wlog)
                async_task = asyncio.create_task(coro)
                active_tasks.add(async_task)
            
            if len(batch) > 1:
                wlog.info(f"Dispatched burst of {len(batch)} tasks - {len(active_tasks)} active")
            
            await asyncio.sleep(0)
            
        except Exception as e:
            wlog.error(f"Dispatcher error: {e}", exc_info=True)
            continue
    
    if active_tasks:
        wlog.info(f"Waiting for {len(active_tasks)} active tasks...")
        await asyncio.gather(*active_tasks, return_exceptions=True)
    
    wlog.info("Request dispatcher stopped")


# --- Concurrent model worker (AsyncLLMEngine version) ---
def concurrent_model_worker(task_queue: Queue, result_queue: Queue, worker_id: int, gpu_id: str, process_id_str: str):
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    os.environ['VLLM_USE_V1'] = '0'
    
    os.environ.setdefault("ROLE", "worker_subproc")
    os.environ.setdefault("SERVICE_NAME", "orchestra")
    setup_logging_for_process(f"workerproc_{worker_id}_gemma3_{process_id_str}.jsonl")
    wlog = logging.getLogger(f"gemma_worker_proc_{process_id_str}")

    wlog.info(f"Gemma3 AsyncLLMEngine worker {worker_id} (PID: {os.getpid()}) starting on GPU {gpu_id}")
    
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
            wlog.info(f"CUDA device set to cuda:0 (physical GPU {gpu_id})")
        except Exception as e:
            wlog.error(f"CUDA setup failed: {e}", exc_info=True)
            result_queue.put(("ERROR_INIT", None, f"CUDA setup failed: {e}", None, None))
            return
    
    engine = None
    
    try:
        vllm_config = get_vllm_engine_config()
        gpu_mem_util = get_dynamic_gpu_memory_utilization()
        wlog.info(
            f"vLLM config: gpu_mem={gpu_mem_util:.2f}, "
            f"max_model_len={vllm_config['max_model_len']}, "
            f"max_num_batched_tokens={vllm_config['max_num_batched_tokens']}, "
            f"max_num_seqs={vllm_config['max_num_seqs']}, "
            f"chunked_prefill={vllm_config['enable_chunked_prefill']}, "
            f"coalesce_ms={vllm_config['coalesce_ms']}"
        )
        
        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=vllm_config["max_model_len"],
            max_num_batched_tokens=vllm_config["max_num_batched_tokens"],
            max_num_seqs=vllm_config["max_num_seqs"],
            enable_chunked_prefill=vllm_config["enable_chunked_prefill"],
            enable_prefix_caching=vllm_config["enable_prefix_caching"],
            disable_log_requests=False,
            enforce_eager=False,
            dtype="auto",
        )
        
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        wlog.info(f"AsyncLLMEngine loaded successfully on GPU {gpu_id}")
        
    except Exception as e:
        wlog.error(f"Failed to load AsyncLLMEngine: {e}", exc_info=True)
        result_queue.put(("ERROR_INIT", None, f"Model load failed: {e}", None, None))
        return
    
    result_queue.put(("READY", None, None, None, None))
    wlog.info(f"Worker {worker_id} is READY")
    
    # Run async dispatcher
    try:
        asyncio.run(request_dispatcher(
            task_queue, engine, result_queue, worker_id, wlog,
            coalesce_ms=vllm_config["coalesce_ms"]
        ))
    except Exception as e:
        wlog.error(f"Dispatcher crashed: {e}", exc_info=True)
    
    # Cleanup
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    wlog.info(f"Worker {worker_id} shut down.")

# --- Worker State Manager (unchanged structure) ---
class ConcurrentWorkerState:
    def __init__(self, worker_id_gunicorn: int, assigned_gpu: str, use_sub_process: bool):
        self.gunicorn_worker_id = worker_id_gunicorn
        self.gpu_id_for_worker = assigned_gpu
        self.enable_sub_process_model = use_sub_process
        self.unique_process_id_str = str(uuid.uuid4())[:8]
        
        self.chat_model_instance = None
        self.model_task_queue = None
        self.model_result_queue = None
        self.model_worker_process_handle = None
        
        # For direct load mode (simplified for vLLM)
        self.direct_llm = None
        self.direct_tokenizer = None

        logger.info(
            f"GunicornWorker-{self.gunicorn_worker_id}: Creating Gemma WorkerState. "
            f"Assigned GPU: {self.gpu_id_for_worker}, Sub-process model: {self.enable_sub_process_model}, "
            f"Sub-process ID if used: {self.unique_process_id_str}"
        )
    
    def initialize(self):
        """Initializes the model, either directly or in a sub-process."""
        if self.enable_sub_process_model:
            logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Starting dedicated model sub-process {self.unique_process_id_str} on GPU {self.gpu_id_for_worker}.")
            
            mp_context = mp.get_context('spawn')
            self.model_task_queue = mp_context.Queue()
            self.model_result_queue = mp_context.Queue()
            
            self.model_worker_process_handle = mp_context.Process(
                target=concurrent_model_worker,
                args=(self.model_task_queue, self.model_result_queue, 
                      self.gunicorn_worker_id, self.gpu_id_for_worker, self.unique_process_id_str),
                daemon=False,  # Changed to False to allow vLLM to spawn child processes
                name=f"gemma_model_proc_{self.unique_process_id_str}"
            )
            self.model_worker_process_handle.start()
            
            logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Waiting for model sub-process to be ready...")
            readiness_timeout_seconds = 600
            start_wait_time = time.time()
            is_ready = False
            
            while not is_ready and (time.time() - start_wait_time) < readiness_timeout_seconds:
                if not self.model_worker_process_handle.is_alive():
                    exit_code = self.model_worker_process_handle.exitcode
                    msg = f"Model sub-process {self.unique_process_id_str} died (exit code: {exit_code})."
                    logger.error(msg)
                    raise RuntimeError(msg)
                try:
                    msg_type_or_task_id, _, error_str, _, _ = self.model_result_queue.get(timeout=2)
                    if msg_type_or_task_id == "READY":
                        is_ready = True
                        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Model sub-process is READY.")
                    elif msg_type_or_task_id == "ERROR_INIT":
                        msg = f"Model sub-process init error: {error_str}"
                        logger.error(msg)
                        raise RuntimeError(msg)
                except mp.queues.Empty:
                    continue
            
            if not is_ready:
                msg = f"Model sub-process timed out after {readiness_timeout_seconds}s."
                logger.error(msg)
                if self.model_worker_process_handle.is_alive():
                    self.model_worker_process_handle.terminate()
                    self.model_worker_process_handle.join(timeout=10)
                raise RuntimeError(msg)

            self.chat_model_instance = ConcurrentModelWrapper(
                self.model_task_queue, self.model_result_queue, self.gunicorn_worker_id, self.unique_process_id_str
            )
        else:
            # Direct load mode - deprecated, always use subprocess for async batching
            logger.warning(f"GunicornWorker-{self.gunicorn_worker_id}: Direct load mode deprecated. Use subprocess mode for async batching.")
            raise RuntimeError("Direct load mode is deprecated. Set use_sub_process=True for async batching support.")

    def cleanup(self):
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Cleaning up Gemma WorkerState...")
        
        # Stop the result router thread in the wrapper first
        if self.chat_model_instance and hasattr(self.chat_model_instance, 'stop'):
            try:
                self.chat_model_instance.stop()
            except Exception as e:
                logger.error(f"Error stopping result router: {e}")
        
        if self.enable_sub_process_model and self.model_worker_process_handle:
            if self.model_worker_process_handle.is_alive():
                logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Signaling model sub-process to shutdown...")
                try:
                    if self.model_task_queue:
                        self.model_task_queue.put(None)
                    self.model_worker_process_handle.join(timeout=30)
                    if self.model_worker_process_handle.is_alive():
                        logger.warning(f"Model sub-process did not exit gracefully, terminating.")
                        self.model_worker_process_handle.terminate()
                        self.model_worker_process_handle.join(timeout=10)
                        if self.model_worker_process_handle.is_alive():
                            logger.error(f"Force killing sub-process")
                            self.model_worker_process_handle.kill()
                            self.model_worker_process_handle.join(timeout=5)
                except Exception as e:
                    logger.error(f"Error during sub-process cleanup: {e}", exc_info=True)
                    # Force kill if all else fails
                    try:
                        self.model_worker_process_handle.kill()
                    except:
                        pass
        

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Cleanup complete.")

# --- Model Wrapper Classes ---
class ConcurrentModelWrapper:
    """Wrapper for sub-process model communication with proper concurrent request handling."""
    def __init__(self, task_q: Queue, result_q: Queue, gunicorn_id: int, sub_proc_id: str):
        self.task_queue = task_q
        self.result_queue = result_q
        self.gunicorn_worker_id = gunicorn_id
        self.sub_process_id = sub_proc_id
        self.request_counter = 0
        self.counter_lock = threading.Lock()
        
        # Concurrent request handling
        self._pending_tasks: Dict[str, Dict] = {}  # task_id -> {"event": Event, "result": tuple}
        self._pending_lock = threading.Lock()
        self._router_running = True
        self._result_router_thread = threading.Thread(
            target=self._result_router,
            daemon=True,
            name=f"gemma_result_router_{gunicorn_id}"
        )
        self._result_router_thread.start()
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Result router thread started for concurrent requests")
    
    def _result_router(self):
        """Background thread that routes results to pending tasks."""
        while self._router_running:
            try:
                result = self.result_queue.get(timeout=1.0)
                if result is None:
                    continue
                
                res_task_id = result[0] if isinstance(result, tuple) else None
                if res_task_id is None:
                    logger.warning(f"Received result without task_id")
                    continue
                
                with self._pending_lock:
                    if res_task_id in self._pending_tasks:
                        self._pending_tasks[res_task_id]["result"] = result
                        self._pending_tasks[res_task_id]["event"].set()
                    else:
                        logger.warning(f"Received result for unknown task_id: {res_task_id}")
                        
            except Exception as e:
                if self._router_running and "Empty" not in str(type(e).__name__):
                    logger.debug(f"Result router: {e}")
    
    def stop(self):
        """Stop the result router thread."""
        self._router_running = False
        if self._result_router_thread and self._result_router_thread.is_alive():
            self._result_router_thread.join(timeout=5)

    def invoke(self, task_payload: Dict[str, Any], task_type: str, **kwargs) -> AIMessage:
        has_images = task_type == "multimodal" and "images" in task_payload and task_payload["images"]
        default_for_task = DEFAULT_MAX_NEW_TOKENS_MULTI if has_images else DEFAULT_MAX_NEW_TOKENS
        req_max_new_tokens = _ensure_int_tokens(
            kwargs.get("max_new_tokens", default_for_task),
            default_for_task,
            logger,
            "max_new_tokens",
        )
        
        with self.counter_lock:
            self.request_counter += 1
            current_task_id = f"gw{self.gunicorn_worker_id}-sp{self.sub_process_id}-req{self.request_counter}"
        
        logger.debug(f"[{current_task_id}] Sending task (type: {task_type}) to sub-process.")
        try:
            corr_id = get_correlation_id()
        except Exception:
            corr_id = None
        
        # Register this task for result routing
        event = threading.Event()
        with self._pending_lock:
            self._pending_tasks[current_task_id] = {"event": event, "result": None}
        
        try:
            # Send task as dict for async dispatcher
            task_dict = {
                "task_id": current_task_id,
                "payload": task_payload,
                "max_new_tokens": req_max_new_tokens,
                "task_type": task_type,
                "correlation_id": corr_id,
            }
            self.task_queue.put(task_dict)
            
            # Wait for result (the result router thread will signal us)
            result_timeout_seconds = 600  # 10 minutes
            if not event.wait(timeout=result_timeout_seconds):
                raise TimeoutError(f"Task {current_task_id} timed out after {result_timeout_seconds}s.")
            
            # Get result
            with self._pending_lock:
                task_info = self._pending_tasks.pop(current_task_id, None)
            
            if task_info is None or task_info["result"] is None:
                raise Exception(f"Task {current_task_id} result not found")
            
            res_task_id, response_content, error_msg, _, _ = task_info["result"]
            
            if error_msg:
                logger.error(f"[{current_task_id}] Model sub-process error: {error_msg}")
                raise Exception(f"Model generation error: {error_msg}")
            
            logger.debug(f"[{current_task_id}] Received response from sub-process.")
            return AIMessage(content=response_content)
            
        except Exception as e:
            # Clean up pending task on error
            with self._pending_lock:
                self._pending_tasks.pop(current_task_id, None)
            raise

# Global instance
_gemma_worker_state_instance: Optional[ConcurrentWorkerState] = None

def initialize_gemma():
    """Initialize the Gemma model (unchanged logic)."""
    global _gemma_worker_state_instance
    if _gemma_worker_state_instance is not None:
        logger.warning(f"GunicornWorker-{os.environ.get('GUNICORN_WORKER_ID', 'N/A')}: Gemma already initialized.")
        return

    try:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn', force=True)
        logger.info(f"Multiprocessing start method: {mp.get_start_method()}")
    except RuntimeError as e:
        logger.info(f"Multiprocessing start method already set: {e}")

    gunicorn_ord_id = int(os.environ.get("GUNICORN_WORKER_ID", "0"))
    assigned_gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    total_gemma_instances_requested = int(os.environ.get("GEMMA3_INSTANCES", "1"))
    num_available_gpus = int(os.environ.get("NUM_GPUS", "1"))
    
    should_use_sub_process_for_model = True  # Keep as before
    
    logger.info(
        f"GunicornWorker-{gunicorn_ord_id}: Initializing Gemma with vLLM. "
        f"Assigned GPU: {assigned_gpu_id}. "
        f"Total Instances: {total_gemma_instances_requested}, GPUs: {num_available_gpus}. "
        f"Sub-Process: {should_use_sub_process_for_model}."
    )
    
    _gemma_worker_state_instance = ConcurrentWorkerState(
        worker_id_gunicorn=gunicorn_ord_id,
        assigned_gpu=assigned_gpu_id,
        use_sub_process=should_use_sub_process_for_model
    )
    
    try:
        _gemma_worker_state_instance.initialize()
        logger.info(f"GunicornWorker-{gunicorn_ord_id}: Gemma vLLM model initialized.")
    except Exception as e:
        logger.error(f"GunicornWorker-{gunicorn_ord_id}: FATAL error during initialization: {e}", exc_info=True)
        _gemma_worker_state_instance = None
        raise

# --- Main Chat Functions (unchanged interface) ---
def chat_gemma3(
    session_id: str, 
    user_input_text: str, 
    image_inputs: Optional[List[Union[str, bytes, Dict[str, Any]]]] = None,
    pdf_page_to_use: Optional[int] = None,
    req_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
):
    """Main chat function - interface unchanged."""
    if _gemma_worker_state_instance is None or _gemma_worker_state_instance.chat_model_instance is None:
        gunicorn_id = os.environ.get('GUNICORN_WORKER_ID', 'N/A')
        logger.error(f"GunicornWorker-{gunicorn_id} (Session: {session_id}): Model not initialized.")
        raise RuntimeError("Gemma model is not available.")

    gunicorn_id_log = _gemma_worker_state_instance.gunicorn_worker_id
    task_payload = {}
    task_type = ""
    
    effective_max_new_tokens = req_max_new_tokens
    if image_inputs and len(image_inputs) > 0:
        task_type = "multimodal"
        effective_max_new_tokens = req_max_new_tokens if req_max_new_tokens != DEFAULT_MAX_NEW_TOKENS else DEFAULT_MAX_NEW_TOKENS_MULTI
        task_payload = {
            "user_text": user_input_text,
            "images": image_inputs,
            "pdf_page_to_use": pdf_page_to_use
        }
        logger.info(f"GunicornWorker-{gunicorn_id_log} (Session: {session_id}): Invoking vLLM (MULTIMODAL).")
    else:
        task_type = "text"
        chat_turn_history = get_history(session_id)
        chat_turn_history.append({"role": "user", "content": user_input_text})
        
        langchain_message_history = []
        for turn in chat_turn_history:
            role, content = turn["role"], turn["content"]
            if role == "user":
                langchain_message_history.append(HumanMessage(content=content))
            elif role == "assistant" or role == "model":
                langchain_message_history.append(AIMessage(content=content))
            elif role == "system":
                langchain_message_history.append(SystemMessage(content=content))

        if not langchain_message_history or not isinstance(langchain_message_history[-1], HumanMessage):
            if chat_turn_history and chat_turn_history[-1]["role"] == "user":
                chat_turn_history.pop()
            raise ValueError("Chat history must end with a user message.")

        task_payload = {"langchain_messages": langchain_message_history}
        logger.info(f"GunicornWorker-{gunicorn_id_log} (Session: {session_id}): Invoking vLLM (TEXT-ONLY).")

    response_aimessage = _gemma_worker_state_instance.chat_model_instance.invoke(
        task_payload, task_type, max_new_tokens=effective_max_new_tokens
    )
    
    generated_response_text = response_aimessage.content
    
    if task_type == "text":
        chat_turn_history.append({"role": "assistant", "content": generated_response_text})
        MAX_HISTORY_ITEMS = 20
        if len(chat_turn_history) > MAX_HISTORY_ITEMS:
            session_histories[session_id] = chat_turn_history[-MAX_HISTORY_ITEMS:]

    return generated_response_text, int(_gemma_worker_state_instance.gpu_id_for_worker)

async def stream_chat_gemma3(session_id: str, user_input_text: str, 
                             image_inputs: Optional[List[Union[str, bytes, Dict[str, Any]]]] = None,
                             pdf_page_to_use: Optional[int] = None,
                             req_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS):
    """Streaming function - interface unchanged."""
    logger.warning(f"[{session_id}] stream_chat_gemma3 using non-streaming backend.")
    effective_max_tokens = req_max_new_tokens
    if image_inputs and len(image_inputs) > 0:
        effective_max_tokens = req_max_new_tokens if req_max_new_tokens != DEFAULT_MAX_NEW_TOKENS else DEFAULT_MAX_NEW_TOKENS_MULTI

    generated_text, gpu_id_int = chat_gemma3(
        session_id, user_input_text, image_inputs, pdf_page_to_use, effective_max_tokens
    )
    yield {"instance": gpu_id_int, "text": generated_text}

def _cleanup_gemma_resources_on_exit():
    """Cleanup function."""
    global _gemma_worker_state_instance
    gunicorn_id = os.environ.get('GUNICORN_WORKER_ID', 'N/A')
    logger.info(f"GunicornWorker-{gunicorn_id}: Running atexit cleanup...")
    if _gemma_worker_state_instance is not None:
        try:
            _gemma_worker_state_instance.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
        _gemma_worker_state_instance = None

atexit.register(_cleanup_gemma_resources_on_exit)

# Signal handlers for graceful shutdown
def _signal_handler(signum, frame):
    """Handle termination signals"""
    logger.info(f"Received signal {signum}, initiating cleanup...")
    _cleanup_gemma_resources_on_exit()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)