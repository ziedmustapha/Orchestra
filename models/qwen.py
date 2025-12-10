# models/qwen.py - vLLM version for Qwen-VL
# Author: Zied Mustapha
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from langchain_core.messages import AIMessage

import torch
import os
import math
import asyncio
import multiprocessing as mp
from multiprocessing import Process, Queue
import time
import logging
import threading
import uuid
import gc
import atexit
import signal
import sys
from typing import Optional, List, Union, Dict, Any
from logging_config import setup_logging_for_process, set_correlation_id, clear_correlation_id

# Import the generic image processor from gemma3 (or move it to a shared utils file)
# For simplicity, we'll just reference it here. Ensure gemma3.py is in the same 'models' package.
from .gemma3 import process_image_internal
from PIL import Image

# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Qwen Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 8000 # Qwen has a larger context
# Safer default for multimodal new tokens to reduce risk of context overflow
DEFAULT_MAX_NEW_TOKENS_MULTI = int(os.environ.get("QWEN_DEFAULT_MAX_NEW_TOKENS_MULTI", "2048"))
# Reserve a conservative budget of tokens per image to account for multimodal tokens added by vLLM
IMAGE_TOKEN_BUDGET = int(os.environ.get("QWEN_IMAGE_TOKENS_PER_IMAGE", "16000"))

# --- CRITICAL: Environment variables for stability ---
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
# Force V0 engine - V1 engine (vLLM 0.8.x default) has stricter memory checks
os.environ['VLLM_USE_V1'] = '0'

# Qwen is typically stateless for vision tasks, so no session history needed.

# --- Concurrent Qwen worker (vLLM version) ---
def concurrent_qwen_worker(task_queue: Queue, result_queue: Queue, worker_id: int, gpu_id: str, process_id_str: str):
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    # Structured JSON logging per sub-process
    os.environ.setdefault("ROLE", "worker_subproc")
    os.environ.setdefault("SERVICE_NAME", "orchestra")
    setup_logging_for_process(f"workerproc_{worker_id}_qwen_{process_id_str}.jsonl")
    worker_process_logger = logging.getLogger(f"qwen_worker_proc_{process_id_str}")

    worker_process_logger.info(f"Qwen worker {worker_id} (PID: {os.getpid()}) starting on GPU {gpu_id}")

    llm = None
    tokenizer = None

    try:
        worker_process_logger.info(f"Loading vLLM model '{MODEL_NAME}' for worker {worker_id}...")
        start_load_time = time.time()
        
        llm = LLM(
            model=MODEL_NAME,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            gpu_memory_utilization=0.3, # Vision models can be memory-intensive
            max_model_len=8192,
            max_num_seqs=8, # Adjust based on VRAM
            enable_chunked_prefill=True
        )
        
        tokenizer = llm.get_tokenizer()
        load_duration = time.time() - start_load_time
        worker_process_logger.info(f"Qwen model loaded in {load_duration:.2f}s for worker {worker_id}.")
        
    except Exception as e:
        load_duration = time.time() - start_load_time
        worker_process_logger.error(f"FAILED to load Qwen model after {load_duration:.2f}s: {e}", exc_info=True)
        result_queue.put(("ERROR_INIT", None, f"Model load failed: {e}", None, None))
        return

    result_queue.put(("READY", None, None, None, None))
    worker_process_logger.info(f"Qwen worker {worker_id} is READY.")

    while True:
        try:
            task = task_queue.get()
            if task is None:
                worker_process_logger.info(f"Qwen worker {worker_id} received shutdown signal.")
                break
            # Unpack allowing optional correlation id (4 or 5 tuple)
            if isinstance(task, tuple) and len(task) >= 4:
                if len(task) == 5:
                    task_id, task_payload, req_max_new_tokens, task_type, corr_id = task
                else:
                    task_id, task_payload, req_max_new_tokens, task_type = task
                    corr_id = None
            else:
                worker_process_logger.error(f"Unexpected task format: {type(task)} {task}")
                continue
            if corr_id:
                try:
                    set_correlation_id(corr_id)
                except Exception:
                    pass
            worker_process_logger.info(f"Qwen worker {worker_id} processing task {task_id} (type: {task_type})")
            
            inference_start_time = time.time()

            # Qwen is primarily used for multimodal tasks here
            if task_type != "multimodal":
                 raise ValueError("Qwen worker currently only supports 'multimodal' tasks.")

            user_text = task_payload.get("user_text", "")
            image_inputs_raw = task_payload.get("images", [])
            pdf_page_to_use = task_payload.get("pdf_page_to_use")
            if not image_inputs_raw:
                raise ValueError("Multimodal task for Qwen requires at least one image.")

            # Process image(s) using the shared utility
            processed_images = []
            worker_process_logger.info(f"[Task {task_id}] Processing {len(image_inputs_raw)} image item(s) for Qwen-VL.")
            for idx, img_item_raw in enumerate(image_inputs_raw):
                try:
                    processed_img_or_list = process_image_internal(img_item_raw, worker_process_logger)
                    if isinstance(processed_img_or_list, list):  # PDF pages
                        if not processed_img_or_list:
                            continue
                        page_idx = 0
                        if pdf_page_to_use is not None and 0 <= int(pdf_page_to_use) < len(processed_img_or_list):
                            page_idx = int(pdf_page_to_use)
                        processed_images.append(processed_img_or_list[page_idx])
                    else:
                        processed_images.append(processed_img_or_list)
                except Exception as e:
                    worker_process_logger.error(f"[Task {task_id}] Failed to process image index {idx}: {e}")

            if not processed_images:
                raise ValueError("No valid images after processing.")

            # Hard cap: if more than 5 images, reject as exceeding token budget
            try:
                max_images_allowed = int(os.environ.get("QWEN_MAX_IMAGES", "8"))
            except Exception:
                max_images_allowed = 5
            if len(processed_images) > max_images_allowed:
                err = (
                    f"TOO_MANY_IMAGES: images={len(processed_images)}, max_images={max_images_allowed}"
                )
                worker_process_logger.warning(f"[Task {task_id}] {err}")
                result_queue.put((task_id, None, err, req_max_new_tokens, task_type))
                continue

            # Build Qwen-VL prompt supporting multiple images: add one image token per image, followed by the text
            user_content = []
            for _ in processed_images:
                user_content.append({"type": "image"})
            user_content.append({"type": "text", "text": user_text})

            messages = [{"role": "user", "content": user_content}]
            
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Context-length pre-check disabled for ≤QWEN_MAX_IMAGES; rely on hard cap above and API timeout
            
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": processed_images[0] if len(processed_images) == 1 else processed_images}
            }

            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=req_max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")],
            )
            
            gen_start_time = time.time()
            outputs = llm.generate([inputs], sampling_params)
            gen_duration = time.time() - gen_start_time
            
            response_text = outputs[0].outputs[0].text.strip()
            num_generated_tokens = len(outputs[0].outputs[0].token_ids)
            tokens_per_second = num_generated_tokens / gen_duration if gen_duration > 0 else 0
            
            inference_duration = time.time() - inference_start_time
            worker_process_logger.info(
                f"Qwen worker {worker_id} completed task {task_id} in {inference_duration:.2f}s. "
                f"Generated {num_generated_tokens} tokens ({tokens_per_second:.2f} tokens/s)."
            )
            result_queue.put((task_id, response_text, None, req_max_new_tokens, task_type))

        except Exception as e:
            current_task_id_err = task_id if 'task_id' in locals() else "UNKNOWN_TASK"
            error_message = f"Qwen worker {worker_id} error on task {current_task_id_err}: {str(e)}"
            worker_process_logger.error(error_message, exc_info=True)
            m_tokens = req_max_new_tokens if 'req_max_new_tokens' in locals() else DEFAULT_MAX_NEW_TOKENS
            t_type = task_type if 'task_type' in locals() else "unknown"
            result_queue.put((current_task_id_err, None, error_message, m_tokens, t_type))
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            try:
                clear_correlation_id()
            except Exception:
                pass
            
    del llm, tokenizer
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    worker_process_logger.info(f"Qwen worker {worker_id} shut down.")

# --- The following classes are direct copies/adaptations from gemma3.py ---
# --- A better design would be a shared `base_worker.py` file, but this is simplest ---

class QwenWorkerState:
    def __init__(self, worker_id_gunicorn: int, assigned_gpu: str):
        self.gunicorn_worker_id = worker_id_gunicorn
        self.gpu_id_for_worker = assigned_gpu
        self.unique_process_id_str = str(uuid.uuid4())[:8]
        self.model_task_queue = None
        self.model_result_queue = None
        self.model_worker_process_handle = None
        self.chat_model_instance = None
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Creating Qwen WorkerState on GPU {self.gpu_id_for_worker}")

    def initialize(self):
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Starting Qwen sub-process {self.unique_process_id_str} on GPU {self.gpu_id_for_worker}.")
        mp_context = mp.get_context('spawn')
        self.model_task_queue = mp_context.Queue()
        self.model_result_queue = mp_context.Queue()
        
        self.model_worker_process_handle = mp_context.Process(
            target=concurrent_qwen_worker,
            args=(self.model_task_queue, self.model_result_queue, self.gunicorn_worker_id, self.gpu_id_for_worker, self.unique_process_id_str),
            daemon=False,
            name=f"qwen_model_proc_{self.unique_process_id_str}"
        )
        self.model_worker_process_handle.start()
        
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Waiting for Qwen sub-process to be ready...")
        try:
            msg_type, _, error_str, _, _ = self.model_result_queue.get(timeout=600)
            if msg_type == "READY":
                logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Qwen sub-process is READY.")
            elif msg_type == "ERROR_INIT":
                raise RuntimeError(f"Qwen sub-process init error: {error_str}")
        except mp.queues.Empty:
            raise RuntimeError("Qwen sub-process timed out during initialization.")

        # Re-using the same wrapper class logic
        from .gemma3 import ConcurrentModelWrapper
        self.chat_model_instance = ConcurrentModelWrapper(self.model_task_queue, self.model_result_queue, self.gunicorn_worker_id, self.unique_process_id_str)

    def cleanup(self):
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Cleaning up Qwen WorkerState...")
        if self.model_worker_process_handle and self.model_worker_process_handle.is_alive():
            logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Signaling Qwen sub-process to shutdown...")
            try:
                if self.model_task_queue: self.model_task_queue.put(None)
                self.model_worker_process_handle.join(timeout=30)
                if self.model_worker_process_handle.is_alive():
                    self.model_worker_process_handle.terminate()
            except Exception as e:
                logger.error(f"Error during Qwen sub-process cleanup: {e}", exc_info=True)
        gc.collect()
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Qwen cleanup complete.")

_qwen_worker_state_instance: Optional[QwenWorkerState] = None

def initialize_qwen():
    global _qwen_worker_state_instance
    if _qwen_worker_state_instance is not None:
        return

    try:
        if mp.get_start_method(allow_none=True) is None: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    gunicorn_ord_id = int(os.environ.get("WORKER_ID", "0"))
    assigned_gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    
    logger.info(f"GunicornWorker-{gunicorn_ord_id}: Initializing Qwen-VL. Assigned GPU: {assigned_gpu_id}.")
    
    _qwen_worker_state_instance = QwenWorkerState(worker_id_gunicorn=gunicorn_ord_id, assigned_gpu=assigned_gpu_id)
    
    try:
        _qwen_worker_state_instance.initialize()
    except Exception as e:
        logger.error(f"GunicornWorker-{gunicorn_ord_id}: FATAL error during Qwen initialization: {e}", exc_info=True)
        _qwen_worker_state_instance = None
        raise

def chat_qwen(
    session_id: str, 
    user_input_text: str, 
    image_inputs: Optional[List[Union[str, bytes, Dict[str, Any]]]],
    req_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    pdf_page_to_use: Optional[int] = None,
):
    if _qwen_worker_state_instance is None or _qwen_worker_state_instance.chat_model_instance is None:
        raise RuntimeError("Qwen model is not available.")

    # Qwen-VL is primarily for multimodal, so we enforce that
    task_type = "multimodal"
    task_payload = {
        "user_text": user_input_text,
        "images": image_inputs,
        "pdf_page_to_use": pdf_page_to_use,
    }
    
    response_aimessage = _qwen_worker_state_instance.chat_model_instance.invoke(
        task_payload, task_type, max_new_tokens=req_max_new_tokens
    )
    
    return response_aimessage.content, int(_qwen_worker_state_instance.gpu_id_for_worker)

def _cleanup_qwen_resources_on_exit():
    global _qwen_worker_state_instance
    if _qwen_worker_state_instance is not None:
        _qwen_worker_state_instance.cleanup()
        _qwen_worker_state_instance = None

atexit.register(_cleanup_qwen_resources_on_exit)

def _signal_handler(signum, frame):
    logger.info(f"Qwen worker received signal {signum}, cleaning up...")
    _cleanup_qwen_resources_on_exit()
    sys.exit(0)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)