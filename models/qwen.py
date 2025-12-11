# models/qwen.py - Qwen-VL vLLM implementation with AsyncLLMEngine
# Author: Zied Mustapha
# Refactored to use AsyncLLMEngine for automatic continuous batching

from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.distributed.parallel_state import destroy_model_parallel
from transformers import AutoTokenizer

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
import signal
import sys
from typing import Optional, List, Union, Dict, Any
from logging_config import setup_logging_for_process, set_correlation_id, clear_correlation_id, get_correlation_id

from .gemma3 import process_image_internal

logger = logging.getLogger(__name__)

# --- Qwen Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 8000
DEFAULT_MAX_NEW_TOKENS_MULTI = int(os.environ.get("QWEN_DEFAULT_MAX_NEW_TOKENS_MULTI", "2048"))

# --- Environment variables for stability ---
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VLLM_USE_V1'] = '0'


def get_dynamic_gpu_memory_utilization():
    """Calculate gpu_memory_utilization based on how many models share this GPU."""
    override = os.environ.get("GPU_MEMORY_OVERRIDE")
    if override:
        try:
            val = float(override)
            logger.info(f"Using GPU_MEMORY_OVERRIDE={val}")
            return max(0.1, min(0.95, val))
        except ValueError:
            pass
    
    models_on_gpu = int(os.environ.get("MODELS_ON_GPU", "1"))
    available_fraction = 0.90
    utilization = available_fraction / models_on_gpu
    return max(0.1, min(0.9, utilization))


async def process_batch_together(
    engine: AsyncLLMEngine,
    tokenizer,
    batch: List[Dict[str, Any]],
    result_queue: Queue,
    worker_id: int,
    logger: logging.Logger
):
    """
    Process a batch of multimodal requests together for maximum batching efficiency.
    """
    request_data = []
    
    for task in batch:
        task_id = task.get("task_id", str(uuid.uuid4()))
        task_payload = task.get("payload", {})
        req_max_new_tokens = task.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS_MULTI)
        corr_id = task.get("correlation_id")
        
        if corr_id:
            try:
                set_correlation_id(corr_id)
            except Exception:
                pass
        
        try:
            user_text = task_payload.get("user_text", "")
            image_inputs_raw = task_payload.get("images", [])
            pdf_page_to_use = task_payload.get("pdf_page_to_use")
            
            if not image_inputs_raw:
                result_queue.put({
                    "task_id": task_id,
                    "error": "Multimodal task for Qwen requires at least one image.",
                    "worker_id": worker_id,
                    "status": "error"
                })
                continue
            
            # Process images
            processed_images = []
            for idx, img_item_raw in enumerate(image_inputs_raw):
                try:
                    processed_img_or_list = process_image_internal(img_item_raw, logger)
                    if isinstance(processed_img_or_list, list):
                        if not processed_img_or_list:
                            continue
                        page_idx = 0
                        if pdf_page_to_use is not None and 0 <= int(pdf_page_to_use) < len(processed_img_or_list):
                            page_idx = int(pdf_page_to_use)
                        processed_images.append(processed_img_or_list[page_idx])
                    else:
                        processed_images.append(processed_img_or_list)
                except Exception as e:
                    logger.error(f"[Task {task_id}] Failed to process image index {idx}: {e}")
            
            if not processed_images:
                result_queue.put({
                    "task_id": task_id,
                    "error": "No valid images after processing.",
                    "worker_id": worker_id,
                    "status": "error"
                })
                continue
            
            # Check max images
            max_images_allowed = int(os.environ.get("QWEN_MAX_IMAGES", "8"))
            if len(processed_images) > max_images_allowed:
                result_queue.put({
                    "task_id": task_id,
                    "error": f"TOO_MANY_IMAGES: images={len(processed_images)}, max={max_images_allowed}",
                    "worker_id": worker_id,
                    "status": "error"
                })
                continue
            
            # Build Qwen-VL prompt
            user_content = []
            for _ in processed_images:
                user_content.append({"type": "image"})
            user_content.append({"type": "text", "text": user_text})
            
            messages = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": processed_images[0] if len(processed_images) == 1 else processed_images
                }
            }
            
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=req_max_new_tokens,
                stop_token_ids=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|im_end|>")
                ],
            )
            
            request_data.append({
                "task_id": task_id,
                "inputs": inputs,
                "sampling_params": sampling_params,
                "start_time": time.time(),
                "correlation_id": corr_id,
            })
            
        except Exception as e:
            logger.error(f"Error preparing task {task_id}: {e}", exc_info=True)
            result_queue.put({
                "task_id": task_id,
                "error": str(e),
                "worker_id": worker_id,
                "status": "error"
            })
    
    if not request_data:
        return
    
    # Create async generators for ALL requests
    generators = []
    for req in request_data:
        gen = engine.generate(req["inputs"], req["sampling_params"], req["task_id"])
        generators.append((req, gen))
    
    logger.info(f"Added {len(generators)} multimodal requests to engine - collecting results")
    
    async def collect_result(req_data, gen):
        try:
            final_output = None
            async for output in gen:
                final_output = output
            
            generation_time = time.time() - req_data["start_time"]
            generated_text = final_output.outputs[0].text.strip()
            num_tokens = len(final_output.outputs[0].token_ids)
            tokens_per_second = num_tokens / generation_time if generation_time > 0 else 0
            
            result = {
                "task_id": req_data["task_id"],
                "response": generated_text,
                "worker_id": worker_id,
                "status": "success",
                "metrics": {
                    "generation_time": generation_time,
                    "num_tokens": num_tokens,
                    "tokens_per_second": tokens_per_second,
                }
            }
            result_queue.put(result)
            logger.info(f"Task {req_data['task_id']} completed - {tokens_per_second:.2f} tok/s")
            
        except Exception as e:
            logger.error(f"Error processing task {req_data['task_id']}: {e}", exc_info=True)
            result_queue.put({
                "task_id": req_data["task_id"],
                "error": str(e),
                "worker_id": worker_id,
                "status": "error"
            })
        finally:
            try:
                clear_correlation_id()
            except Exception:
                pass
    
    await asyncio.gather(*[collect_result(req, gen) for req, gen in generators])


async def task_receiver_loop(
    task_queue: Queue,
    engine: AsyncLLMEngine,
    tokenizer,
    result_queue: Queue,
    worker_id: int,
    logger: logging.Logger
):
    """
    Main async loop that batch-collects tasks, then processes them together.
    """
    loop = asyncio.get_event_loop()
    
    logger.info(f"Task receiver loop started - batch processing enabled for Qwen-VL")
    
    while True:
        try:
            task = await loop.run_in_executor(None, task_queue.get)
            
            if task is None:
                logger.info("Shutdown signal received")
                break
            
            # Batch collection
            batch = [task]
            await asyncio.sleep(0.300)  # 100ms collection window
            
            while len(batch) < 64:  # Lower cap for multimodal due to memory
                try:
                    t = task_queue.get_nowait()
                    if t is None:
                        task_queue.put(None)
                        break
                    batch.append(t)
                except queue.Empty:
                    break
            
            logger.info(f"Collected batch of {len(batch)} task(s) - processing together")
            await process_batch_together(engine, tokenizer, batch, result_queue, worker_id, logger)
            
        except Exception as e:
            logger.error(f"Error in task receiver loop: {e}", exc_info=True)
            continue
    
    logger.info("Task receiver loop ended")


def concurrent_qwen_worker(
    task_queue: Queue,
    result_queue: Queue,
    worker_id: int,
    gpu_id: str,
    process_id_str: str
):
    """
    Qwen-VL worker subprocess using AsyncLLMEngine for automatic continuous batching.
    """
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    os.environ.setdefault("ROLE", "worker_subproc")
    os.environ.setdefault("SERVICE_NAME", "orchestra")
    setup_logging_for_process(f"workerproc_{worker_id}_qwen_{process_id_str}.jsonl")
    worker_logger = logging.getLogger(f"qwen_worker_proc_{process_id_str}")
    
    worker_logger.info(f"Qwen-VL async worker {worker_id} (PID: {os.getpid()}) starting on GPU {gpu_id}")
    
    engine = None
    tokenizer = None
    
    try:
        gpu_mem_util = get_dynamic_gpu_memory_utilization()
        worker_logger.info(
            f"Loading Qwen-VL with AsyncLLMEngine - gpu_memory_utilization={gpu_mem_util:.2f} "
            f"(MODELS_ON_GPU={os.environ.get('MODELS_ON_GPU', '1')})"
        )
        
        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=8192,
            disable_log_requests=True,
        )
        
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        worker_logger.info(f"AsyncLLMEngine (Qwen-VL) loaded successfully on GPU {gpu_id}")
        
    except Exception as e:
        worker_logger.error(f"Failed to load AsyncLLMEngine: {e}", exc_info=True)
        result_queue.put({"error": f"Model loading failed: {str(e)}"})
        return
    
    result_queue.put({"status": "initialized", "worker_id": worker_id})
    worker_logger.info(f"Qwen-VL async worker {worker_id} is READY (continuous batching enabled)")
    
    try:
        asyncio.run(
            task_receiver_loop(task_queue, engine, tokenizer, result_queue, worker_id, worker_logger)
        )
    except Exception as e:
        worker_logger.error(f"Async loop error: {e}", exc_info=True)
    
    # Cleanup
    try:
        worker_logger.info("Shutting down AsyncLLMEngine...")
        if engine is not None:
            del engine
            destroy_model_parallel()
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        worker_logger.info("Qwen-VL async worker cleanup completed")
    except Exception as e:
        worker_logger.error(f"Error during cleanup: {e}")


class QwenWorkerState:
    """Manages the Qwen-VL worker subprocess and request routing."""
    
    def __init__(self, worker_id_gunicorn: int, assigned_gpu: str):
        self.gunicorn_worker_id = worker_id_gunicorn
        self.gpu_id_for_worker = assigned_gpu
        self.unique_process_id_str = str(uuid.uuid4())[:8]
        self.model_task_queue: Optional[Queue] = None
        self.model_result_queue: Optional[Queue] = None
        self.model_worker_process_handle: Optional[Process] = None
        
        self._pending_tasks: Dict[str, Dict] = {}
        self._pending_lock = threading.Lock()
        self._result_router_thread: Optional[threading.Thread] = None
        self._router_running = False
        
        logger.info(
            f"GunicornWorker-{self.gunicorn_worker_id}: Creating Qwen-VL WorkerState "
            f"on GPU {self.gpu_id_for_worker} (AsyncLLMEngine mode)"
        )
    
    def _result_router(self):
        """Background thread that routes results from result_queue to pending tasks."""
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Result router thread started")
        while self._router_running:
            try:
                result = self.model_result_queue.get(timeout=1.0)
                if result is None:
                    continue
                
                task_id = result.get("task_id")
                if task_id is None:
                    logger.warning(f"Received result without task_id: {result}")
                    continue
                
                with self._pending_lock:
                    if task_id in self._pending_tasks:
                        self._pending_tasks[task_id]["result"] = result
                        self._pending_tasks[task_id]["event"].set()
                    else:
                        logger.warning(f"Received result for unknown task_id: {task_id}")
                        
            except queue.Empty:
                continue
            except Exception as e:
                if self._router_running:
                    logger.debug(f"Result router: {e}")
        
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Result router thread stopped")

    def initialize(self):
        try:
            logger.info(
                f"GunicornWorker-{self.gunicorn_worker_id}: Initializing Qwen-VL "
                f"worker process (AsyncLLMEngine)..."
            )
            
            mp_context = mp.get_context('spawn')
            self.model_task_queue = mp_context.Queue(maxsize=256)
            self.model_result_queue = mp_context.Queue(maxsize=256)
            
            self.model_worker_process_handle = mp_context.Process(
                target=concurrent_qwen_worker,
                args=(
                    self.model_task_queue,
                    self.model_result_queue,
                    self.gunicorn_worker_id,
                    self.gpu_id_for_worker,
                    self.unique_process_id_str
                ),
                daemon=False,
                name=f"qwen_model_proc_{self.unique_process_id_str}"
            )
            self.model_worker_process_handle.start()
            
            # Wait for initialization
            init_result = self.model_result_queue.get(timeout=600)
            if "error" in init_result:
                raise Exception(f"Worker initialization failed: {init_result['error']}")
            
            # Start result router thread
            self._router_running = True
            self._result_router_thread = threading.Thread(
                target=self._result_router,
                daemon=True,
                name=f"qwen_result_router_{self.gunicorn_worker_id}"
            )
            self._result_router_thread.start()
            
            logger.info(
                f"GunicornWorker-{self.gunicorn_worker_id}: Qwen-VL AsyncLLMEngine "
                f"initialized successfully (continuous batching enabled)"
            )
            
        except Exception as e:
            logger.error(
                f"GunicornWorker-{self.gunicorn_worker_id}: Failed to initialize: {e}"
            )
            self.cleanup()
            raise

    def cleanup(self):
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Cleaning up Qwen-VL worker...")
        
        self._router_running = False
        if self._result_router_thread and self._result_router_thread.is_alive():
            self._result_router_thread.join(timeout=5)
        
        with self._pending_lock:
            for task_id, task_info in self._pending_tasks.items():
                task_info["result"] = {
                    "status": "error",
                    "error": "Worker shutting down",
                    "task_id": task_id
                }
                task_info["event"].set()
            self._pending_tasks.clear()
        
        try:
            if (self.model_task_queue and self.model_worker_process_handle 
                    and self.model_worker_process_handle.is_alive()):
                self.model_task_queue.put(None)
                self.model_worker_process_handle.join(timeout=30)
                
                if self.model_worker_process_handle.is_alive():
                    logger.warning(
                        f"GunicornWorker-{self.gunicorn_worker_id}: Force terminating worker"
                    )
                    self.model_worker_process_handle.terminate()
                    self.model_worker_process_handle.join(timeout=10)
                    
                    if self.model_worker_process_handle.is_alive():
                        self.model_worker_process_handle.kill()
                        
        except Exception as e:
            logger.error(f"GunicornWorker-{self.gunicorn_worker_id}: Cleanup error: {e}")
        
        self.model_task_queue = None
        self.model_result_queue = None
        self.model_worker_process_handle = None
        self._result_router_thread = None
        gc.collect()
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Qwen-VL cleanup complete.")


_qwen_worker_state_instance: Optional[QwenWorkerState] = None


def initialize_qwen():
    """Initialize the Qwen-VL worker for this process."""
    global _qwen_worker_state_instance
    
    if _qwen_worker_state_instance is not None:
        logger.info("Qwen-VL already initialized")
        return
    
    try:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    try:
        worker_id = int(os.environ.get("WORKER_ID", "0"))
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        
        logger.info(f"GunicornWorker-{worker_id}: Initializing Qwen-VL. Assigned GPU: {gpu_id}.")
        
        _qwen_worker_state_instance = QwenWorkerState(worker_id, gpu_id)
        _qwen_worker_state_instance.initialize()
        
        logger.info("Qwen-VL initialization completed (AsyncLLMEngine mode)")
        
    except Exception as e:
        logger.error(f"Failed to initialize Qwen-VL: {e}")
        _qwen_worker_state_instance = None
        raise


def chat_qwen(
    session_id: str,
    user_input_text: str,
    image_inputs: Optional[List[Union[str, bytes, Dict[str, Any]]]],
    req_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_MULTI,
    pdf_page_to_use: Optional[int] = None,
    timeout: float = 300.0
) -> tuple:
    """Chat with Qwen-VL model - supports concurrent requests via AsyncLLMEngine."""
    global _qwen_worker_state_instance
    
    if _qwen_worker_state_instance is None:
        raise RuntimeError("Qwen-VL model is not available.")
    
    state = _qwen_worker_state_instance
    task_id = str(uuid.uuid4())
    
    task = {
        "task_id": task_id,
        "payload": {
            "user_text": user_input_text,
            "images": image_inputs,
            "pdf_page_to_use": pdf_page_to_use,
        },
        "max_new_tokens": req_max_new_tokens,
    }
    try:
        task["correlation_id"] = get_correlation_id()
    except Exception:
        pass
    
    # Register for result routing
    event = threading.Event()
    with state._pending_lock:
        state._pending_tasks[task_id] = {"event": event, "result": None}
    
    try:
        state.model_task_queue.put(task, timeout=5)
        
        if not event.wait(timeout=timeout):
            raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
        
        with state._pending_lock:
            task_info = state._pending_tasks.pop(task_id, None)
        
        if task_info is None:
            raise Exception(f"Task {task_id} result not found")
        
        result = task_info["result"]
        
        if result.get("status") == "error":
            raise Exception(f"Worker error: {result.get('error', 'Unknown error')}")
        
        if "metrics" in result:
            metrics = result["metrics"]
            logger.info(
                f"Task {task_id} metrics - Tokens/s: {metrics.get('tokens_per_second', 0):.2f}, "
                f"Generation time: {metrics.get('generation_time', 0):.2f}s"
            )
        
        return result.get("response", ""), int(state.gpu_id_for_worker)
    
    except Exception as e:
        with state._pending_lock:
            state._pending_tasks.pop(task_id, None)
        logger.error(f"Error in chat_qwen: {e}")
        raise


async def chat_qwen_async(
    session_id: str,
    user_input_text: str,
    image_inputs: Optional[List[Union[str, bytes, Dict[str, Any]]]],
    req_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_MULTI,
    pdf_page_to_use: Optional[int] = None,
) -> tuple:
    """Async wrapper for chat_qwen."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: chat_qwen(
            session_id,
            user_input_text,
            image_inputs,
            req_max_new_tokens,
            pdf_page_to_use
        )
    )


def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded model."""
    return {
        "model_name": MODEL_NAME,
        "max_tokens": DEFAULT_MAX_NEW_TOKENS,
        "backend": "vLLM AsyncLLMEngine",
        "type": "multimodal",
        "optimizations": [
            "Continuous Batching",
            "Automatic Request Scheduling",
        ]
    }


def _cleanup_qwen_resources_on_exit():
    global _qwen_worker_state_instance
    if _qwen_worker_state_instance is not None:
        _qwen_worker_state_instance.cleanup()
        _qwen_worker_state_instance = None


atexit.register(_cleanup_qwen_resources_on_exit)


def _signal_handler(signum, frame):
    logger.info(f"Qwen-VL received signal {signum}, cleaning up...")
    _cleanup_qwen_resources_on_exit()
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)