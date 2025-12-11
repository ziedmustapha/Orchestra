# models/qwen3_vllm.py - Qwen3-30B-A3B vLLM implementation with AsyncLLMEngine
# Author: Zied Mustapha
# Refactored to use AsyncLLMEngine for automatic continuous batching

from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.distributed.parallel_state import destroy_model_parallel
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
import traceback
from logging_config import setup_logging_for_process, set_correlation_id, clear_correlation_id, get_correlation_id

logger = logging.getLogger(__name__)

# --- Qwen3 Configuration ---
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
DEFAULT_MAX_NEW_TOKENS = 32768

# --- Dynamic GPU memory allocation ---
def get_dynamic_gpu_memory_utilization():
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

# --- Environment variables for stability ---
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['VLLM_USE_V1'] = '0'


async def process_batch_together(
    engine: AsyncLLMEngine,
    batch: List[Dict[str, Any]],
    result_queue: Queue,
    worker_id: int,
    logger: logging.Logger
):
    """
    Process a batch of requests together for maximum batching efficiency.
    Adds all requests to the engine first, then collects results.
    """
    from vllm.outputs import RequestOutput
    
    # Prepare all requests
    request_data = []
    for task in batch:
        task_id = task.get("task_id", str(uuid.uuid4()))
        user_input = task.get("user_input", "")
        max_new_tokens = task.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        temperature = task.get("temperature", 0.7)
        top_p = task.get("top_p", 0.9)
        use_beam_search = task.get("use_beam_search", False)
        
        prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        
        if use_beam_search and temperature == 0:
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0,
                best_of=5,
                use_beam_search=True,
            )
        else:
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=50,
                repetition_penalty=1.1,
                skip_special_tokens=True,
                stop=["<|im_end|>", "<|endoftext|>"],
            )
        
        request_data.append({
            "task_id": task_id,
            "prompt": prompt,
            "sampling_params": sampling_params,
            "start_time": time.time(),
            "correlation_id": task.get("correlation_id"),
        })
    
    # Create async generators for ALL requests before iterating any of them
    generators = []
    for req in request_data:
        if req["correlation_id"]:
            try:
                set_correlation_id(req["correlation_id"])
            except Exception:
                pass
        gen = engine.generate(req["prompt"], req["sampling_params"], req["task_id"])
        generators.append((req, gen))
    
    logger.info(f"Added {len(generators)} requests to engine - now collecting results")
    
    # Process all generators concurrently
    async def collect_result(req_data, gen):
        try:
            final_output = None
            async for output in gen:
                final_output = output
            
            generation_time = time.time() - req_data["start_time"]
            generated_text = final_output.outputs[0].text
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
                    "prompt_tokens": len(final_output.prompt_token_ids),
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
    
    # Run all result collectors concurrently
    await asyncio.gather(*[collect_result(req, gen) for req, gen in generators])


async def task_receiver_loop(
    task_queue: Queue,
    engine: AsyncLLMEngine,
    result_queue: Queue,
    worker_id: int,
    logger: logging.Logger
):
    """
    Main async loop that batch-collects tasks from the queue, then processes them together.
    """
    loop = asyncio.get_event_loop()
    
    logger.info(f"Task receiver loop started - batch processing enabled")
    
    while True:
        try:
            # Wait for at least one task (blocking)
            task = await loop.run_in_executor(None, task_queue.get)
            
            if task is None:
                logger.info("Shutdown signal received")
                break
            
            # Batch collection: gather all available tasks
            batch = [task]
            
            # Wait for concurrent requests to arrive
            await asyncio.sleep(0.100)  # 100ms collection window
            
            # Drain any additional queued tasks (non-blocking)
            while len(batch) < 128:
                try:
                    t = task_queue.get_nowait()
                    if t is None:
                        task_queue.put(None)
                        break
                    batch.append(t)
                except queue.Empty:
                    break
            
            logger.info(f"Collected batch of {len(batch)} task(s) - processing together")
            
            # Process entire batch together for maximum batching
            await process_batch_together(engine, batch, result_queue, worker_id, logger)
            
        except Exception as e:
            logger.error(f"Error in task receiver loop: {e}", exc_info=True)
            continue
    
    logger.info("Task receiver loop ended")


def concurrent_vllm_qwen3_worker(
    task_queue: Queue,
    result_queue: Queue,
    worker_id: int,
    gpu_id: str,
    process_id_str: str
):
    """
    vLLM worker subprocess using AsyncLLMEngine for automatic continuous batching.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    # Setup logging
    os.environ.setdefault("ROLE", "worker_subproc")
    os.environ.setdefault("SERVICE_NAME", "orchestra")
    setup_logging_for_process(f"workerproc_{worker_id}_qwen3_{process_id_str}.jsonl")
    worker_logger = logging.getLogger(f"qwen3_vllm_worker_proc_{process_id_str}")
    
    worker_logger.info(f"vLLM Qwen3 async worker {worker_id} (PID: {os.getpid()}) starting on GPU {gpu_id}")
    
    engine = None
    
    try:
        # Initialize AsyncLLMEngine
        gpu_mem_util = get_dynamic_gpu_memory_utilization()
        worker_logger.info(
            f"Loading Qwen3 with AsyncLLMEngine - gpu_memory_utilization={gpu_mem_util:.2f} "
            f"(MODELS_ON_GPU={os.environ.get('MODELS_ON_GPU', '1')})"
        )
        
        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=4000,
            #max_num_batched_tokens=16384,
            #max_num_seqs=512,
            # AsyncLLMEngine specific settings
            disable_log_requests=True,  # Reduce log spam
        )
        
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        worker_logger.info(f"AsyncLLMEngine loaded successfully on GPU {gpu_id}")
        
    except Exception as e:
        worker_logger.error(f"Failed to load AsyncLLMEngine: {e}", exc_info=True)
        result_queue.put({"error": f"Model loading failed: {str(e)}"})
        return
    
    # Signal initialization complete
    result_queue.put({"status": "initialized", "worker_id": worker_id})
    worker_logger.info(f"vLLM Qwen3 async worker {worker_id} is READY (continuous batching enabled)")
    
    # Run the async event loop - this handles all concurrent requests
    try:
        asyncio.run(
            task_receiver_loop(task_queue, engine, result_queue, worker_id, worker_logger)
        )
    except Exception as e:
        worker_logger.error(f"Async loop error: {e}", exc_info=True)
    
    # Cleanup
    try:
        worker_logger.info("Shutting down AsyncLLMEngine...")
        if engine is not None:
            # Shutdown the engine
            del engine
            destroy_model_parallel()
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        worker_logger.info("vLLM Qwen3 async worker cleanup completed")
    except Exception as e:
        worker_logger.error(f"Error during cleanup: {e}")


class Qwen3VLLMWorkerState:
    """Manages the vLLM worker subprocess and request routing."""
    
    def __init__(self, worker_id_gunicorn: int, assigned_gpu: str):
        self.gunicorn_worker_id = worker_id_gunicorn
        self.gpu_id_for_worker = assigned_gpu
        self.unique_process_id_str = str(uuid.uuid4())[:8]
        self.model_task_queue: Optional[Queue] = None
        self.model_result_queue: Optional[Queue] = None
        self.model_worker_process_handle: Optional[Process] = None
        
        # Concurrent request handling
        self._pending_tasks: Dict[str, Dict] = {}
        self._pending_lock = threading.Lock()
        self._result_router_thread: Optional[threading.Thread] = None
        self._router_running = False
        
        logger.info(
            f"GunicornWorker-{self.gunicorn_worker_id}: Creating vLLM Qwen3 WorkerState "
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
                f"GunicornWorker-{self.gunicorn_worker_id}: Initializing vLLM Qwen3 "
                f"worker process (AsyncLLMEngine)..."
            )
            
            # Create queues - larger for better batching opportunities
            self.model_task_queue = Queue(maxsize=256)
            self.model_result_queue = Queue(maxsize=256)
            
            # Start the worker process
            self.model_worker_process_handle = Process(
                target=concurrent_vllm_qwen3_worker,
                args=(
                    self.model_task_queue,
                    self.model_result_queue,
                    self.gunicorn_worker_id,
                    self.gpu_id_for_worker,
                    self.unique_process_id_str
                )
            )
            self.model_worker_process_handle.start()
            
            # Wait for initialization confirmation
            init_result = self.model_result_queue.get(timeout=300)
            if "error" in init_result:
                raise Exception(f"Worker initialization failed: {init_result['error']}")
            
            # Start the result router thread
            self._router_running = True
            self._result_router_thread = threading.Thread(
                target=self._result_router,
                daemon=True,
                name=f"qwen3_result_router_{self.gunicorn_worker_id}"
            )
            self._result_router_thread.start()
            
            logger.info(
                f"GunicornWorker-{self.gunicorn_worker_id}: vLLM Qwen3 AsyncLLMEngine "
                f"initialized successfully (continuous batching enabled)"
            )
            
        except Exception as e:
            logger.error(
                f"GunicornWorker-{self.gunicorn_worker_id}: Failed to initialize: {e}"
            )
            self.cleanup()
            raise

    def cleanup(self):
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Cleaning up vLLM Qwen3 worker...")
        
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


# Global worker state instance
_qwen3_vllm_worker_state_instance: Optional[Qwen3VLLMWorkerState] = None


def initialize_qwen3():
    """Initialize the vLLM Qwen3 worker for this process."""
    global _qwen3_vllm_worker_state_instance
    
    if _qwen3_vllm_worker_state_instance is not None:
        logger.info("vLLM Qwen3 already initialized")
        return
    
    try:
        worker_id = int(os.environ.get("WORKER_ID", "0"))
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        
        _qwen3_vllm_worker_state_instance = Qwen3VLLMWorkerState(worker_id, gpu_id)
        _qwen3_vllm_worker_state_instance.initialize()
        
        logger.info("vLLM Qwen3 initialization completed (AsyncLLMEngine mode)")
        
    except Exception as e:
        logger.error(f"Failed to initialize vLLM Qwen3: {e}")
        _qwen3_vllm_worker_state_instance = None
        raise


def chat_qwen3(
    session_id: str,
    user_input_text: str,
    req_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_beam_search: bool = False,
    timeout: float = 300.0
) -> Dict[str, Any]:
    """Chat with vLLM Qwen3 model - supports concurrent requests via AsyncLLMEngine."""
    global _qwen3_vllm_worker_state_instance
    
    if _qwen3_vllm_worker_state_instance is None:
        raise Exception("vLLM Qwen3 not initialized")
    
    state = _qwen3_vllm_worker_state_instance
    task_id = str(uuid.uuid4())
    task = {
        "task_id": task_id,
        "user_input": user_input_text,
        "max_new_tokens": req_max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "use_beam_search": use_beam_search
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
                f"Generation time: {metrics.get('generation_time', 0):.2f}s, "
                f"Output tokens: {metrics.get('num_tokens', 0)}"
            )
        
        return {
            "response": result.get("response", ""),
            "session_id": session_id,
            "metrics": result.get("metrics", {})
        }
    
    except Exception as e:
        with state._pending_lock:
            state._pending_tasks.pop(task_id, None)
        logger.error(f"Error in chat_qwen3: {e}")
        raise


async def chat_qwen3_async(
    session_id: str,
    user_input_text: str,
    req_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_beam_search: bool = False
) -> Dict[str, Any]:
    """Async wrapper for chat_qwen3."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        chat_qwen3,
        session_id,
        user_input_text,
        req_max_new_tokens,
        temperature,
        top_p,
        use_beam_search
    )


def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded model."""
    return {
        "model_name": MODEL_NAME,
        "max_tokens": DEFAULT_MAX_NEW_TOKENS,
        "quantization": "FP8",
        "backend": "vLLM AsyncLLMEngine",
        "optimizations": [
            "Flash Attention",
            "Continuous Batching",
            "Automatic Request Scheduling",
            "FP8 Quantization"
        ]
    }


def _cleanup_qwen3_resources_on_exit():
    global _qwen3_vllm_worker_state_instance
    if _qwen3_vllm_worker_state_instance:
        _qwen3_vllm_worker_state_instance.cleanup()


atexit.register(_cleanup_qwen3_resources_on_exit)


def _signal_handler(signum, frame):
    logger.info(f"vLLM Qwen3 received signal {signum}, cleaning up...")
    _cleanup_qwen3_resources_on_exit()
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)