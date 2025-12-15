# models/qwen3_vllm.py - Qwen3-30B-A3B vLLM implementation with AsyncLLMEngine
# Refactored: Submit requests immediately, let vLLM handle continuous batching

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
from typing import Optional, Dict, Any
from logging_config import setup_logging_for_process, set_correlation_id, clear_correlation_id, get_correlation_id

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
DEFAULT_MAX_NEW_TOKENS = 32768

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
    available_fraction = 0.95
    utilization = available_fraction / models_on_gpu
    return max(0.1, min(0.95, utilization))


def get_vllm_engine_config() -> Dict[str, Any]:
    """
    Get vLLM engine configuration from environment variables.
    Allows tuning without code changes for H100 optimization.
    
    Environment variables:
        VLLM_MAX_MODEL_LEN: Maximum sequence length (default: 4000)
        VLLM_MAX_NUM_BATCHED_TOKENS: Max tokens per scheduler step (default: 16384)
        VLLM_MAX_NUM_SEQS: Max concurrent sequences (default: 256)
        VLLM_NUM_SCHEDULER_STEPS: Multi-step scheduling (default: 1, try 8-16 for better throughput)
        VLLM_ENABLE_CHUNKED_PREFILL: Enable chunked prefill (default: true)
        VLLM_ENABLE_PREFIX_CACHING: Enable prefix/prompt caching (default: false)
    """
    config = {
        "max_model_len": int(os.environ.get("VLLM_MAX_MODEL_LEN", "10000")),
        "max_num_batched_tokens": int(os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "65536")),
        "max_num_seqs": int(os.environ.get("VLLM_MAX_NUM_SEQS", "256")),
        "num_scheduler_steps": int(os.environ.get("VLLM_NUM_SCHEDULER_STEPS", "10")),
        "enable_chunked_prefill": os.environ.get("VLLM_ENABLE_CHUNKED_PREFILL", "true").lower() in ("true", "1", "yes"),
        "enable_prefix_caching": os.environ.get("VLLM_ENABLE_PREFIX_CACHING", "true").lower() in ("true", "1", "yes"),
    }
    return config

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_COALESCE_MS'] = '1'


async def process_single_request(
    engine: AsyncLLMEngine,
    task: Dict[str, Any],
    result_queue: Queue,
    worker_id: int,
    logger: logging.Logger
):
    """Process a single request - vLLM batches these internally at the token level."""
    task_id = task.get("task_id", str(uuid.uuid4()))
    correlation_id = task.get("correlation_id")
    
    if correlation_id:
        try:
            set_correlation_id(correlation_id)
        except Exception:
            pass
    
    try:
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
        
        start_time = time.time()
        
        # Submit immediately - vLLM scheduler handles batching
        final_output = None
        async for output in engine.generate(prompt, sampling_params, task_id):
            final_output = output
        
        generation_time = time.time() - start_time
        generated_text = final_output.outputs[0].text
        num_tokens = len(final_output.outputs[0].token_ids)
        tokens_per_second = num_tokens / generation_time if generation_time > 0 else 0
        
        result = {
            "task_id": task_id,
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
        logger.info(f"Task {task_id} completed - {tokens_per_second:.2f} tok/s")
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
        result_queue.put({
            "task_id": task_id,
            "error": str(e),
            "worker_id": worker_id,
            "status": "error"
        })
    finally:
        try:
            clear_correlation_id()
        except Exception:
            pass


def _blocking_queue_get(q: Queue, timeout: float = 0.1):
    """Blocking get with timeout - thread properly releases on timeout."""
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return "EMPTY_SENTINEL"


async def request_dispatcher(
    task_queue: Queue,
    engine: AsyncLLMEngine,
    result_queue: Queue,
    worker_id: int,
    logger: logging.Logger
):
    """
    Dispatch requests to vLLM with aggressive queue draining.
    Uses a coalescing window to batch burst arrivals together.
    """
    loop = asyncio.get_event_loop()
    active_tasks: set = set()
    
    # Coalescing: after first task, wait briefly for more to arrive
    coalesce_ms = float(os.environ.get("VLLM_COALESCE_MS", "5"))
    
    logger.info(f"Request dispatcher started - burst-aware mode (coalesce={coalesce_ms}ms)")
    
    def cleanup_done_tasks():
        done = {t for t in active_tasks if t.done()}
        for t in done:
            try:
                t.result()
            except Exception as e:
                logger.error(f"Task exception: {e}")
        active_tasks.difference_update(done)
    
    while True:
        try:
            cleanup_done_tasks()
            
            # Wait for at least one task (blocking with timeout)
            task = await loop.run_in_executor(
                None, _blocking_queue_get, task_queue, 0.05  # Faster polling
            )
            
            if task == "EMPTY_SENTINEL":
                continue
            
            if task is None:
                logger.info("Shutdown signal received")
                break
            
            # Coalescing window: wait briefly for more tasks to arrive
            if coalesce_ms > 0:
                await asyncio.sleep(coalesce_ms / 1000.0)
            
            # Collect this task and drain any others immediately available
            batch = [task]
            while len(batch) < 512:  # Increased upper bound for H100
                try:
                    t = task_queue.get_nowait()
                    if t is None:
                        task_queue.put(None)  # Re-queue shutdown signal
                        break
                    batch.append(t)
                except queue.Empty:
                    break
            
            # Submit ALL collected tasks to vLLM at once
            for t in batch:
                coro = process_single_request(engine, t, result_queue, worker_id, logger)
                async_task = asyncio.create_task(coro)
                active_tasks.add(async_task)
            
            if len(batch) > 1:
                logger.info(f"Dispatched burst of {len(batch)} tasks - {len(active_tasks)} active")
            else:
                logger.debug(f"Dispatched 1 task - {len(active_tasks)} active")
            
            # Yield to let the async tasks start their engine.generate() calls
            await asyncio.sleep(0)
            
        except Exception as e:
            logger.error(f"Dispatcher error: {e}", exc_info=True)
            continue
    
    if active_tasks:
        logger.info(f"Waiting for {len(active_tasks)} active tasks...")
        await asyncio.gather(*active_tasks, return_exceptions=True)
    
    logger.info("Request dispatcher stopped")


def concurrent_vllm_qwen3_worker(
    task_queue: Queue,
    result_queue: Queue,
    worker_id: int,
    gpu_id: str,
    process_id_str: str
):
    """vLLM worker subprocess using AsyncLLMEngine."""
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    os.environ.setdefault("ROLE", "worker_subproc")
    os.environ.setdefault("SERVICE_NAME", "orchestra")
    setup_logging_for_process(f"workerproc_{worker_id}_qwen3_{process_id_str}.jsonl")
    worker_logger = logging.getLogger(f"qwen3_vllm_worker_proc_{process_id_str}")
    
    worker_logger.info(f"vLLM Qwen3 worker {worker_id} (PID: {os.getpid()}) starting on GPU {gpu_id}")
    
    engine = None
    
    try:
        gpu_mem_util = get_dynamic_gpu_memory_utilization()
        worker_logger.info(
            f"Loading Qwen3 with AsyncLLMEngine - gpu_memory_utilization={gpu_mem_util:.2f}"
        )
        
        vllm_config = get_vllm_engine_config()
        worker_logger.info(f"vLLM config: {vllm_config}")
        
        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=vllm_config["max_model_len"],
            disable_log_requests=False,  # Enable for monitoring
            # Enable chunked prefill - critical for concurrent requests
            # Without this, ALL prefills must complete before ANY decode starts
            enable_chunked_prefill=vllm_config["enable_chunked_prefill"],
            # Max tokens per scheduler iteration (prefill chunks + decode tokens)
            # Default 2048 limits batch to ~28 requests - increase for H100
            max_num_batched_tokens=vllm_config["max_num_batched_tokens"],
            # Max concurrent sequences - H100 has plenty of KV cache headroom
            max_num_seqs=vllm_config["max_num_seqs"],
            # Multi-step scheduling - process multiple decode steps per iteration
            # Reduces scheduling overhead and improves GPU utilization
            num_scheduler_steps=vllm_config["num_scheduler_steps"],
            # Prefix caching for repeated prompts
            enable_prefix_caching=vllm_config["enable_prefix_caching"],
            enforce_eager=False,
            dtype="auto",
        )
        
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        worker_logger.info(f"AsyncLLMEngine loaded successfully on GPU {gpu_id}")
        
    except Exception as e:
        worker_logger.error(f"Failed to load AsyncLLMEngine: {e}", exc_info=True)
        result_queue.put({"error": f"Model loading failed: {str(e)}"})
        return
    
    result_queue.put({"status": "initialized", "worker_id": worker_id})
    worker_logger.info(f"vLLM Qwen3 worker {worker_id} READY")
    
    try:
        asyncio.run(
            request_dispatcher(task_queue, engine, result_queue, worker_id, worker_logger)
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
        worker_logger.info("Cleanup completed")
    except Exception as e:
        worker_logger.error(f"Cleanup error: {e}")


class Qwen3VLLMWorkerState:
    """Manages the vLLM worker subprocess and request routing."""
    
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
    
    def _result_router(self):
        """Routes results from result_queue to pending tasks."""
        logger.info(f"Result router started")
        while self._router_running:
            try:
                result = self.model_result_queue.get(timeout=1.0)
                if result is None:
                    continue
                
                task_id = result.get("task_id")
                if task_id is None:
                    continue
                
                with self._pending_lock:
                    if task_id in self._pending_tasks:
                        self._pending_tasks[task_id]["result"] = result
                        self._pending_tasks[task_id]["event"].set()
                        
            except queue.Empty:
                continue
            except Exception as e:
                if self._router_running:
                    logger.debug(f"Result router: {e}")

    def initialize(self):
        try:
            self.model_task_queue = Queue(maxsize=256)
            self.model_result_queue = Queue(maxsize=256)
            
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
            
            init_result = self.model_result_queue.get(timeout=300)
            if "error" in init_result:
                raise Exception(f"Worker init failed: {init_result['error']}")
            
            self._router_running = True
            self._result_router_thread = threading.Thread(
                target=self._result_router,
                daemon=True,
                name=f"qwen3_result_router_{self.gunicorn_worker_id}"
            )
            self._result_router_thread.start()
            
            logger.info(f"vLLM Qwen3 initialized - immediate dispatch mode")
            
        except Exception as e:
            logger.error(f"Init failed: {e}")
            self.cleanup()
            raise

    def cleanup(self):
        logger.info("Cleaning up vLLM Qwen3 worker...")
        self._router_running = False
        
        if self._result_router_thread and self._result_router_thread.is_alive():
            self._result_router_thread.join(timeout=5)
        
        with self._pending_lock:
            for task_id, task_info in self._pending_tasks.items():
                task_info["result"] = {"status": "error", "error": "Shutdown", "task_id": task_id}
                task_info["event"].set()
            self._pending_tasks.clear()
        
        try:
            if self.model_task_queue and self.model_worker_process_handle:
                if self.model_worker_process_handle.is_alive():
                    self.model_task_queue.put(None)
                    self.model_worker_process_handle.join(timeout=30)
                    if self.model_worker_process_handle.is_alive():
                        self.model_worker_process_handle.terminate()
                        self.model_worker_process_handle.join(timeout=10)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        
        self.model_task_queue = None
        self.model_result_queue = None
        self.model_worker_process_handle = None


_qwen3_vllm_worker_state_instance: Optional[Qwen3VLLMWorkerState] = None


def initialize_qwen3():
    global _qwen3_vllm_worker_state_instance
    if _qwen3_vllm_worker_state_instance is not None:
        return
    
    worker_id = int(os.environ.get("WORKER_ID", "0"))
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    
    _qwen3_vllm_worker_state_instance = Qwen3VLLMWorkerState(worker_id, gpu_id)
    _qwen3_vllm_worker_state_instance.initialize()


def chat_qwen3(
    session_id: str,
    user_input_text: str,
    req_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_beam_search: bool = False,
    timeout: float = 300.0
) -> Dict[str, Any]:
    """Chat with vLLM Qwen3 - concurrent requests batched by vLLM internally."""
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
    
    event = threading.Event()
    with state._pending_lock:
        state._pending_tasks[task_id] = {"event": event, "result": None}
    
    try:
        state.model_task_queue.put(task, timeout=5)
        
        if not event.wait(timeout=timeout):
            raise TimeoutError(f"Task {task_id} timed out")
        
        with state._pending_lock:
            task_info = state._pending_tasks.pop(task_id, None)
        
        if task_info is None:
            raise Exception(f"Task {task_id} result not found")
        
        result = task_info["result"]
        
        if result.get("status") == "error":
            raise Exception(f"Worker error: {result.get('error')}")
        
        return {
            "response": result.get("response", ""),
            "session_id": session_id,
            "metrics": result.get("metrics", {})
        }
    
    except Exception as e:
        with state._pending_lock:
            state._pending_tasks.pop(task_id, None)
        raise


async def chat_qwen3_async(
    session_id: str,
    user_input_text: str,
    req_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_beam_search: bool = False
) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, chat_qwen3, session_id, user_input_text,
        req_max_new_tokens, temperature, top_p, use_beam_search
    )


def get_model_info() -> Dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "max_tokens": DEFAULT_MAX_NEW_TOKENS,
        "backend": "vLLM AsyncLLMEngine",
        "dispatch_mode": "immediate"
    }


def _cleanup_on_exit():
    global _qwen3_vllm_worker_state_instance
    if _qwen3_vllm_worker_state_instance:
        _qwen3_vllm_worker_state_instance.cleanup()

atexit.register(_cleanup_on_exit)

def _signal_handler(signum, frame):
    _cleanup_on_exit()
    sys.exit(0)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)