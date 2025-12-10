# models/qwen3_vllm.py - Qwen3-30B-A3B-Thinking-2507 vLLM implementation
# Author: Zied Mustapha
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import torch
import os
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
import traceback
from logging_config import setup_logging_for_process, set_correlation_id, clear_correlation_id, get_correlation_id

# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Qwen3 Configuration ---
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
DEFAULT_MAX_NEW_TOKENS = 32768  # Qwen3 supports very long context for reasoning

# --- vLLM Optimized Settings ---
# Check vLLM version to use compatible parameters
try:
    import vllm
    VLLM_VERSION = vllm.__version__ if hasattr(vllm, '__version__') else "0.0.0"
except:
    VLLM_VERSION = "0.0.0"

# --- Dynamic GPU memory allocation ---
# Calculate gpu_memory_utilization based on how many models share this GPU
# MODELS_ON_GPU is set by run_api.sh based on instance configuration
def get_dynamic_gpu_memory_utilization():
    models_on_gpu = int(os.environ.get("MODELS_ON_GPU", "1"))
    # Reserve 10% for system overhead, divide rest among models
    available_fraction = 0.90
    utilization = available_fraction / models_on_gpu
    # Clamp between 0.1 and 0.9 for safety
    return max(0.1, min(0.9, utilization))

# --- CRITICAL: Environment variables for stability ---
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'  # Use Flash Attention
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable async CUDA operations
# Force V0 engine - V1 engine (vLLM 0.8.x default) has stricter memory checks
# that fail when running multiple models on same GPU
os.environ['VLLM_USE_V1'] = '0'

# --- Concurrent vLLM Qwen3 worker ---
def concurrent_vllm_qwen3_worker(task_queue: Queue, result_queue: Queue, worker_id: int, gpu_id: str, process_id_str: str):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    # Structured JSON logging per sub-process
    os.environ.setdefault("ROLE", "worker_subproc")
    os.environ.setdefault("SERVICE_NAME", "orchestra")
    setup_logging_for_process(f"workerproc_{worker_id}_qwen3_{process_id_str}.jsonl")
    worker_process_logger = logging.getLogger(f"qwen3_vllm_worker_proc_{process_id_str}")

    worker_process_logger.info(f"vLLM Qwen3 worker {worker_id} (PID: {os.getpid()}) starting on GPU {gpu_id}")

    llm = None
    
    try:
        # Initialize vLLM model - calculate gpu_memory_utilization at runtime
        gpu_mem_util = get_dynamic_gpu_memory_utilization()
        worker_process_logger.info(f"Loading Qwen3 with vLLM - gpu_memory_utilization={gpu_mem_util:.2f} (MODELS_ON_GPU={os.environ.get('MODELS_ON_GPU', '1')})")
        
        llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=8192,  # Must set - default would be too high
        )
        worker_process_logger.info(f"vLLM Qwen3 model loaded successfully on GPU {gpu_id}")
        
        # Get actual model config for logging
        try:
            model_config = llm.llm_engine.model_config
            worker_process_logger.info(f"Model config - dtype: {model_config.dtype}, "
                                      f"max_model_len: {model_config.max_model_len}")
        except:
            pass
        
    except Exception as e:
        worker_process_logger.error(f"Failed to load vLLM Qwen3 model: {e}", exc_info=True)
        result_queue.put({"error": f"Model loading failed: {str(e)}"})
        return

    # Signal that initialization is complete
    result_queue.put({"status": "initialized", "worker_id": worker_id})
    
    # Main processing loop
    worker_process_logger.info(f"vLLM Qwen3 worker {worker_id} is READY and listening for tasks.")
    
    while True:
        try:
            task = task_queue.get()  # Block until task arrives (no timeout)
            if task is None:
                worker_process_logger.info("Received shutdown signal")
                break
                
            task_id = task.get("task_id", "unknown")
            user_input = task.get("user_input", "")
            max_new_tokens = task.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
            temperature = task.get("temperature", 0.7)
            top_p = task.get("top_p", 0.9)
            use_beam_search = task.get("use_beam_search", False)
            corr_id = task.get("correlation_id")
            if corr_id:
                try:
                    set_correlation_id(corr_id)
                except Exception:
                    pass
            
            worker_process_logger.info(f"Processing task {task_id} - max_tokens: {max_new_tokens}")
            
            try:
                # Format the prompt using Qwen chat template
                messages = [{"role": "user", "content": user_input}]
                
                # Apply chat template if needed (vLLM handles this internally for chat models)
                # For Qwen3, we can pass the messages directly or format manually
                prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
                
                # Create sampling parameters
                if use_beam_search and temperature == 0:
                    # Beam search for deterministic output
                    sampling_params = SamplingParams(
                        max_tokens=max_new_tokens,
                        temperature=0,
                        best_of=5,  # Beam width
                        use_beam_search=True,
                    )
                else:
                    # Standard sampling
                    sampling_params = SamplingParams(
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=50,
                        repetition_penalty=1.1,
                        skip_special_tokens=True,
                        stop=["<|im_end|>", "<|endoftext|>"],  # Qwen stop tokens
                    )
                
                # Generate response
                start_time = time.time()
                outputs = llm.generate([prompt], sampling_params)
                generation_time = time.time() - start_time
                
                # Extract the generated text
                generated_text = outputs[0].outputs[0].text
                
                # Calculate tokens per second
                num_tokens = len(outputs[0].outputs[0].token_ids)
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
                        "prompt_tokens": len(outputs[0].prompt_token_ids),
                    }
                }
                
                result_queue.put(result)
                worker_process_logger.info(f"Task {task_id} completed - {tokens_per_second:.2f} tok/s")
                
            except Exception as e:
                worker_process_logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
                result_queue.put({
                    "task_id": task_id,
                    "error": str(e),
                    "worker_id": worker_id,
                    "status": "error"
                })
                
        except Exception as e:
            worker_process_logger.error(f"Unexpected worker loop error: {e}", exc_info=True)
            continue
        finally:
            try:
                clear_correlation_id()
            except Exception:
                pass
    
    # Cleanup
    try:
        if llm is not None:
            # Properly cleanup vLLM resources
            worker_process_logger.info("Destroying vLLM model parallel state...")
            destroy_model_parallel()
            del llm
            
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        worker_process_logger.info("vLLM Qwen3 worker cleanup completed")
    except Exception as e:
        worker_process_logger.error(f"Error during cleanup: {e}")

class Qwen3VLLMWorkerState:
    def __init__(self, worker_id_gunicorn: int, assigned_gpu: str):
        self.gunicorn_worker_id = worker_id_gunicorn
        self.gpu_id_for_worker = assigned_gpu
        self.unique_process_id_str = str(uuid.uuid4())[:8]
        self.model_task_queue = None
        self.model_result_queue = None
        self.model_worker_process_handle = None
        
        # Concurrent request handling
        self._pending_tasks: Dict[str, Dict] = {}  # task_id -> {"event": Event, "result": Any}
        self._pending_lock = threading.Lock()
        self._result_router_thread = None
        self._router_running = False
        
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Creating vLLM Qwen3 WorkerState on GPU {self.gpu_id_for_worker}")
    
    def _result_router(self):
        """Background thread that routes results from result_queue to pending tasks."""
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Result router thread started")
        while self._router_running:
            try:
                # Use timeout to allow checking _router_running flag
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
                        
            except Exception as e:
                if self._router_running:  # Only log if not shutting down
                    if "Empty" not in str(type(e).__name__):
                        logger.debug(f"Result router: {e}")
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Result router thread stopped")

    def initialize(self):
        try:
            logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Initializing vLLM Qwen3 worker process...")
            
            # Create queues for communication
            self.model_task_queue = Queue(maxsize=100)  # Larger queue for batching
            self.model_result_queue = Queue(maxsize=100)
            
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
            init_result = self.model_result_queue.get(timeout=300)  # 5 minutes timeout
            if "error" in init_result:
                raise Exception(f"Worker initialization failed: {init_result['error']}")
            
            # Start the result router thread for concurrent request handling
            self._router_running = True
            self._result_router_thread = threading.Thread(
                target=self._result_router,
                daemon=True,
                name=f"qwen3_result_router_{self.gunicorn_worker_id}"
            )
            self._result_router_thread.start()
            
            logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: vLLM Qwen3 worker initialized successfully (concurrent mode)")
            
        except Exception as e:
            logger.error(f"GunicornWorker-{self.gunicorn_worker_id}: Failed to initialize vLLM Qwen3 worker: {e}")
            self.cleanup()
            raise

    def cleanup(self):
        logger.info(f"GunicornWorker-{self.gunicorn_worker_id}: Cleaning up vLLM Qwen3 worker...")
        
        # Stop the result router thread first
        self._router_running = False
        if self._result_router_thread and self._result_router_thread.is_alive():
            self._result_router_thread.join(timeout=5)
        
        # Wake up any pending tasks with error
        with self._pending_lock:
            for task_id, task_info in self._pending_tasks.items():
                task_info["result"] = {"status": "error", "error": "Worker shutting down", "task_id": task_id}
                task_info["event"].set()
            self._pending_tasks.clear()
        
        try:
            if self.model_task_queue and self.model_worker_process_handle and self.model_worker_process_handle.is_alive():
                self.model_task_queue.put(None)  # Shutdown signal
                self.model_worker_process_handle.join(timeout=30)
                
                if self.model_worker_process_handle.is_alive():
                    logger.warning(f"GunicornWorker-{self.gunicorn_worker_id}: Force terminating vLLM Qwen3 worker")
                    self.model_worker_process_handle.terminate()
                    self.model_worker_process_handle.join(timeout=10)
                    
                    if self.model_worker_process_handle.is_alive():
                        self.model_worker_process_handle.kill()
                        
        except Exception as e:
            logger.error(f"GunicornWorker-{self.gunicorn_worker_id}: Error during vLLM Qwen3 cleanup: {e}")
        
        self.model_task_queue = None
        self.model_result_queue = None
        self.model_worker_process_handle = None
        self._result_router_thread = None

# Global worker state instance
_qwen3_vllm_worker_state_instance: Optional[Qwen3VLLMWorkerState] = None

def initialize_qwen3():
    """Initialize the vLLM Qwen3 worker for this process"""
    global _qwen3_vllm_worker_state_instance
    
    if _qwen3_vllm_worker_state_instance is not None:
        logger.info("vLLM Qwen3 already initialized")
        return
    
    try:
        # Get worker configuration from environment
        worker_id = int(os.environ.get("WORKER_ID", "0"))
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        
        _qwen3_vllm_worker_state_instance = Qwen3VLLMWorkerState(worker_id, gpu_id)
        _qwen3_vllm_worker_state_instance.initialize()
        
        logger.info("vLLM Qwen3 initialization completed successfully")
        
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
    """Chat with vLLM Qwen3 model - supports concurrent requests via task-id matching"""
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
    
    # Register this task for result routing
    event = threading.Event()
    with state._pending_lock:
        state._pending_tasks[task_id] = {"event": event, "result": None}
    
    try:
        # Send task to worker
        state.model_task_queue.put(task, timeout=5)
        
        # Wait for result (the result router thread will signal us)
        if not event.wait(timeout=timeout):
            raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
        
        # Get result
        with state._pending_lock:
            task_info = state._pending_tasks.pop(task_id, None)
        
        if task_info is None:
            raise Exception(f"Task {task_id} result not found")
        
        result = task_info["result"]
        
        if result.get("status") == "error":
            raise Exception(f"Worker error: {result.get('error', 'Unknown error')}")
        
        # Log performance metrics if available
        if "metrics" in result:
            metrics = result["metrics"]
            logger.info(f"Task {task_id} metrics - Tokens/s: {metrics.get('tokens_per_second', 0):.2f}, "
                       f"Generation time: {metrics.get('generation_time', 0):.2f}s, "
                       f"Output tokens: {metrics.get('num_tokens', 0)}")
        
        return {
            "response": result.get("response", ""),
            "session_id": session_id,
            "metrics": result.get("metrics", {})
        }
    
    except Exception as e:
        # Clean up pending task on error
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
    """Async wrapper for chat_qwen3"""
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

def batch_chat_qwen3(
    requests: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Process multiple chat requests in batch for better throughput"""
    global _qwen3_vllm_worker_state_instance
    
    if _qwen3_vllm_worker_state_instance is None:
        raise Exception("vLLM Qwen3 not initialized")
    
    results = []
    task_ids = []
    
    try:
        # Send all tasks to the queue
        for req in requests:
            task_id = str(uuid.uuid4())
            task = {
                "task_id": task_id,
                "user_input": req.get("user_input", ""),
                "max_new_tokens": req.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
                "temperature": req.get("temperature", 0.7),
                "top_p": req.get("top_p", 0.9),
                "use_beam_search": req.get("use_beam_search", False)
            }
            try:
                task["correlation_id"] = get_correlation_id()
            except Exception:
                pass
            task_ids.append(task_id)
            _qwen3_vllm_worker_state_instance.model_task_queue.put(task, timeout=5)
        
        # Collect all results
        for _ in task_ids:
            result = _qwen3_vllm_worker_state_instance.model_result_queue.get(timeout=300)
            results.append(result)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch_chat_qwen3: {e}")
        raise

def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded model"""
    return {
        "model_name": MODEL_NAME,
        "max_tokens": DEFAULT_MAX_NEW_TOKENS,
        "quantization": "FP8",
        "backend": "vLLM",
        "optimizations": [
            "Flash Attention",
            "CUDA Graphs",
            "Prefix Caching",
            "Chunked Prefill",
            "FP8 Quantization"
        ]
    }

def _cleanup_qwen3_resources_on_exit():
    """Cleanup function called on exit"""
    global _qwen3_vllm_worker_state_instance
    if _qwen3_vllm_worker_state_instance:
        _qwen3_vllm_worker_state_instance.cleanup()

atexit.register(_cleanup_qwen3_resources_on_exit)

def _signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"vLLM Qwen3 received signal {signum}, cleaning up...")
    _cleanup_qwen3_resources_on_exit()
    sys.exit(0)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)