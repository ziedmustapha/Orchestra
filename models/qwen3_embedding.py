# models/qwen3_embedding.py - Qwen3-Embedding-4B vLLM implementation with AsyncLLMEngine
# Author: Zied Mustapha
# Optimized embedding model with dedicated environment for parallel GPU execution

from vllm import LLM
from vllm.engine.arg_utils import EngineArgs
import torch
import torch.nn.functional as F
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
from typing import Optional, List, Dict, Any, Union
from logging_config import setup_logging_for_process, set_correlation_id, clear_correlation_id, get_correlation_id

logger = logging.getLogger(__name__)

# --- Qwen3-Embedding Configuration ---
MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
DEFAULT_MAX_LENGTH = 8192  # Max context length for embedding
DEFAULT_EMBEDDING_DIM = 2560  # Full embedding dimension (can be truncated)

# --- Environment variables for stability ---
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
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
    available_fraction = 0.95
    utilization = available_fraction / models_on_gpu
    return max(0.1, min(0.95, utilization))


def get_embedding_engine_config() -> Dict[str, Any]:
    """
    Get vLLM engine configuration from environment variables.
    Optimized for Qwen3-Embedding-4B embedding model.
    
    Environment variables:
        QWEN3_EMB_MAX_MODEL_LEN: Maximum sequence length (default: 8192)
        QWEN3_EMB_MAX_NUM_SEQS: Max concurrent sequences (default: 256)
        QWEN3_EMB_ENABLE_PREFIX_CACHING: Enable prefix/prompt caching (default: true)
        QWEN3_EMB_COALESCE_MS: Coalescing window in ms (default: 5)
    """
    config = {
        "max_model_len": int(os.environ.get("QWEN3_EMB_MAX_MODEL_LEN", "8192")),
        "max_num_seqs": int(os.environ.get("QWEN3_EMB_MAX_NUM_SEQS", "256")),
        "enable_prefix_caching": os.environ.get("QWEN3_EMB_ENABLE_PREFIX_CACHING", "true").lower() in ("true", "1", "yes"),
        "coalesce_ms": float(os.environ.get("QWEN3_EMB_COALESCE_MS", "5")),
    }
    return config


def _blocking_queue_get(q: Queue, timeout: float = 0.1):
    """Blocking get with timeout - thread properly releases on timeout."""
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return "EMPTY_SENTINEL"


def process_embedding_batch(
    model: LLM,
    batch: List[Dict[str, Any]],
    result_queue: Queue,
    worker_id: int,
    wlog: logging.Logger
):
    """
    Process a batch of embedding requests together for maximum throughput.
    vLLM's embed() function handles batching internally.
    """
    task_ids = []
    input_texts = []
    embedding_dims = []
    correlation_ids = []
    
    for task in batch:
        task_id = task.get("task_id", str(uuid.uuid4()))
        task_ids.append(task_id)
        
        text = task.get("text", "")
        instruction = task.get("instruction")
        
        # Format with instruction if provided
        if instruction:
            formatted_text = f"Instruct: {instruction}\nQuery:{text}"
        else:
            formatted_text = text
        
        input_texts.append(formatted_text)
        embedding_dims.append(task.get("embedding_dim", DEFAULT_EMBEDDING_DIM))
        correlation_ids.append(task.get("correlation_id"))
    
    if not input_texts:
        return
    
    start_time = time.time()
    
    try:
        # Batch embed all texts together
        outputs = model.embed(input_texts)
        
        batch_time = time.time() - start_time
        avg_time = batch_time / len(outputs) if outputs else 0
        
        wlog.info(f"Embedded batch of {len(outputs)} texts in {batch_time:.3f}s ({avg_time*1000:.1f}ms/text)")
        
        for i, output in enumerate(outputs):
            task_id = task_ids[i]
            embedding_dim = embedding_dims[i]
            corr_id = correlation_ids[i]
            
            if corr_id:
                try:
                    set_correlation_id(corr_id)
                except Exception:
                    pass
            
            try:
                # Get embedding and normalize
                embedding = torch.tensor(output.outputs.embedding)
                
                # Truncate to requested dimension if needed (MRL support)
                if embedding_dim and embedding_dim < len(embedding):
                    embedding = embedding[:embedding_dim]
                
                # Normalize embedding
                embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1).squeeze(0)
                
                result = {
                    "task_id": task_id,
                    "embedding": embedding.tolist(),
                    "embedding_dim": len(embedding),
                    "worker_id": worker_id,
                    "status": "success",
                    "metrics": {
                        "processing_time": avg_time,
                        "batch_size": len(outputs),
                    }
                }
                result_queue.put(result)
                
            except Exception as e:
                wlog.error(f"Error processing embedding for task {task_id}: {e}", exc_info=True)
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
                    
    except Exception as e:
        wlog.error(f"Batch embedding failed: {e}", exc_info=True)
        for task_id in task_ids:
            result_queue.put({
                "task_id": task_id,
                "error": str(e),
                "worker_id": worker_id,
                "status": "error"
            })


def embedding_dispatcher_loop(
    task_queue: Queue,
    model: LLM,
    result_queue: Queue,
    worker_id: int,
    wlog: logging.Logger,
    coalesce_ms: float = 5.0
):
    """
    Synchronous dispatcher for embedding requests with coalescing for better batching.
    """
    wlog.info(f"Embedding dispatcher started - burst-aware mode (coalesce={coalesce_ms}ms)")
    
    while True:
        try:
            # Wait for at least one task
            task = _blocking_queue_get(task_queue, 0.05)
            
            if task == "EMPTY_SENTINEL":
                continue
            
            if task is None:
                wlog.info("Shutdown signal received")
                break
            
            # Coalescing window: wait briefly for more tasks to arrive
            if coalesce_ms > 0:
                time.sleep(coalesce_ms / 1000.0)
            
            # Collect this task and drain any others immediately available
            batch = [task]
            while len(batch) < 512:  # Large batch for embedding efficiency
                try:
                    t = task_queue.get_nowait()
                    if t is None:
                        task_queue.put(None)  # Re-queue shutdown signal
                        break
                    batch.append(t)
                except queue.Empty:
                    break
            
            # Process batch
            process_embedding_batch(model, batch, result_queue, worker_id, wlog)
            
        except Exception as e:
            wlog.error(f"Dispatcher error: {e}", exc_info=True)
            continue
    
    wlog.info("Embedding dispatcher stopped")


def concurrent_embedding_worker(
    task_queue: Queue,
    result_queue: Queue,
    worker_id: int,
    gpu_id: str,
    process_id_str: str
):
    """Qwen3-Embedding worker subprocess using vLLM."""
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    
    os.environ.setdefault("ROLE", "worker_subproc")
    os.environ.setdefault("SERVICE_NAME", "orchestra")
    setup_logging_for_process(f"workerproc_{worker_id}_qwen3emb_{process_id_str}.jsonl")
    wlog = logging.getLogger(f"qwen3emb_worker_proc_{process_id_str}")
    
    wlog.info(f"Qwen3-Embedding worker {worker_id} (PID: {os.getpid()}) starting on GPU {gpu_id}")
    
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
            wlog.info(f"CUDA device set to cuda:0 (physical GPU {gpu_id})")
        except Exception as e:
            wlog.error(f"CUDA setup failed: {e}", exc_info=True)
            result_queue.put({"error": f"CUDA setup failed: {e}"})
            return
    
    model = None
    
    try:
        emb_config = get_embedding_engine_config()
        gpu_mem_util = get_dynamic_gpu_memory_utilization()
        wlog.info(
            f"Qwen3-Embedding config: gpu_mem={gpu_mem_util:.2f}, "
            f"max_model_len={emb_config['max_model_len']}, "
            f"max_num_seqs={emb_config['max_num_seqs']}, "
            f"coalesce_ms={emb_config['coalesce_ms']}"
        )
        
        # Use vLLM's synchronous LLM class with task="embed"
        model = LLM(
            model=MODEL_NAME,
            task="embed",
            trust_remote_code=True,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=emb_config["max_model_len"],
            max_num_seqs=emb_config["max_num_seqs"],
            enable_prefix_caching=emb_config["enable_prefix_caching"],
            enforce_eager=False,
            dtype="auto",
        )
        wlog.info(f"Qwen3-Embedding model loaded successfully on GPU {gpu_id}")
        
    except Exception as e:
        wlog.error(f"Failed to load Qwen3-Embedding model: {e}", exc_info=True)
        result_queue.put({"error": f"Model loading failed: {str(e)}"})
        return
    
    result_queue.put({"status": "initialized", "worker_id": worker_id})
    wlog.info(f"Qwen3-Embedding worker {worker_id} READY")
    
    try:
        embedding_dispatcher_loop(
            task_queue, model, result_queue, worker_id, wlog,
            coalesce_ms=emb_config["coalesce_ms"]
        )
    except Exception as e:
        wlog.error(f"Dispatcher loop error: {e}", exc_info=True)
    
    # Cleanup
    try:
        wlog.info("Shutting down Qwen3-Embedding model...")
        if model is not None:
            del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        wlog.info("Cleanup completed")
    except Exception as e:
        wlog.error(f"Cleanup error: {e}")


class Qwen3EmbeddingWorkerState:
    """Manages the embedding worker subprocess and request routing."""
    
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
        logger.info(f"Qwen3-Embedding result router started")
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
            self.model_task_queue = Queue(maxsize=512)
            self.model_result_queue = Queue(maxsize=512)
            
            self.model_worker_process_handle = Process(
                target=concurrent_embedding_worker,
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
                name=f"qwen3emb_result_router_{self.gunicorn_worker_id}"
            )
            self._result_router_thread.start()
            
            logger.info(f"Qwen3-Embedding initialized - batch embedding mode")
            
        except Exception as e:
            logger.error(f"Init failed: {e}")
            self.cleanup()
            raise

    def cleanup(self):
        logger.info("Cleaning up Qwen3-Embedding worker...")
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


_qwen3_embedding_worker_state_instance: Optional[Qwen3EmbeddingWorkerState] = None


def initialize_qwen3_embedding():
    """Initialize the Qwen3-Embedding worker."""
    global _qwen3_embedding_worker_state_instance
    if _qwen3_embedding_worker_state_instance is not None:
        return
    
    worker_id = int(os.environ.get("WORKER_ID", "0"))
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    
    _qwen3_embedding_worker_state_instance = Qwen3EmbeddingWorkerState(worker_id, gpu_id)
    _qwen3_embedding_worker_state_instance.initialize()


def embed_text(
    text: str,
    instruction: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    timeout: float = 60.0
) -> Dict[str, Any]:
    """
    Generate embedding for a single text.
    
    Args:
        text: The text to embed
        instruction: Optional instruction for task-specific embedding
        embedding_dim: Optional dimension to truncate embedding to (MRL support)
        timeout: Request timeout in seconds
    
    Returns:
        Dict with 'embedding' (list of floats), 'embedding_dim', and 'metrics'
    """
    global _qwen3_embedding_worker_state_instance
    
    if _qwen3_embedding_worker_state_instance is None:
        raise Exception("Qwen3-Embedding not initialized")
    
    state = _qwen3_embedding_worker_state_instance
    task_id = str(uuid.uuid4())
    task = {
        "task_id": task_id,
        "text": text,
        "instruction": instruction,
        "embedding_dim": embedding_dim or DEFAULT_EMBEDDING_DIM,
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
            raise TimeoutError(f"Embedding task {task_id} timed out")
        
        with state._pending_lock:
            task_info = state._pending_tasks.pop(task_id, None)
        
        if task_info is None:
            raise Exception(f"Task {task_id} result not found")
        
        result = task_info["result"]
        
        if result.get("status") == "error":
            raise Exception(f"Worker error: {result.get('error')}")
        
        return {
            "embedding": result.get("embedding", []),
            "embedding_dim": result.get("embedding_dim", 0),
            "metrics": result.get("metrics", {})
        }
    
    except Exception as e:
        with state._pending_lock:
            state._pending_tasks.pop(task_id, None)
        raise


def embed_texts(
    texts: List[str],
    instruction: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    timeout: float = 120.0
) -> Dict[str, Any]:
    """
    Generate embeddings for multiple texts in a batch.
    
    Args:
        texts: List of texts to embed
        instruction: Optional instruction applied to all texts
        embedding_dim: Optional dimension to truncate embeddings to (MRL support)
        timeout: Request timeout in seconds
    
    Returns:
        Dict with 'embeddings' (list of embedding lists), 'embedding_dim', and 'metrics'
    """
    global _qwen3_embedding_worker_state_instance
    
    if _qwen3_embedding_worker_state_instance is None:
        raise Exception("Qwen3-Embedding not initialized")
    
    state = _qwen3_embedding_worker_state_instance
    task_ids = []
    events = []
    
    # Submit all tasks
    for text in texts:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        
        task = {
            "task_id": task_id,
            "text": text,
            "instruction": instruction,
            "embedding_dim": embedding_dim or DEFAULT_EMBEDDING_DIM,
        }
        try:
            task["correlation_id"] = get_correlation_id()
        except Exception:
            pass
        
        event = threading.Event()
        events.append(event)
        
        with state._pending_lock:
            state._pending_tasks[task_id] = {"event": event, "result": None}
        
        state.model_task_queue.put(task, timeout=5)
    
    # Wait for all results
    embeddings = []
    total_time = 0
    
    try:
        for i, (task_id, event) in enumerate(zip(task_ids, events)):
            if not event.wait(timeout=timeout):
                raise TimeoutError(f"Embedding task {task_id} timed out")
            
            with state._pending_lock:
                task_info = state._pending_tasks.pop(task_id, None)
            
            if task_info is None:
                raise Exception(f"Task {task_id} result not found")
            
            result = task_info["result"]
            
            if result.get("status") == "error":
                raise Exception(f"Worker error: {result.get('error')}")
            
            embeddings.append(result.get("embedding", []))
            total_time += result.get("metrics", {}).get("processing_time", 0)
        
        return {
            "embeddings": embeddings,
            "embedding_dim": embedding_dim or DEFAULT_EMBEDDING_DIM,
            "count": len(embeddings),
            "metrics": {
                "total_time": total_time,
                "avg_time_per_text": total_time / len(embeddings) if embeddings else 0,
            }
        }
    
    except Exception as e:
        # Cleanup any remaining pending tasks
        with state._pending_lock:
            for task_id in task_ids:
                state._pending_tasks.pop(task_id, None)
        raise


async def embed_text_async(
    text: str,
    instruction: Optional[str] = None,
    embedding_dim: Optional[int] = None
) -> Dict[str, Any]:
    """Async wrapper for embed_text."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, embed_text, text, instruction, embedding_dim
    )


async def embed_texts_async(
    texts: List[str],
    instruction: Optional[str] = None,
    embedding_dim: Optional[int] = None
) -> Dict[str, Any]:
    """Async wrapper for embed_texts."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, embed_texts, texts, instruction, embedding_dim
    )


def get_model_info() -> Dict[str, Any]:
    """Return model information."""
    return {
        "model_name": MODEL_NAME,
        "model_type": "embedding",
        "max_length": DEFAULT_MAX_LENGTH,
        "embedding_dim": DEFAULT_EMBEDDING_DIM,
        "backend": "vLLM",
        "supports_instructions": True,
        "supports_mrl": True,  # Matryoshka Representation Learning
    }


def _cleanup_on_exit():
    global _qwen3_embedding_worker_state_instance
    if _qwen3_embedding_worker_state_instance:
        _qwen3_embedding_worker_state_instance.cleanup()

atexit.register(_cleanup_on_exit)

def _signal_handler(signum, frame):
    _cleanup_on_exit()
    sys.exit(0)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
