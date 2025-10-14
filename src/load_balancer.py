# load_balancer.py
import asyncio
import aiohttp
import time
import logging
import os
import json
import uuid
from typing import Dict, List, Optional, Tuple
from collections import deque
from fastapi import FastAPI, Request, HTTPException, Header, Response
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
from ipaddress import ip_address, ip_network
from logging_config import set_correlation_id, clear_correlation_id, get_correlation_id, setup_logging_for_process, build_dict_config

# Ensure environment metadata for logs
os.environ.setdefault("ROLE", "balancer")
os.environ.setdefault("SERVICE_NAME", "orchestra")

# Initialize structured logging to a rotating JSONL file
setup_logging_for_process("load_balancer.jsonl")

logger = logging.getLogger("load_balancer")

# --- Logging payload controls (env configurable) ---
def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1","true","yes","on"}

LOG_PAYLOADS = _env_bool("LOG_PAYLOADS", True)
LOG_INPUT_MAX_CHARS = int(os.environ.get("LOG_INPUT_MAX_CHARS", "1500"))
LOG_OUTPUT_MAX_CHARS = int(os.environ.get("LOG_OUTPUT_MAX_CHARS", "1500"))
LOG_LIST_MAX_ITEMS = int(os.environ.get("LOG_LIST_MAX_ITEMS", "20"))
SENSITIVE_KEYS = {"audio","image","images","pdf","file","file_content","image_url","image_data"}

def _truncate_str(s: str, max_len: int) -> str:
    try:
        if max_len <= 0:
            return ""
        if s is None:
            return ""
        if len(s) <= max_len:
            return s
        head = s[: max_len // 2]
        tail = s[- max_len // 2 :]
        return f"{head}…{tail} (len={len(s)})"
    except Exception:
        return "<str>"

def _summarize_payload(obj, max_chars: int = LOG_INPUT_MAX_CHARS, max_items: int = LOG_LIST_MAX_ITEMS):
    try:
        if obj is None:
            return None
        if isinstance(obj, (int, float, bool)):
            return obj
        if isinstance(obj, str):
            return _truncate_str(obj, max_chars)
        if isinstance(obj, bytes):
            return f"<bytes len={len(obj)}>"
        if isinstance(obj, list):
            out = []
            for i, v in enumerate(obj):
                if i >= max_items:
                    out.append(f"<… {len(obj) - max_items} more items>")
                    break
                out.append(_summarize_payload(v, max_chars=max_chars, max_items=max_items))
            return out
        if isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                try:
                    key = str(k)
                except Exception:
                    key = "<key>"
                if key.lower() in SENSITIVE_KEYS:
                    # Replace with a compact descriptor
                    if isinstance(v, str):
                        res[key] = f"<redacted str len={len(v)}>"
                    elif isinstance(v, (list, tuple)):
                        res[key] = f"<redacted list len={len(v)}>"
                    elif isinstance(v, bytes):
                        res[key] = f"<redacted bytes len={len(v)}>"
                    else:
                        res[key] = "<redacted>"
                else:
                    res[key] = _summarize_payload(v, max_chars=max_chars, max_items=max_items)
            return res
        # Fallback to string repr truncated
        return _truncate_str(str(obj), max_chars)
    except Exception:
        return "<unserializable>"

# Exclude noisy dashboard-refresh endpoints from access logs
EXCLUDED_ACCESS_PATHS = set(
    p.strip() for p in os.environ.get(
        "LB_ACCESS_EXCLUDE",
        "/status,/metrics,/logs_json,/logs,/dashboard,/favicon.ico"
    ).split(",") if p.strip()
)
EXCLUDED_ACCESS_PREFIXES = ["/assets/", "/static/"]

# Load API key configuration
def load_api_config():
    """Load API key configuration from env or local file.

    Resolution order:
      1) API_KEYS_FILE env var (absolute or relative path)
      2) ./api_keys.json in current working directory
      3) <repo_dir>/api_keys.json next to this file
    """
    # 1) Env var override
    path = os.environ.get("API_KEYS_FILE")
    # 2) CWD fallback
    if not path:
        cwd_candidate = os.path.join(os.getcwd(), "api_keys.json")
        if os.path.exists(cwd_candidate):
            path = cwd_candidate
    # 3) Repo-local fallback
    if not path:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        repo_candidate = os.path.join(base_dir, "api_keys.json")
        path = repo_candidate

    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"API keys config file not found at {path}, allowing all access")
        return {"valid_keys": [], "require_auth_for_external": False, "local_networks": ["127.0.0.1", "localhost", "::1"]}
    except Exception as e:
        logger.error(f"Error loading API config from {path}: {e}")
        return {"valid_keys": [], "require_auth_for_external": False, "local_networks": ["127.0.0.1", "localhost", "::1"]}

def is_local_request(client_ip: str, local_networks: List[str]) -> bool:
    """Check if the request is from a local network"""
    try:
        client_addr = ip_address(client_ip)
        for network in local_networks:
            if network in ['localhost', '127.0.0.1', '::1']:
                if str(client_addr) in ['127.0.0.1', '::1'] or client_ip == 'localhost':
                    return True
            else:
                try:
                    if client_addr in ip_network(network, strict=False):
                        return True
                except ValueError:
                    continue
        return False
    except ValueError:
        # If we can't parse the IP, assume it's not local
        return False

def validate_api_key(api_key: Optional[str], valid_keys: List[str]) -> bool:
    """Validate the provided API key"""
    return api_key is not None and api_key in valid_keys

class WorkerInfo:
    def __init__(self, worker_id: int, port: int, model_name: str):
        self.worker_id = worker_id
        self.port = port
        self.model_name = model_name
        self.is_busy = False
        self.queue = deque()
        self.lock = asyncio.Lock()
        # --- Metrics ---
        self.requests: int = 0
        self.success: int = 0
        self.errors: int = 0
        self.latencies_ms: deque = deque(maxlen=200)
        self.last_duration_ms: float = 0.0
        self.last_error: Optional[str] = None

class LoadBalancer:
    def __init__(self, base_port: int, gemma_workers: int, qwen_workers: int, qwen3_workers: int = 0, whissent_workers: int = 0):
        self.worker_pools: Dict[str, List[WorkerInfo]] = {
            "gemma3": [],
            "qwen": [],
            "qwen3": [],
            "whissent": [],
        }
        
        worker_id_counter = 0
        # Initialize Gemma workers
        for i in range(gemma_workers):
            port = base_port + worker_id_counter
            self.worker_pools["gemma3"].append(WorkerInfo(worker_id_counter, port, "gemma3"))
            worker_id_counter += 1
            
        # Initialize Qwen workers
        for i in range(qwen_workers):
            port = base_port + worker_id_counter
            self.worker_pools["qwen"].append(WorkerInfo(worker_id_counter, port, "qwen"))
            worker_id_counter += 1
            
        # Initialize Qwen3 workers
        for i in range(qwen3_workers):
            port = base_port + worker_id_counter
            self.worker_pools["qwen3"].append(WorkerInfo(worker_id_counter, port, "qwen3"))
            worker_id_counter += 1
        
        # Initialize WhisSent workers
        for i in range(whissent_workers):
            port = base_port + worker_id_counter
            self.worker_pools["whissent"].append(WorkerInfo(worker_id_counter, port, "whissent"))
            worker_id_counter += 1
            
        self.session: Optional[aiohttp.ClientSession] = None
        logger.info(
            f"Load Balancer configured for: {gemma_workers} Gemma3, {qwen_workers} Qwen, {qwen3_workers} Qwen3, {whissent_workers} WhisSent workers."
        )
        # --- Global metrics ---
        self.start_time = time.time()
        self.total_requests = 0
        self.total_success = 0
        self.total_errors = 0
        self.per_model_metrics: Dict[str, Dict[str, int]] = {
            "gemma3": {"requests": 0, "success": 0, "errors": 0},
            "qwen": {"requests": 0, "success": 0, "errors": 0},
            "qwen3": {"requests": 0, "success": 0, "errors": 0},
            "whissent": {"requests": 0, "success": 0, "errors": 0},
        }

    async def start(self):
        self.session = aiohttp.ClientSession()
        
    async def stop(self):
        if self.session:
            await self.session.close()
            
    def get_worker_for_model(self, model_name: str) -> Optional[WorkerInfo]:
        """Finds the best worker for a given model (free or shortest queue)."""
        if model_name not in self.worker_pools or not self.worker_pools[model_name]:
            return None
        
        pool = self.worker_pools[model_name]
        
        # Prioritize non-busy workers
        free_workers = [w for w in pool if not w.is_busy]
        if free_workers:
            # Simple round-robin among free workers
            return min(free_workers, key=lambda w: w.worker_id)
        
        # If all are busy, find the one with the shortest queue
        return min(pool, key=lambda w: len(w.queue))

    def _is_overcharged(self, worker: 'WorkerInfo') -> bool:
        """Return True when the worker's queue is over 10 (strictly > 10)."""
        try:
            q_len = len(worker.queue)
        except Exception:
            q_len = 0
        return q_len > 10

    def _worker_state(self, worker: 'WorkerInfo') -> str:
        """Derive a simple state for UI/metrics: OVERCHARGED, BUSY, or IDLE.

        - OVERCHARGED when queue length > 10
        - BUSY when queue length is between 1 and 10 (inclusive) or the worker is currently busy
        - IDLE otherwise
        """
        try:
            q_len = len(worker.queue)
        except Exception:
            q_len = 0
        if q_len > 10:
            return "OVERCHARGED"
        if q_len > 0 or worker.is_busy:
            return "BUSY"
        return "IDLE"

    async def forward_request(self, worker: WorkerInfo, request_data: dict) -> dict:
        url = f"http://127.0.0.1:{worker.port}/infer"
        try:
            headers = {"X-Request-ID": get_correlation_id()}
            async with self.session.post(url, json=request_data, headers=headers, timeout=300000) as response:
                if response.status != 200:
                    error_detail = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"Worker {worker.worker_id} error: {error_detail}")
                result = await response.json()
                result['load_balancer_worker_id'] = worker.worker_id
                return result
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail=f"Request to worker {worker.worker_id} timed out.")
        except Exception as e:
            logger.error(f"Error forwarding to worker {worker.worker_id}: {e}", exc_info=True)
            raise HTTPException(status_code=502, detail=f"Bad Gateway: Error communicating with worker {worker.worker_id}.")

    async def process_request(self, request_data: dict) -> dict:
        model_name = request_data.get("model_name")
        if not model_name:
            raise HTTPException(status_code=400, detail="`model_name` is required.")
        
        worker = self.get_worker_for_model(model_name)
        if not worker:
            raise HTTPException(status_code=503, detail=f"No workers available for model '{model_name}'.")

        future = asyncio.Future()
        async with worker.lock:
            if worker.is_busy:
                # Queue the request
                worker.queue.append((request_data, future))
                logger.info(f"Queuing request for '{model_name}' on worker {worker.worker_id} (queue size: {len(worker.queue)})")
                # The task will be picked up by the worker when it's free
            else:
                # Process immediately
                worker.is_busy = True
                future.set_result(True) # Signal to proceed
                logger.info(f"Routing request for '{model_name}' directly to free worker {worker.worker_id}")

        # Wait for our turn
        await future
        # Metrics: mark dispatch state
        t0 = time.perf_counter()
        queue_at_dispatch = len(worker.queue)
        try:
            result = await self.forward_request(worker, request_data)
            # Success path metrics
            dt_ms = (time.perf_counter() - t0) * 1000.0
            worker.latencies_ms.append(dt_ms)
            worker.last_duration_ms = dt_ms
            worker.requests += 1
            worker.success += 1
            self.total_requests += 1
            self.total_success += 1
            if model_name in self.per_model_metrics:
                self.per_model_metrics[model_name]["requests"] += 1
                self.per_model_metrics[model_name]["success"] += 1
            # annotate response
            result["lb_latency_ms"] = round(dt_ms, 1)
            result["lb_queue_at_dispatch"] = queue_at_dispatch
            return result
        except Exception as e:
            # Error path metrics
            dt_ms = (time.perf_counter() - t0) * 1000.0
            worker.last_duration_ms = dt_ms
            worker.requests += 1
            worker.errors += 1
            worker.last_error = str(e)
            self.total_requests += 1
            self.total_errors += 1
            if model_name in self.per_model_metrics:
                self.per_model_metrics[model_name]["requests"] += 1
                self.per_model_metrics[model_name]["errors"] += 1
            raise
        finally:
            # This block runs after the request is completed or fails
            async with worker.lock:
                # If there are items in the queue, start processing the next one
                if worker.queue:
                    next_request_data, next_future = worker.queue.popleft()
                    # The worker remains busy for the next task
                    logger.info(f"Dequeuing request for worker {worker.worker_id}")
                    next_future.set_result(True) # Allow the next waiting task to proceed
                else:
                    # No more tasks, worker is now free
                    worker.is_busy = False

    def get_status(self) -> dict:
        status = {"model_pools": {}}
        for model_name, pool in self.worker_pools.items():
            status["model_pools"][model_name] = [
                {
                    "worker_id": w.worker_id,
                    "port": w.port,
                    "is_busy": w.is_busy,
                    "queue_length": len(w.queue),
                    "requests": w.requests,
                    "success": w.success,
                    "errors": w.errors,
                    "avg_recent_latency_ms": round((sum(list(w.latencies_ms)[-10:]) / max(1, len(list(w.latencies_ms)[-10:]))) if w.latencies_ms else 0.0, 1),
                    "last_duration_ms": round(w.last_duration_ms, 1),
                    "overcharged": self._is_overcharged(w),
                    "state": self._worker_state(w),
                } for w in pool
            ]
        return status

    def get_metrics(self) -> dict:
        now = time.time()
        uptime = max(0.0, now - self.start_time)
        rps = (self.total_success / uptime) if uptime > 0 else 0.0
        per_worker: Dict[int, Dict[str, float | int | str]] = {}
        for model_name, pool in self.worker_pools.items():
            for w in pool:
                latencies = list(w.latencies_ms)
                recent10 = latencies[-10:]
                avg_recent = (sum(recent10) / len(recent10)) if recent10 else 0.0
                per_worker[w.worker_id] = {
                    "model": model_name,
                    "port": w.port,
                    "requests": w.requests,
                    "success": w.success,
                    "errors": w.errors,
                    "avg_recent_latency_ms": round(avg_recent, 1),
                    "last_duration_ms": round(w.last_duration_ms, 1),
                    "overcharged": self._is_overcharged(w),
                    "state": self._worker_state(w),
                }
        return {
            "uptime_seconds": round(uptime, 1),
            "total_requests": self.total_requests,
            "success": self.total_success,
            "errors": self.total_errors,
            "rps": round(rps, 2),
            "per_model": self.per_model_metrics,
            "per_worker": per_worker,
            "timestamp": now,
        }

app = FastAPI(title="ORCHESTRA - Multi-Model Smart Load Balancer")
load_balancer: Optional[LoadBalancer] = None

@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    """Assign and propagate a correlation/request ID for every request and log access."""
    rid = request.headers.get("X-Request-ID") or request.headers.get("X-Correlation-ID") or uuid.uuid4().hex
    set_correlation_id(rid)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.error(
            "REQUEST_ERROR",
            exc_info=True,
            extra={
                "event": "http_request_error",
                "path": request.url.path,
                "method": request.method,
                "client_ip": request.client.host if request.client else None,
            },
        )
        clear_correlation_id()
        raise
    duration_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Request-ID"] = rid
    try:
        path = request.url.path
        excluded = (path in EXCLUDED_ACCESS_PATHS) or any(path.startswith(pref) for pref in EXCLUDED_ACCESS_PREFIXES)
        if not excluded:
            logger.info(
                "ACCESS",
                extra={
                    "event": "http_request",
                    "path": path,
                    "method": request.method,
                    "status_code": getattr(response, "status_code", 0),
                    "duration_ms": round(duration_ms, 1),
                    "client_ip": request.client.host if request.client else None,
                    "query": str(request.url.query) if request.url.query else "",
                },
            )
    finally:
        clear_correlation_id()
    return response

@app.on_event("startup")
async def startup():
    global load_balancer
    gemma_instances = int(os.environ.get("GEMMA3_INSTANCES", "1"))
    qwen_instances = int(os.environ.get("QWEN_INSTANCES", "0"))
    qwen3_instances = int(os.environ.get("QWEN3_INSTANCES", "0"))
    whissent_instances = int(os.environ.get("WHISSENT_INSTANCES", "0"))
    base_port = int(os.environ.get("WORKER_BASE_PORT", "9100"))
    load_balancer = LoadBalancer(base_port, gemma_instances, qwen_instances, qwen3_instances, whissent_instances)
    await load_balancer.start()

# ... (shutdown, infer, status, root endpoints are similar, just use the new load_balancer logic) ...
@app.on_event("shutdown")
async def shutdown():
    if load_balancer: await load_balancer.stop()

@app.post("/infer")
async def infer(request: Request, x_api_key: Optional[str] = Header(None)):
    if not load_balancer: 
        raise HTTPException(status_code=503, detail="Load balancer not initialized")
    
    # Load API configuration
    api_config = load_api_config()
    
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Check if authentication is required
    if api_config["require_auth_for_external"]:
        is_local = is_local_request(client_ip, api_config["local_networks"])
        
        if not is_local:
            # External request - require API key
            if not validate_api_key(x_api_key, api_config["valid_keys"]):
                logger.warning(f"Unauthorized access attempt from {client_ip} with key: {x_api_key}")
                raise HTTPException(
                    status_code=401, 
                    detail="Invalid or missing API key. Please provide a valid X-API-Key header."
                )
            logger.info(f"Authenticated external request from {client_ip}")
        else:
            logger.info(f"Local request from {client_ip} - no authentication required")
    
    request_data = await request.json()
    # Log input payload (with safe truncation/redaction)
    if LOG_PAYLOADS:
        try:
            client_ip = request.client.host if request.client else "unknown"
            logger.info(
                "LB_INFER_INPUT",
                extra={
                    "event": "lb_infer_input",
                    "client_ip": client_ip,
                    "payload": _summarize_payload(request_data, max_chars=LOG_INPUT_MAX_CHARS, max_items=LOG_LIST_MAX_ITEMS),
                },
            )
        except Exception:
            pass
    result = await load_balancer.process_request(request_data)
    # Log output payload (with safe truncation)
    if LOG_PAYLOADS:
        try:
            logger.info(
                "LB_INFER_OUTPUT",
                extra={
                    "event": "lb_infer_output",
                    "result": _summarize_payload(result, max_chars=LOG_OUTPUT_MAX_CHARS, max_items=LOG_LIST_MAX_ITEMS),
                },
            )
        except Exception:
            pass
    return result

@app.get("/status")
async def status():
    if not load_balancer: return {"error": "Load balancer not initialized"}
    return load_balancer.get_status()
    
@app.get("/metrics")
async def metrics():
    if not load_balancer: return {"error": "Load balancer not initialized"}
    return load_balancer.get_metrics()

@app.get("/logs")
async def logs(kind: str = "balancer", worker_id: Optional[int] = None, lines: int = 200):
    """Return the last N lines of logs.
    kind: 'balancer' or 'worker'
    worker_id: required when kind='worker'
    lines: number of lines to tail
    """
    try:
        log_dir = os.environ.get("LOG_DIR", "./logs")
        if kind == "balancer":
            path = os.path.join(log_dir, "load_balancer.jsonl")
            if not os.path.exists(path):
                path = os.path.join(log_dir, "load_balancer.log")
        elif kind == "worker":
            if not load_balancer:
                raise HTTPException(status_code=503, detail="Load balancer not initialized")
            if worker_id is None:
                raise HTTPException(status_code=400, detail="worker_id is required for kind=worker")
            # Find worker and build file name
            found: Optional[WorkerInfo] = None
            for pool in load_balancer.worker_pools.values():
                for w in pool:
                    if w.worker_id == worker_id:
                        found = w
                        break
                if found:
                    break
            if not found:
                raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")
            path = os.path.join(log_dir, f"worker_{found.worker_id}_{found.model_name}.jsonl")
            if not os.path.exists(path):
                path = os.path.join(log_dir, f"worker_{found.worker_id}_{found.model_name}.log")
        else:
            raise HTTPException(status_code=400, detail="Invalid kind. Use 'balancer' or 'worker'.")

        if not os.path.exists(path):
            return Response(content=f"Log file not found: {path}", media_type="text/plain")
        with open(path, "r", errors="ignore") as f:
            all_lines = f.readlines()
            tail = all_lines[-max(1, int(lines)) :]
        return Response(content="".join(tail), media_type="text/plain")
    except HTTPException:
        raise
    except Exception as e:
        return Response(content=f"Error reading logs: {e}", media_type="text/plain")

@app.get("/logs_json")
async def logs_json(
    kind: str = "balancer",
    worker_id: Optional[int] = None,
    lines: int = 200,
    q: Optional[str] = None,
    level: Optional[str] = None,
    event: Optional[str] = None,
    corr: Optional[str] = None,
):
    """Return last N log lines as structured JSON objects with basic filtering.

    Query parameters:
      - kind: 'balancer' or 'worker'
      - worker_id: required when kind='worker'
      - lines: max lines to read from end of file
      - q: substring search across the raw line and serialized JSON
      - level: exact level name (e.g., INFO, WARNING, ERROR)
      - event: exact 'event' field match if present
      - corr: correlation id substring match (field 'correlation_id')
    """
    try:
        log_dir = os.environ.get("LOG_DIR", "./logs")
        if kind == "balancer":
            path = os.path.join(log_dir, "load_balancer.jsonl")
            if not os.path.exists(path):
                path = os.path.join(log_dir, "load_balancer.log")
        elif kind == "worker":
            if not load_balancer:
                raise HTTPException(status_code=503, detail="Load balancer not initialized")
            if worker_id is None:
                raise HTTPException(status_code=400, detail="worker_id is required for kind=worker")
            found: Optional[WorkerInfo] = None
            for pool in load_balancer.worker_pools.values():
                for w in pool:
                    if w.worker_id == worker_id:
                        found = w
                        break
                if found:
                    break
            if not found:
                raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")
            path = os.path.join(log_dir, f"worker_{found.worker_id}_{found.model_name}.jsonl")
            if not os.path.exists(path):
                path = os.path.join(log_dir, f"worker_{found.worker_id}_{found.model_name}.log")
        else:
            raise HTTPException(status_code=400, detail="Invalid kind. Use 'balancer' or 'worker'.")

        if not os.path.exists(path):
            return JSONResponse(content={"items": [], "error": f"Log file not found: {path}"}, status_code=200)

        with open(path, "r", errors="ignore") as f:
            all_lines = f.readlines()
            last = all_lines[-max(1, int(lines)) :]

        items = []
        total = len(all_lines)
        start_index = total - len(last)
        for idx, raw in enumerate(last):
            raw = raw.strip('\n')
            line_no = start_index + idx + 1  # 1-based absolute line number in file
            rec = None
            try:
                rec = json.loads(raw)
            except Exception:
                # Not JSON, pack as basic record
                rec = {"ts": None, "level": "INFO", "logger": "raw", "msg": raw}

            # attach stable identity metadata
            try:
                rec.setdefault("_line_no", line_no)
                rec.setdefault("_k", f"{os.path.basename(path)}:{line_no}")
            except Exception:
                pass

            # apply filters
            if level and str(rec.get("level", "")).upper() != str(level).upper():
                continue
            if event and str(rec.get("event", "")) != str(event):
                continue
            if corr:
                cid = str(rec.get("correlation_id", ""))
                if corr not in cid:
                    continue
            if q:
                sline = raw
                try:
                    sline = sline + " " + json.dumps(rec, ensure_ascii=False)
                except Exception:
                    pass
                if q not in sline:
                    continue
            items.append(rec)

        return JSONResponse(content={"items": items, "count": len(items), "path": path})
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(content={"items": [], "error": str(e)}, status_code=200)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Simple Tailwind-based dashboard for live status/metrics/logs."""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>AI Instances Dashboard</title>
      <script src="https://cdn.tailwindcss.com"></script>
      <style>
        .card { transition: transform .1s ease-in-out; }
        .card:hover { transform: translateY(-2px); }
        .blink { animation: blinker 1s linear infinite; }
        @keyframes blinker { 50% { opacity: 0.3; } }
        pre { white-space: pre-wrap; }
        .lvl-INFO { color: #a7f3d0; }
        .lvl-WARNING { color: #fbbf24; }
        .lvl-ERROR { color: #f87171; }
        .lvl-DEBUG { color: #93c5fd; }
        .tag { font-size: 0.75rem; padding: 0.1rem 0.4rem; border-radius: 0.25rem; }
        .tag-key { background: rgba(148,163,184,0.2); color: #e2e8f0; }
        .tag-val { background: rgba(30,41,59,0.8); color: #a5b4fc; }
        /* Modal styles */
        .modal-overlay { position: fixed; inset: 0; background: rgba(2,6,23,0.75); display: none; z-index: 50; }
        .modal { position: fixed; top: 5%; left: 50%; transform: translateX(-50%); width: 92%; max-width: 960px; max-height: 90vh; overflow: auto; background: #0f172a; border: 1px solid #334155; border-radius: 0.5rem; display: none; z-index: 51; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        .modal-header { display:flex; justify-content: space-between; align-items:center; padding: 0.75rem 1rem; border-bottom: 1px solid #334155; }
        .modal-body { padding: 1rem; }
        .modal-close { cursor: pointer; }
      </style>
    </head>
    <body class="bg-slate-950 text-slate-100">
      <div class="max-w-7xl mx-auto p-4">
        <header class="mb-4">
          <h1 class="text-3xl font-bold">ORCHESTRA - Multi-Model Smart AI Load Balancer</h1>
          <p class="text-slate-400">Live view of workers, metrics, and logs</p>
        </header>

        <section id="overview" class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <div class="card bg-slate-900/80 rounded-lg p-4">
            <div class="text-slate-400">Uptime</div>
            <div id="uptime" class="text-2xl font-semibold">--</div>
          </div>
          <div class="card bg-slate-900/80 rounded-lg p-4">
            <div class="text-slate-400">Total Requests</div>
            <div id="totalReq" class="text-2xl font-semibold">--</div>
          </div>
          <div class="card bg-slate-900/80 rounded-lg p-4">
            <div class="text-slate-400">Success</div>
            <div id="success" class="text-2xl font-semibold text-emerald-400">--</div>
          </div>
          <div class="card bg-slate-900/80 rounded-lg p-4">
            <div class="text-slate-400">Errors</div>
            <div id="errors" class="text-2xl font-semibold text-rose-400">--</div>
          </div>
        </section>

        <section id="models" class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-6"></section>

        <section>
          <div class="flex items-center justify-between mb-2">
            <h2 class="text-xl font-semibold">Workers</h2>
            <div class="flex items-center gap-2 flex-wrap">
              <label for="logSelect" class="text-sm text-slate-400">Logs:</label>
              <select id="logSelect" class="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-sm"></select>
              <input id="logQuery" placeholder="Search anything" class="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-sm w-48" />
              <input id="corrInput" placeholder="Correlation ID" class="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-sm w-44" />
              <input id="eventInput" placeholder="Event" class="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-sm w-32" />
              <select id="levelSelect" class="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-sm">
                <option value="">level</option>
                <option>DEBUG</option>
                <option>INFO</option>
                <option>WARNING</option>
                <option>ERROR</option>
              </select>
              <input id="maxLines" type="number" min="50" max="5000" value="300" class="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-sm w-24" />
              <label class="text-sm text-slate-400 flex items-center gap-1"><input id="autoRefreshLogs" type="checkbox" checked /> Auto</label>
              <button id="refreshLogs" class="bg-slate-800 hover:bg-slate-700 rounded px-3 py-1 text-sm">Refresh</button>
            </div>
          </div>
          <div id="workers" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mb-4"></div>
          <div class="bg-slate-900/70 rounded-lg p-3">
            <div id="logList" class="text-sm text-slate-300 h-80 overflow-y-auto space-y-2"></div>
          </div>
        </section>
      </div>

      <!-- Modal elements -->
      <div id="logModalOverlay" class="modal-overlay"></div>
      <div id="logModal" class="modal">
        <div class="modal-header">
          <div id="logModalTitle" class="text-sm text-slate-300">Log Details</div>
          <button id="logModalClose" class="modal-close bg-slate-800 hover:bg-slate-700 rounded px-3 py-1 text-sm">Close</button>
        </div>
        <div class="modal-body">
          <pre id="logModalPre" class="text-xs text-slate-200 overflow-x-auto"></pre>
        </div>
      </div>

      <script>
        const fmt = (n) => new Intl.NumberFormat().format(n);
        function fmtUptime(sec){
          sec = Math.floor(sec);
          const h = Math.floor(sec/3600), m = Math.floor((sec%3600)/60), s = sec%60;
          return `${h}h ${m}m ${s}s`;
        }
        // Keep track of which log entries' details are expanded across refreshes
        const openDetailsKeys = new Set();
        // Keep records by key for modal popup
        const logRecords = new Map();
        let modalOpen = false;

        async function fetchJSON(path){
          const res = await fetch(path);
          return await res.json();
        }

        function badge(text, color){
          return `<span class="px-2 py-0.5 rounded text-xs ${color}">${text}</span>`;
        }

        function workerCard(w){
          const busy = w.is_busy;
          const over = w.overcharged;
          const base = over ? 'border-rose-600/60 shadow shadow-rose-900' : (busy ? 'border-amber-500/40' : 'border-emerald-600/40');
          const status = over ? badge('OVERCHARGED', 'bg-rose-900/60 text-rose-300') : (busy ? badge('BUSY', 'bg-amber-900/60 text-amber-300') : badge('IDLE', 'bg-emerald-900/60 text-emerald-300'));
          return `
            <div class="card bg-slate-900/80 border ${base} rounded-lg p-4">
              <div class="flex items-center justify-between mb-2">
                <div class="font-semibold">Worker #${w.worker_id} • ${w.port}</div>
                ${status}
              </div>
              <div class="text-slate-400 text-sm mb-2">Model: <span class="text-slate-200">${w.model || ''}</span></div>
              <div class="grid grid-cols-2 gap-2 text-sm">
                <div>Queue: <span class="font-mono">${w.queue_length}</span></div>
                <div>Req: <span class="font-mono">${w.requests}</span> | Ok: <span class="text-emerald-400 font-mono">${w.success}</span> | Err: <span class="text-rose-400 font-mono">${w.errors}</span></div>
                <div>Avg Lat (10): <span class="font-mono">${w.avg_recent_latency_ms} ms</span></div>
                <div>Last: <span class="font-mono">${w.last_duration_ms} ms</span></div>
              </div>
            </div>`;
        }

        function render(status, metrics){
          // Overview
          document.getElementById('uptime').textContent = fmtUptime(metrics.uptime_seconds || 0);
          document.getElementById('totalReq').textContent = fmt(metrics.total_requests || 0);
          document.getElementById('success').textContent = fmt(metrics.success || 0);
          document.getElementById('errors').textContent = fmt(metrics.errors || 0);

          // Models
          const modelsEl = document.getElementById('models');
          modelsEl.innerHTML = '';
          const modelPools = status.model_pools || {};
          Object.keys(modelPools).forEach(m => {
            const mm = metrics.per_model?.[m] || {requests:0, success:0, errors:0};
            const html = `
              <div class="card bg-slate-900/80 border border-slate-700 rounded-lg p-4">
                <div class="flex items-center justify-between">
                  <div class="font-semibold">${m}</div>
                  <div class="text-slate-400 text-sm">Workers: ${modelPools[m].length}</div>
                </div>
                <div class="mt-2 text-sm">Req: <span class="font-mono">${mm.requests}</span> • Ok: <span class="text-emerald-400 font-mono">${mm.success}</span> • Err: <span class="text-rose-400 font-mono">${mm.errors}</span></div>
              </div>`;
            modelsEl.insertAdjacentHTML('beforeend', html);
          });

          // Workers
          const workersEl = document.getElementById('workers');
          workersEl.innerHTML = '';
          const options = [['balancer','Load Balancer Logs']];
          Object.entries(modelPools).forEach(([model, arr]) => {
            arr.forEach(w => {
              w.model = model; // annotate for UI
              workersEl.insertAdjacentHTML('beforeend', workerCard(w));
              options.push([`worker:${w.worker_id}`, `Worker ${w.worker_id} (${model})`]);
            })
          })
          const sel = document.getElementById('logSelect');
          const cur = sel.value;
          sel.innerHTML = '';
          options.forEach(([v, label]) => {
            const opt = document.createElement('option');
            opt.value = v; opt.textContent = label; sel.appendChild(opt);
          });
          if (cur) sel.value = cur;
        }

        async function refresh(){
          try {
            const [status, metrics] = await Promise.all([fetchJSON('/status'), fetchJSON('/metrics')]);
            render(status, metrics);
          } catch(e) { console.error(e); }
        }

        function buildLogsUrl(){
          const sel = document.getElementById('logSelect');
          const q = encodeURIComponent(document.getElementById('logQuery').value || '');
          const corr = encodeURIComponent(document.getElementById('corrInput').value || '');
          const event = encodeURIComponent(document.getElementById('eventInput').value || '');
          const level = encodeURIComponent(document.getElementById('levelSelect').value || '');
          const lines = Number(document.getElementById('maxLines').value || 300);
          const val = sel.value || 'balancer';
          let base = '/logs_json?kind=balancer';
          if (val.startsWith('worker:')){
            const id = val.split(':')[1];
            base = `/logs_json?kind=worker&worker_id=${id}`;
          }
          const params = [`lines=${lines}`];
          if (q) params.push(`q=${q}`);
          if (corr) params.push(`corr=${corr}`);
          if (event) params.push(`event=${event}`);
          if (level) params.push(`level=${level}`);
          return `${base}&${params.join('&')}`;
        }

        function buildKey(rec){
          const ts = rec.ts || '';
          const event = rec.event || '';
          const lvl = rec.level || 'INFO';
          const msg = rec.msg || '';
          const cid = rec.correlation_id || '';
          return (rec._k) ? rec._k : (cid ? `cid:${cid}|${ts}|${event}|${lvl}` : `${ts}|${event}|${lvl}|${(msg||'').slice(0,120)}`);
        }

        function createLogNode(rec){
          const lvl = rec.level || 'INFO';
          const ts = rec.ts || '';
          const event = rec.event || '';
          const msg = rec.msg || '';
          const cid = rec.correlation_id || '';
          const model = rec.model || '';
          const wid = rec.worker_id || '';
          const role = rec.role || '';
          const key = buildKey(rec);
          logRecords.set(key, rec);
          const top = document.createElement('div');
          top.className = 'border border-slate-700 rounded p-2 bg-slate-900/60';
          top.dataset.key = key;
          top.dataset.out = '0';
          const h = document.createElement('div');
          h.className = 'flex items-center justify-between';
          h.innerHTML = `
            <div class="flex items-center gap-2">
              <span class="tag lvl-${lvl}">${lvl}</span>
              <span class="text-slate-400">${ts}</span>
            </div>
            <div class="text-slate-400 text-xs flex items-center gap-2">
              <span>${role}${model? ' • '+model: ''}${wid? ' • w'+wid: ''}</span>
              <button class="log-popup-btn bg-slate-800 hover:bg-slate-700 rounded px-2 py-0.5 text-xs" title="Open popup">Popup</button>
            </div>
          `;
          top.appendChild(h);
          const body = document.createElement('div');
          body.className = 'mt-1';
          const em = event ? `<span class="tag tag-key">event</span> <span class="tag tag-val">${event}</span>` : '';
          const cc = cid ? `<span class="tag tag-key">cid</span> <span class="tag tag-val cursor-pointer" title="Copy" onclick="navigator.clipboard.writeText('${cid}')">${cid}</span>` : '';
          body.innerHTML = `
            <div class="text-slate-200">${msg || ''}</div>
            <div class="mt-1 flex items-center gap-2">${em} ${cc}</div>
            <details class="mt-1">
              <summary class="cursor-pointer text-slate-400">Details</summary>
              <pre class="text-xs text-slate-400 overflow-x-auto">${JSON.stringify(rec, null, 2)}</pre>
            </details>
          `;
          const det = body.querySelector('details');
          if (det){
            det.dataset.key = key;
            det.open = openDetailsKeys.has(key);
            det.addEventListener('toggle', () => {
              if (det.open) {
                openDetailsKeys.add(key);
              } else {
                openDetailsKeys.delete(key);
                // If this node is outside the current window, remove it when closed
                if (top?.dataset?.out === '1') {
                  top.remove();
                }
              }
            });
          }
          const popupBtn = h.querySelector('.log-popup-btn');
          if (popupBtn){
            popupBtn.addEventListener('click', (e) => { e.stopPropagation(); openLogModal(key); });
          }
          top.appendChild(body);
          return top;
        }

        function renderLogs(items){
          const cont = document.getElementById('logList');
          if (!items || !items.length){
            cont.innerHTML = '<div class="text-slate-500">No log entries.</div>';
            return;
          }
          // Desired (newest first) order
          const arr = items.slice().reverse();
          // Update record map for modal
          for (const rec of arr){
            const key = buildKey(rec);
            logRecords.set(key, rec);
          }
          const desiredSet = new Set(arr.map(rec => buildKey(rec)));
          // Map existing nodes
          const existingNodes = Array.from(cont.children);
          const existingMap = new Map();
          for (const node of existingNodes){
            const k = node.dataset?.key;
            if (k) existingMap.set(k, node);
          }
          // Remove nodes that dropped out of the window unless they are expanded (pinned)
          for (const node of existingNodes){
            const k = node.dataset?.key;
            if (!k) { node.remove(); continue; }
            if (desiredSet.has(k)) { node.dataset.out = '0'; continue; }
            // Not in desired window
            if (openDetailsKeys.has(k)) {
              // pin while open
              node.dataset.out = '1';
              continue;
            }
            node.remove();
          }
          // Insert only new nodes at the top, newest-first so the very latest stays at the top
          let anchor = cont.firstChild;
          for (const rec of arr){
            const key = buildKey(rec);
            if (existingMap.has(key)) continue;
            const node = createLogNode(rec);
            cont.insertBefore(node, anchor || null);
          }
          // Restore open state for any details blocks that should remain expanded
          for (const det of cont.querySelectorAll('details')){
            const k = det.dataset?.key;
            if (k && openDetailsKeys.has(k)) det.open = true;
          }
        }

        // Modal logic
        const overlayEl = document.getElementById('logModalOverlay');
        const modalEl = document.getElementById('logModal');
        const modalTitle = document.getElementById('logModalTitle');
        const modalPre = document.getElementById('logModalPre');
        const modalClose = document.getElementById('logModalClose');

        function openLogModal(key){
          const rec = logRecords.get(key);
          if (!rec) return;
          modalTitle.textContent = `${rec.level || 'INFO'} • ${rec.ts || ''} • ${rec.event || ''}`;
          modalPre.textContent = JSON.stringify(rec, null, 2);
          overlayEl.style.display = 'block';
          modalEl.style.display = 'block';
          modalOpen = true;
          document.body.style.overflow = 'hidden';
        }
        function closeLogModal(){
          overlayEl.style.display = 'none';
          modalEl.style.display = 'none';
          modalOpen = false;
          document.body.style.overflow = '';
        }
        overlayEl.addEventListener('click', closeLogModal);
        modalClose.addEventListener('click', closeLogModal);
        window.addEventListener('keydown', (e)=>{ if(e.key === 'Escape' && modalOpen) closeLogModal(); });

        async function refreshLogs(){
          const url = buildLogsUrl();
          try{
            const res = await fetch(url);
            const data = await res.json();
            renderLogs(data.items || []);
          }catch(e){ console.error(e); }
        }

        document.getElementById('refreshLogs').addEventListener('click', refreshLogs);
        setInterval(refresh, 2000);
        refresh();
        function maybeAuto(){
          const auto = document.getElementById('autoRefreshLogs').checked;
          if (!auto) return;
          // Pause auto-refresh while any Details are expanded
          if (openDetailsKeys.size > 0) return;
          if (modalOpen) return;
          refreshLogs();
        }
        setInterval(maybeAuto, 3000);
        refreshLogs();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
    
@app.get("/")
async def root(): return {"service": "Multi-Model Load Balancer", "status": "running"}

if __name__ == "__main__":
    # When running the balancer directly, apply our JSON logging config to uvicorn
    log_dir = os.environ.get("LOG_DIR", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    log_cfg = build_dict_config(os.path.join(log_dir, "load_balancer.jsonl"))
    host = os.environ.get("LOAD_BALANCER_HOST", "127.0.0.1")
    port = int(os.environ.get("LOAD_BALANCER_PORT", "9001"))
    uvicorn.run(app, host=host, port=port, log_config=log_cfg)