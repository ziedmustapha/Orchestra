# main.py
# Author: Zied Mustapha
import time
import asyncio
import os
import logging
from fastapi import FastAPI, Request, HTTPException
import uuid
from typing import Optional
from logging_config import (
    setup_logging_for_process,
    set_correlation_id,
    clear_correlation_id,
)

# Structured logging setup per worker
os.environ.setdefault("ROLE", "worker")
os.environ.setdefault("SERVICE_NAME", "orchestra")
_worker_id_env = os.environ.get("WORKER_ID", "0")
_model_env = os.environ.get("MODEL_TO_LOAD", "unknown")
_default_log_filename = f"worker_{_worker_id_env}_{_model_env}.jsonl"
setup_logging_for_process(_default_log_filename)
logger = logging.getLogger("inference_api_worker")

# --- Logging payload controls (env configurable, mirrored with load_balancer) ---
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
        return _truncate_str(str(obj), max_chars)
    except Exception:
        return "<unserializable>"

app = FastAPI(title="Model Inference Worker")

# Global holder for the loaded model module for this specific worker
loaded_model_module = None
# The model this worker is configured to run
MODEL_TYPE = os.environ.get("MODEL_TO_LOAD")


def _coerce_positive_int(val, default_val: int, name: str = "max_new_tokens") -> int:
    """Return a positive int or default_val if invalid. Avoid bool coercion."""
    try:
        if val is None:
            return int(default_val)
        if isinstance(val, bool):
            raise ValueError("boolean not allowed")
        coerced = int(val)
        if coerced < 1:
            raise ValueError("must be >= 1")
        return coerced
    except Exception as e:
        try:
            logger.warning(f"Invalid {name}='{val}' ({e}); using default {default_val}.")
        except Exception:
            pass
        return int(default_val)

@app.on_event("startup")
async def preload_model():
    # Ensure our JSONL logging is active even if Uvicorn CLI configured defaults
    setup_logging_for_process(_default_log_filename)
    global loaded_model_module
    
    if not MODEL_TYPE:
        logger.error("FATAL: MODEL_TO_LOAD environment variable not set. This worker doesn't know which model to load.")
        return

    logger.info(f"Worker process starting. Assigned model type: {MODEL_TYPE}")
    
    try:
        if MODEL_TYPE == "gemma3":
            import models.gemma3 as gemma_module
            loaded_model_module = gemma_module
            await asyncio.to_thread(gemma_module.initialize_gemma)
            logger.info("Gemma3 model initialized successfully.")
        elif MODEL_TYPE == "qwen":
            import models.qwen as qwen_module
            loaded_model_module = qwen_module
            await asyncio.to_thread(qwen_module.initialize_qwen)
            logger.info("Qwen model initialized successfully.")
        elif MODEL_TYPE == "qwen3":
            import models.qwen3 as qwen3_module
            loaded_model_module = qwen3_module
            await asyncio.to_thread(qwen3_module.initialize_qwen3)
            logger.info("Qwen3 model initialized successfully.")
        elif MODEL_TYPE == "whissent":
            # Whisper ASR + FR Emotion classifier
            import models.whisSent as whissent_module
            loaded_model_module = whissent_module
            await asyncio.to_thread(whissent_module.initialize_whissent)
            logger.info("WhisSent model initialized successfully.")
        else:
            logger.error(f"Unknown MODEL_TO_LOAD value: '{MODEL_TYPE}'")

    except Exception as e:
        logger.error(f"Error during {MODEL_TYPE} initialization: {e}", exc_info=True)
        loaded_model_module = None # Ensure it's None on failure

@app.post("/infer")
async def infer(request: Request):
    request_json_data = await request.json()
    mn = request_json_data.get("model_name")
    rb = request_json_data.get("request_body")

    if not mn or rb is None:
        raise HTTPException(status_code=400, detail="model_name and request_body are required.")
    
    # This worker only serves its assigned model type
    if mn != MODEL_TYPE:
        logger.warning(f"Received request for wrong model '{mn}', but this worker is for '{MODEL_TYPE}'.")
        raise HTTPException(status_code=400, detail=f"This worker only handles '{MODEL_TYPE}' model, not '{mn}'.")

    if loaded_model_module is None:
        logger.error(f"Request received for '{mn}' but the model module failed to load.")
        raise HTTPException(status_code=503, detail=f"The '{MODEL_TYPE}' model is not available on this worker.")

    session_id = rb.get("session_id", f"req-{uuid.uuid4().hex[:8]}")
    start_time = time.time()
    pid = os.getpid()
    # Log input payload (with safe truncation/redaction)
    if LOG_PAYLOADS:
        try:
            logger.info(
                "WORKER_INFER_INPUT",
                extra={
                    "event": "worker_infer_input",
                    "model": mn,
                    "pid": pid,
                    "session_id": session_id,
                    "payload": _summarize_payload(rb, max_chars=LOG_INPUT_MAX_CHARS, max_items=LOG_LIST_MAX_ITEMS),
                },
            )
        except Exception:
            pass
    logger.info(
        f"API_INFER_START",
        extra={
            "event": "api_infer_start",
            "session_id": session_id,
            "model": mn,
            "pid": pid,
        },
    )

    try:
        if mn == "gemma3":
            # Unpack Gemma arguments
            user_input = rb.get("input")
            image_inputs = rb.get("images")
            pdf_page = rb.get("pdf_page")
            # Select default based on presence of images (multimodal default is larger)
            default_tokens = loaded_model_module.DEFAULT_MAX_NEW_TOKENS
            if image_inputs:
                default_tokens = getattr(loaded_model_module, "DEFAULT_MAX_NEW_TOKENS_MULTI", default_tokens)
            max_tokens = _coerce_positive_int(rb.get("max_new_tokens"), default_tokens)
            
            # Use asyncio.to_thread because the underlying model communication uses blocking queues
            output_text, gpu_id = await asyncio.to_thread(
                loaded_model_module.chat_gemma3,
                session_id, user_input, image_inputs, pdf_page, max_tokens
            )
            final_response = {"response": output_text, "gpu_id_of_model_worker": gpu_id}
            
        elif mn == "qwen":
            # Unpack Qwen arguments
            user_input = rb.get("input")
            image_inputs = rb.get("images")
            pdf_page = rb.get("pdf_page")
            default_tokens = getattr(loaded_model_module, "DEFAULT_MAX_NEW_TOKENS", 2048)
            if image_inputs:
                default_tokens = getattr(loaded_model_module, "DEFAULT_MAX_NEW_TOKENS_MULTI", default_tokens)
            max_tokens = _coerce_positive_int(rb.get("max_new_tokens"), default_tokens)
            try:
                qwen_timeout = int(os.environ.get("QWEN_REQUEST_TIMEOUT_SECONDS", "180"))
                output_text, gpu_id = await asyncio.wait_for(
                    asyncio.to_thread(
                        loaded_model_module.chat_qwen,
                        session_id, user_input, image_inputs, max_tokens, pdf_page
                    ),
                    timeout=qwen_timeout,
                )
                final_response = {"response": output_text, "gpu_id_of_model_worker": gpu_id}
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail={
                        "error": "timeout",
                        "message": "Qwen request timed out",
                    },
                )
            except Exception as e:
                msg = str(e)
                # Detect our tagged error from the Qwen worker
                if "CONTEXT_LENGTH_EXCEEDED" in msg:
                    # Return a client error with details, do not crash worker
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "context_length_exceeded",
                            "message": msg,
                        },
                    )
                # Too many images guard
                if "TOO_MANY_IMAGES" in msg:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "too_many_images",
                            "message": msg,
                        },
                    )
                # Fallback: detect common vLLM length errors
                low = msg.lower()
                if any(k in low for k in ["max_model_len", "max_seq_len", "context length", "prompt is too long", "exceed", "too long"]):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "context_length_exceeded",
                            "message": msg,
                        },
                    )
                # Propagate other errors to outer handler
                raise
            
        elif mn == "qwen3":
            # Unpack Qwen3 arguments
            user_input = rb.get("input")
            default_tokens = getattr(loaded_model_module, "DEFAULT_MAX_NEW_TOKENS", 2048)
            max_tokens = _coerce_positive_int(rb.get("max_new_tokens"), default_tokens)

            result = await asyncio.to_thread(
                loaded_model_module.chat_qwen3,
                session_id, user_input, max_tokens
            )
            final_response = {
                "response": result["response"],
                "gpu_id_of_model_worker": "auto"
            }
        elif mn == "whissent":
            # Audio inference: Whisper ASR + Emotion
            audio = rb.get("audio")
            if not audio:
                raise HTTPException(status_code=400, detail="'audio' is required for whissent")
            return_timestamps = bool(rb.get("return_timestamps", True))
            asr_task = rb.get("task", "transcribe")  # or 'translate'
            language = rb.get("language")
            emotion_top_k = int(rb.get("emotion_top_k", 3))

            result, gpu_id = await asyncio.to_thread(
                loaded_model_module.infer_whissent,
                session_id,
                audio,
                return_timestamps,
                asr_task,
                language,
                emotion_top_k,
            )
            final_response = {**result, "gpu_id_of_model_worker": gpu_id}

        else:
            # This case should not be reached due to the check at the beginning
            raise HTTPException(status_code=500, detail="Internal worker logic error.")

    except HTTPException as e:
        # Preserve explicit HTTP errors (e.g., 400 for length exceeded)
        raise e
    except Exception as e:
        logger.error(
            f"API_INFER_ERROR {e}",
            exc_info=True,
            extra={"event": "api_infer_error", "session_id": session_id, "model": mn},
        )
        raise HTTPException(status_code=500, detail=f"Internal server error on worker: {str(e)}")
    
    duration = time.time() - start_time
    logger.info(
        "API_INFER_SUCCESS",
        extra={
            "event": "api_infer_success",
            "session_id": session_id,
            "model": mn,
            "pid": pid,
            "duration_s": round(duration, 3),
        },
    )
    # Log output payload (with safe truncation)
    if LOG_PAYLOADS:
        try:
            logger.info(
                "WORKER_INFER_OUTPUT",
                extra={
                    "event": "worker_infer_output",
                    "model": mn,
                    "pid": pid,
                    "session_id": session_id,
                    "result": _summarize_payload(final_response, max_chars=LOG_OUTPUT_MAX_CHARS, max_items=LOG_LIST_MAX_ITEMS),
                },
            )
        except Exception:
            pass
    
    return {
        **final_response,
        "model_name_processed": mn,
        "duration_seconds": round(duration, 2),
        "api_worker_pid": pid
    }

@app.get("/")
async def root():
    return {"service": "Model Inference Worker", "model_type": MODEL_TYPE, "status": "running" if loaded_model_module else "degraded"}

# Access/correlation middleware
@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or request.headers.get("X-Correlation-ID") or uuid.uuid4().hex
    set_correlation_id(rid)
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
    response.headers["X-Request-ID"] = rid
    clear_correlation_id()
    return response