import logging
import logging.config
from logging.handlers import RotatingFileHandler
import os
import sys
import json
from datetime import datetime
from contextvars import ContextVar
from typing import Any, Dict

# Author: Zied Mustapha

# Context variable to carry correlation/request IDs across async tasks
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")


def set_correlation_id(value: str) -> None:
    try:
        correlation_id_var.set(str(value))
    except Exception:
        # Avoid crashing logging on context issues
        pass


def get_correlation_id() -> str:
    try:
        return correlation_id_var.get()
    except Exception:
        return "-"


def clear_correlation_id() -> None:
    try:
        correlation_id_var.set("-")
    except Exception:
        pass


class CorrelationIdFilter(logging.Filter):
    """Injects correlation_id and process metadata into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Standard fields
        record.correlation_id = get_correlation_id()
        record.pid = os.getpid()
        record.role = os.environ.get("ROLE", "")
        record.service = os.environ.get("SERVICE_NAME", "orchestra")
        record.worker_id = os.environ.get("WORKER_ID", "")
        record.model = os.environ.get("MODEL_TO_LOAD", "")
        return True


class JsonFormatter(logging.Formatter):
    """Minimal dependency-free JSON formatter.

    This formats logs as one JSON object per line (JSONL) with consistent keys.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Base payload
        payload: Dict[str, Any] = {
            "ts": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "-"),
            "pid": getattr(record, "pid", os.getpid()),
            "role": getattr(record, "role", ""),
            "service": getattr(record, "service", "orchestra"),
            "worker_id": getattr(record, "worker_id", ""),
            "model": getattr(record, "model", ""),
        }

        # Include selected extras if present (non-standard attributes)
        standard = set([
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName", "process",
            "processName", "asctime",
            # injected by filter
            "correlation_id", "pid", "role", "service", "worker_id", "model",
        ])
        for k, v in record.__dict__.items():
            if k not in standard and not k.startswith("_"):
                try:
                    json.dumps(v)  # ensure serializable
                    payload[k] = v
                except Exception:
                    payload[k] = str(v)

        # Exception formatting
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            # Fallback to basic string if serialization fails for some reason
            safe_payload = {k: str(v) for k, v in payload.items()}
            return json.dumps(safe_payload, ensure_ascii=False)


def build_dict_config(log_file: str, level: str = "INFO", max_bytes: int = 50 * 1024 * 1024, backups: int = 10) -> dict:
    """Return a dictConfig suitable for logging.config.dictConfig.

    Note: If you are starting Uvicorn via CLI, prefer passing a JSON file that mirrors this
    structure. See run_api.sh which generates per-process configs referencing this module.
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "correlation": {"()": "logging_config.CorrelationIdFilter"}
        },
        "formatters": {
            "json": {"()": "logging_config.JsonFormatter"}
        },
        "handlers": {
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": level,
                "filename": log_file,
                "maxBytes": max_bytes,
                "backupCount": backups,
                "encoding": "utf-8",
                "formatter": "json",
                "filters": ["correlation"],
            }
        },
        "root": {"level": level, "handlers": ["file"]},
        "loggers": {
            # Quiet down uvicorn.access; we emit our own structured access logs via middleware
            "uvicorn.access": {"level": "WARNING", "handlers": ["file"], "propagate": False},
            "uvicorn.error": {"level": level, "handlers": ["file"], "propagate": False},
            "uvicorn": {"level": level, "handlers": ["file"], "propagate": False},
            "fastapi": {"level": level, "handlers": ["file"], "propagate": False},
        },
    }


def setup_logging_for_process(default_filename: str, env_var_name: str = "LOG_FILE_PATH") -> None:
    """Programmatic setup for processes not launched via Uvicorn CLI.

    - Uses env var LOG_FILE_PATH if set, else default_filename under LOG_DIR.
    - Applies JSON + rotating file handler using dictConfig above.
    """
    log_dir = os.environ.get("LOG_DIR", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    target_file = os.environ.get(env_var_name) or os.path.join(log_dir, default_filename)
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    try:
        cfg = build_dict_config(target_file, level=level)
        logging.config.dictConfig(cfg)
    except Exception as e:
        # Fallback minimal config
        logging.basicConfig(level=getattr(logging, level, logging.INFO))
        logging.getLogger(__name__).error(f"Failed to apply dictConfig logging: {e}")
