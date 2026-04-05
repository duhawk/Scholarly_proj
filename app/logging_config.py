"""
Structured JSON logging for Scholarly.

Call configure_logging() once at application startup.
Use get_logger(__name__) to get a module-level logger.
"""
import json
import logging
import sys
from datetime import datetime, timezone

# Built-in LogRecord attributes — extra dicts must not reuse these names.
_RESERVED_LOG_ATTRS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname",
    "filename", "module", "exc_info", "exc_text", "stack_info",
    "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "processName", "process", "message",
    "taskName",
})


class _SafeLogger(logging.Logger):
    """Logger subclass that prefixes conflicting extra keys instead of crashing.

    Python's LogRecord.makeRecord() raises KeyError when an extra={} dict
    contains keys that shadow built-in attributes (e.g. "filename").
    This subclass renames the offending keys to "extra_<key>" so callers
    like ``logger.info("msg", extra={"filename": ...})`` never 500.
    """

    def makeRecord(self, name, level, fn, lno, msg, args, exc_info,
                   func=None, extra=None, sinfo=None):
        if extra:
            safe: dict = {}
            for key, val in extra.items():
                safe["extra_" + key if key in _RESERVED_LOG_ATTRS else key] = val
            extra = safe
        return super().makeRecord(name, level, fn, lno, msg, args, exc_info,
                                  func, extra, sinfo)


# Keep a reference to the original makeRecord so the global patch can call it.
_orig_make_record = logging.Logger.makeRecord


def _safe_make_record(self, name, level, fn, lno, msg, args, exc_info,
                      func=None, extra=None, sinfo=None):
    """Standalone safe makeRecord that can be patched onto logging.Logger."""
    if extra:
        safe: dict = {}
        for key, val in extra.items():
            safe["extra_" + key if key in _RESERVED_LOG_ATTRS else key] = val
        extra = safe
    return _orig_make_record(self, name, level, fn, lno, msg, args, exc_info,
                             func, extra, sinfo)


# Patch immediately at import time so ANY logger (even ones created before
# configure_logging() is called) benefits from the safe behaviour.
logging.Logger.makeRecord = _safe_make_record


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any extra fields passed via logger.info("...", extra={...})
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }:
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    # Register _SafeLogger so every getLogger() call returns a safe instance.
    logging.setLoggerClass(_SafeLogger)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Suppress noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
