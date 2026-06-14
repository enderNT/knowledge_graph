from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.observability import ObservabilityService

from app.trace import get_run_id, get_step

_SKIP_LOGGERS = frozenset({"app.observability", "httpx._client", "asyncio", "uvicorn.access"})


class ObservabilityLogHandler(logging.Handler):
    def __init__(self, service: ObservabilityService) -> None:
        super().__init__(level=logging.INFO)
        self._service = service

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.name in _SKIP_LOGGERS:
                return
            rec = record.__dict__
            run_id = rec.get("run_id") or get_run_id()
            step = rec.get("step") or get_step()
            event: dict = {
                "ts": rec.get("ts", ""),
                "ts_epoch_ms": int(record.created * 1000),
                "level": record.levelname,
                "service": "",
                "logger": record.name,
                "event": record.getMessage(),
                "run_id": run_id,
                "step": step,
                "job_id": rec.get("job_id", ""),
                "session_id": rec.get("session_id", ""),
                "episode_id": rec.get("episode_id", ""),
                "block_id": rec.get("block_id", ""),
                "tool_name": rec.get("tool_name", ""),
                "path": rec.get("path", ""),
                "status": rec.get("status", "") or rec.get("session_status", ""),
                "duration_ms": rec.get("duration_ms"),
                "input_shape": rec.get("input_shape"),
                "output_shape": rec.get("output_shape"),
                "counts": rec.get("counts"),
                # passthrough for transition detection
                "session_status": rec.get("session_status", ""),
            }
            if record.exc_info and record.exc_info[1] is not None:
                exc = record.exc_info[1]
                event["error_type"] = type(exc).__name__
                event["error_message"] = str(exc)[:500]
            self._service.enqueue(event)
        except Exception:
            self.handleError(record)
