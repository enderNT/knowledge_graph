from __future__ import annotations

import json
import logging
import time

from app.trace import get_run_id, get_step


_SKIP_ATTRS = frozenset({
    "args", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "message", "module", "msecs", "msg",
    "name", "pathname", "process", "processName", "relativeCreated",
    "stack_info", "thread", "threadName", "taskName",
})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self._utc_ts(record.created),
            "level": record.levelname,
            "logger": record.name,
        }
        run_id = get_run_id()
        step = get_step()
        if run_id:
            payload["run_id"] = run_id
        if step:
            payload["step"] = step
        payload["msg"] = record.getMessage()
        for key, val in record.__dict__.items():
            if key not in _SKIP_ATTRS and not key.startswith("_"):
                payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        json_line = json.dumps(payload, ensure_ascii=False, default=str)
        if record.levelno >= logging.ERROR:
            return f"\n--- ERROR ---\n{json_line}\n"
        if record.levelno >= logging.WARNING:
            return f"\n--- WARNING ---\n{json_line}\n"
        return json_line + "\n"

    @staticmethod
    def _utc_ts(created: float) -> str:
        t = time.gmtime(created)
        ms = int((created % 1) * 1000)
        return time.strftime("%Y-%m-%dT%H:%M:%S", t) + f".{ms:03d}Z"


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()
    root.addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
