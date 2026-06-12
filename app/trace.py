from __future__ import annotations

from contextvars import ContextVar

_run_id: ContextVar[str] = ContextVar("run_id", default="")
_step: ContextVar[str] = ContextVar("step", default="")


def get_run_id() -> str:
    return _run_id.get()


def get_step() -> str:
    return _step.get()


def bind(*, run_id: str = "", step: str = "") -> None:
    """Set trace context for the current async task."""
    if run_id:
        _run_id.set(run_id)
    if step:
        _step.set(step)


def clear() -> None:
    _run_id.set("")
    _step.set("")
