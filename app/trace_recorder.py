from __future__ import annotations

import time
from collections import Counter
from typing import Any

from app.trace_models import (
    CanonicalTrace,
    TraceBoundaryPayload,
    TraceEvent,
    TraceEventRole,
    TraceEventType,
    TraceExecutionType,
    TraceStatus,
    TraceSummary,
)
from app.utils import make_prefixed_id, utcnow_iso


class TraceRecorder:
    def __init__(
        self,
        *,
        execution_type: TraceExecutionType,
        execution_id: str,
        episode_id: str | None = None,
        trace_id: str | None = None,
        started_at: str | None = None,
    ) -> None:
        self.trace_id = trace_id or make_prefixed_id("tr")
        self.execution_type = execution_type
        self.execution_id = execution_id
        self.episode_id = episode_id
        self.started_at = started_at or utcnow_iso()
        self._started_monotonic = time.monotonic()
        self._sequence = 0
        self._events: list[TraceEvent] = []
        self._closed = False

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def record_step(
        self,
        *,
        type: TraceEventType,
        status: TraceStatus,
        title: str,
        summary: str = "",
        input: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        detail: dict[str, Any] | None = None,
        boundary_payload: TraceBoundaryPayload | None = None,
    ) -> TraceEvent:
        return self._record(
            type=type,
            role="step",
            status=status,
            title=title,
            summary=summary,
            input=input,
            output=output,
            detail=detail,
            boundary_payload=boundary_payload,
        )

    def record_decision(
        self,
        *,
        parent_event_id: str,
        type: TraceEventType,
        status: TraceStatus,
        title: str,
        summary: str = "",
        input: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        detail: dict[str, Any] | None = None,
        boundary_payload: TraceBoundaryPayload | None = None,
    ) -> TraceEvent:
        return self._record(
            type=type,
            role="decision",
            status=status,
            title=title,
            summary=summary,
            parent_event_id=parent_event_id,
            input=input,
            output=output,
            detail=detail,
            boundary_payload=boundary_payload,
        )

    def close(
        self,
        *,
        status: TraceStatus,
        domain: str = "",
        semantic_counts: dict[str, int] | None = None,
        ended_at: str | None = None,
    ) -> CanonicalTrace:
        ended = ended_at or utcnow_iso()
        duration_ms = int((time.monotonic() - self._started_monotonic) * 1000)
        status_counts = Counter(event.status for event in self._events)
        summary = TraceSummary(
            trace_id=self.trace_id,
            execution_type=self.execution_type,
            execution_id=self.execution_id,
            episode_id=self.episode_id,
            status=status,
            started_at=self.started_at,
            ended_at=ended,
            duration_ms=duration_ms,
            total_steps=sum(1 for event in self._events if event.role == "step"),
            total_decisions=sum(1 for event in self._events if event.role == "decision"),
            error_count=sum(1 for event in self._events if event.status == "failed"),
            status_counts=dict(status_counts),
            semantic_counts=semantic_counts or {},
            domain=domain,
        )
        self._closed = True
        return CanonicalTrace(summary=summary, events=self.events)

    def _record(
        self,
        *,
        type: TraceEventType,
        role: TraceEventRole,
        status: TraceStatus,
        title: str,
        summary: str,
        parent_event_id: str | None = None,
        input: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        detail: dict[str, Any] | None = None,
        boundary_payload: TraceBoundaryPayload | None = None,
    ) -> TraceEvent:
        if self._closed:
            raise RuntimeError("trace recorder is closed")
        if parent_event_id and parent_event_id not in {event.event_id for event in self._events}:
            raise ValueError("parent_event_id must reference an existing event")
        self._sequence += 1
        event = TraceEvent(
            event_id=make_prefixed_id("te"),
            trace_id=self.trace_id,
            parent_event_id=parent_event_id,
            sequence=self._sequence,
            type=type,
            role=role,
            status=status,
            title=title,
            summary=summary,
            input=input or {},
            output=output or {},
            detail=detail or {},
            boundary_payload=boundary_payload,
            created_at=utcnow_iso(),
        )
        self._events.append(event)
        return event
