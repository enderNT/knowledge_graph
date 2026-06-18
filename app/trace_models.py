from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


TraceExecutionType = Literal["ingestion_job"]
TraceEventType = Literal[
    "fragment_received",
    "episode_embedded",
    "knowledge_extracted",
    "extraction_vetted",
    "concepts_resolved",
    "claims_created",
    "pedagogical_evidence_vetted",
    "relations_created",
    "ingestion_finalized",
    "ingestion_failed",
]
TraceEventRole = Literal["step", "decision"]
TraceStatus = Literal[
    "succeeded",
    "empty",
    "partial",
    "skipped",
    "fallback",
    "needs_review",
    "failed",
]
TraceBoundaryKind = Literal["llm", "embedding", "database"]


class TraceBoundaryPayload(BaseModel):
    kind: TraceBoundaryKind
    provider: str = ""
    model: str = ""
    request_text: str | None = None
    response_text: str | None = None
    request_json: dict[str, Any] | None = None
    response_json: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    event_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    parent_event_id: str | None = None
    sequence: int = Field(ge=1)
    type: TraceEventType
    role: TraceEventRole = "step"
    status: TraceStatus
    title: str = Field(min_length=1)
    summary: str = ""
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    detail: dict[str, Any] = Field(default_factory=dict)
    boundary_payload: TraceBoundaryPayload | None = None
    created_at: str


class TraceSummary(BaseModel):
    trace_id: str = Field(min_length=1)
    execution_type: TraceExecutionType
    execution_id: str = Field(min_length=1)
    episode_id: str | None = None
    status: TraceStatus
    started_at: str
    ended_at: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    total_steps: int = Field(default=0, ge=0)
    total_decisions: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)
    semantic_counts: dict[str, int] = Field(default_factory=dict)
    domain: str = ""

    @model_validator(mode="after")
    def validate_trace_identity(self) -> "TraceSummary":
        if self.trace_id == self.execution_id:
            raise ValueError("trace_id must differ from execution_id")
        return self


class CanonicalTrace(BaseModel):
    summary: TraceSummary
    events: list[TraceEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_event_ownership(self) -> "CanonicalTrace":
        trace_id = self.summary.trace_id
        event_ids = {event.event_id for event in self.events}
        for event in self.events:
            if event.trace_id != trace_id:
                raise ValueError("event trace_id must match summary trace_id")
            if event.parent_event_id and event.parent_event_id not in event_ids:
                raise ValueError("parent_event_id must reference an event in the same trace")
        return self
