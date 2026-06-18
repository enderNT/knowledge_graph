from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.schema_bootstrap_traces import _build_commands
from app.store import InMemoryKnowledgeStore
from app.trace_models import CanonicalTrace, TraceEvent, TraceSummary
from app.trace_recorder import TraceRecorder


def _event(**overrides):
    base = {
        "event_id": "te_child",
        "trace_id": "tr_123",
        "parent_event_id": "te_parent",
        "sequence": 2,
        "type": "knowledge_extracted",
        "role": "decision",
        "status": "succeeded",
        "title": "Conocimiento extraido",
        "created_at": "2026-06-17T00:00:00+00:00",
    }
    base.update(overrides)
    return TraceEvent.model_validate(base)


def _summary(**overrides):
    base = {
        "trace_id": "tr_123",
        "execution_type": "ingestion_job",
        "execution_id": "job_123",
        "episode_id": "ep_123",
        "status": "succeeded",
        "started_at": "2026-06-17T00:00:00+00:00",
    }
    base.update(overrides)
    return TraceSummary.model_validate(base)


def test_trace_summary_rejects_execution_id_as_trace_id():
    with pytest.raises(ValidationError, match="trace_id must differ from execution_id"):
        _summary(trace_id="job_123", execution_id="job_123")


def test_trace_event_rejects_unknown_type_or_status():
    with pytest.raises(ValidationError):
        _event(type="raw_log_line")
    with pytest.raises(ValidationError):
        _event(status="INFO")


def test_canonical_trace_requires_explicit_parent_identity():
    parent = _event(event_id="te_parent", parent_event_id=None, sequence=1, role="step")
    child = _event()
    trace = CanonicalTrace(summary=_summary(), events=[parent, child])
    assert trace.events[1].parent_event_id == parent.event_id

    orphan = _event(parent_event_id="missing_parent")
    with pytest.raises(ValidationError, match="parent_event_id must reference"):
        CanonicalTrace(summary=_summary(), events=[orphan])


def test_trace_schema_bootstrap_declares_canonical_types():
    commands = _build_commands()
    assert "CREATE DOCUMENT TYPE CanonicalTrace IF NOT EXISTS" in commands
    assert "CREATE DOCUMENT TYPE CanonicalTraceEvent IF NOT EXISTS" in commands
    assert "CREATE INDEX ON CanonicalTrace (trace_id) UNIQUE" in commands
    assert "CREATE INDEX ON CanonicalTraceEvent (event_id) UNIQUE" in commands


@pytest.mark.asyncio
async def test_in_memory_store_persists_and_lists_canonical_trace():
    store = InMemoryKnowledgeStore(
        Settings(
            app_env="test",
            API_KEY="test-api-key",
            ARCADEDB_ROOT_PASSWORD="test-password",
            embedding_dimensions=16,
        )
    )
    parent = _event(event_id="te_parent", parent_event_id=None, sequence=2, role="step")
    child = _event(event_id="te_child", sequence=1)
    summary = _summary(status_counts={"succeeded": 2}, total_steps=1, total_decisions=1)
    await store.persist_canonical_trace(CanonicalTrace(summary=summary, events=[parent, child]))

    trace = await store.get_canonical_trace("tr_123")
    assert trace is not None
    assert [event.sequence for event in trace.events] == [1, 2]

    listed = await store.list_canonical_traces(execution_id="job_123")
    assert [item.trace_id for item in listed] == ["tr_123"]
    assert listed[0].status_counts == {"succeeded": 2}


def test_trace_recorder_records_monotonic_sequence_and_close_summary():
    recorder = TraceRecorder(
        trace_id="tr_456",
        execution_type="ingestion_job",
        execution_id="job_456",
        episode_id="ep_456",
    )
    parent = recorder.record_step(
        type="knowledge_extracted",
        status="succeeded",
        title="Conocimiento extraido",
        output={"concepts": 2},
    )
    recorder.record_decision(
        parent_event_id=parent.event_id,
        type="knowledge_extracted",
        status="needs_review",
        title="Concepto requiere revision",
    )
    trace = recorder.close(status="partial", domain="Psicologia", semantic_counts={"concepts": 2})

    assert [event.sequence for event in trace.events] == [1, 2]
    assert trace.summary.total_steps == 1
    assert trace.summary.total_decisions == 1
    assert trace.summary.status_counts == {"succeeded": 1, "needs_review": 1}
    assert trace.summary.semantic_counts == {"concepts": 2}
    assert trace.summary.domain == "Psicologia"


def test_trace_recorder_requires_existing_parent_and_rejects_write_after_close():
    recorder = TraceRecorder(
        trace_id="tr_789",
        execution_type="ingestion_job",
        execution_id="job_789",
    )
    with pytest.raises(ValueError, match="parent_event_id must reference"):
        recorder.record_decision(
            parent_event_id="missing",
            type="knowledge_extracted",
            status="failed",
            title="Decision huerfana",
        )
    recorder.close(status="empty")
    with pytest.raises(RuntimeError, match="trace recorder is closed"):
        recorder.record_step(
            type="ingestion_finalized",
            status="succeeded",
            title="Ingesta finalizada",
        )
