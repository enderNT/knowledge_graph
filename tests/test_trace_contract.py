from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.trace_models import CanonicalTrace, TraceEvent, TraceSummary


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
