from __future__ import annotations

from app.trace_models import CanonicalTrace, TraceEvent, TraceSummary


def _trace() -> CanonicalTrace:
    summary = TraceSummary(
        trace_id="tr_api",
        execution_type="ingestion_job",
        execution_id="job_api",
        episode_id="ep_api",
        status="succeeded",
        started_at="2026-06-17T00:00:00+00:00",
        ended_at="2026-06-17T00:00:01+00:00",
        duration_ms=1000,
        total_steps=1,
        status_counts={"succeeded": 1},
        semantic_counts={"created_concepts": 1},
        domain="General",
    )
    event = TraceEvent(
        event_id="te_api",
        trace_id="tr_api",
        sequence=1,
        type="ingestion_finalized",
        status="succeeded",
        title="Ingesta finalizada",
        summary="El job y el episodio quedaron cerrados con resumen final.",
        output={"created_concepts": 1},
        created_at="2026-06-17T00:00:01+00:00",
    )
    return CanonicalTrace(summary=summary, events=[event])


def test_trace_api_lists_reads_and_exports(client, auth_headers, store):
    import asyncio

    asyncio.run(store.persist_canonical_trace(_trace()))

    listed = client.get("/v1/traces", headers=auth_headers)
    assert listed.status_code == 200
    assert listed.json()[0]["trace_id"] == "tr_api"
    assert listed.json()[0]["execution_id"] == "job_api"

    detail = client.get("/v1/traces/tr_api", headers=auth_headers)
    assert detail.status_code == 200
    assert detail.json()["summary"]["trace_id"] == "tr_api"
    assert detail.json()["events"][0]["title"] == "Ingesta finalizada"

    exported = client.get("/v1/traces/tr_api/export", headers=auth_headers)
    assert exported.status_code == 200
    assert exported.headers["content-type"].startswith("text/plain")
    assert "Traza: tr_api" in exported.text
    assert "1. Ingesta finalizada [succeeded]" in exported.text


def test_trace_api_requires_auth_and_returns_404(client, auth_headers):
    unauthorized = client.get("/v1/traces")
    assert unauthorized.status_code == 401

    missing = client.get("/v1/traces/tr_missing", headers=auth_headers)
    assert missing.status_code == 404
