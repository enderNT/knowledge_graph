from __future__ import annotations

import json

import httpx
import pytest

from app.config import Settings
from app.mcp_backend_client import (
    MCPBackendAuthError,
    MCPBackendClient,
    MCPBackendError,
    MCPBackendNotFoundError,
)


def _settings(**overrides) -> Settings:
    base_kwargs = {
        "app_env": "test",
        "API_KEY": "test-api-key",
        "ARCADEDB_ROOT_PASSWORD": "test-password",
        "KG_API_KEY": "internal-api-key",
        "KG_API_BASE_URL": "http://backend.test",
        "MCP_POLL_INTERVAL_SECONDS": 0.0,
        "MCP_INGESTION_TIMEOUT_SECONDS": 0.01,
    }
    base_kwargs.update(overrides)
    return Settings(**base_kwargs)


def _response(request: httpx.Request, status_code: int, payload: dict) -> httpx.Response:
    return httpx.Response(status_code=status_code, json=payload, request=request)


@pytest.mark.anyio
async def test_ingestion_returns_completed_job_result():
    calls = {"jobs": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/knowledge/fragments":
            return _response(request, 202, {"episode_id": "ep_1", "job_id": "job_1", "status": "queued"})
        if request.url.path == "/v1/jobs/job_1":
            calls["jobs"] += 1
            if calls["jobs"] == 1:
                return _response(request, 200, {"uid": "job_1", "episode_id": "ep_1", "status": "processing"})
            return _response(
                request,
                200,
                {
                    "uid": "job_1",
                    "episode_id": "ep_1",
                    "status": "completed",
                    "result": {"episode_id": "ep_1", "domain": "Psicología", "relations": []},
                },
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = MCPBackendClient(_settings(), transport=httpx.MockTransport(handler))
    try:
        result = await client.ingest_fragment_and_wait(text="Texto")
    finally:
        await client.close()

    assert result == {
        "status": "completed",
        "episode_id": "ep_1",
        "job_id": "job_1",
        "result": {"episode_id": "ep_1", "domain": "Psicología", "relations": []},
    }


@pytest.mark.anyio
async def test_ingestion_returns_failed_status_without_raising():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/knowledge/fragments":
            return _response(request, 202, {"episode_id": "ep_1", "job_id": "job_1", "status": "queued"})
        if request.url.path == "/v1/jobs/job_1":
            return _response(
                request,
                200,
                {"uid": "job_1", "episode_id": "ep_1", "status": "failed", "error": "llm extraction failed"},
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = MCPBackendClient(_settings(), transport=httpx.MockTransport(handler))
    try:
        result = await client.ingest_fragment_and_wait(text="Texto")
    finally:
        await client.close()

    assert result == {
        "status": "failed",
        "episode_id": "ep_1",
        "job_id": "job_1",
        "error": "llm extraction failed",
    }


@pytest.mark.anyio
async def test_ingestion_timeout_returns_processing_status():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/knowledge/fragments":
            return _response(request, 202, {"episode_id": "ep_1", "job_id": "job_1", "status": "queued"})
        if request.url.path == "/v1/jobs/job_1":
            return _response(request, 200, {"uid": "job_1", "episode_id": "ep_1", "status": "processing"})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = MCPBackendClient(_settings(MCP_INGESTION_TIMEOUT_SECONDS=0.0), transport=httpx.MockTransport(handler))
    try:
        result = await client.ingest_fragment_and_wait(text="Texto")
    finally:
        await client.close()

    assert result == {"status": "processing", "episode_id": "ep_1", "job_id": "job_1"}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (401, MCPBackendAuthError),
        (404, MCPBackendNotFoundError),
        (500, MCPBackendError),
    ],
)
async def test_backend_error_translation(status_code: int, error_type: type[Exception]):
    async def handler(request: httpx.Request) -> httpx.Response:
        return _response(request, status_code, {"detail": f"error-{status_code}"})

    client = MCPBackendClient(_settings(), transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(error_type) as exc_info:
            await client.search_candidates(query="memoria")
    finally:
        await client.close()

    assert str(exc_info.value) == f"error-{status_code}"


@pytest.mark.anyio
async def test_learning_context_request_and_response_shape():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/search/learning-context"
        assert json.loads(request.content.decode()) == {
            "query": "memoria",
            "domain_hint": "Psicología",
            "candidate_limit": 4,
            "concept_limit": 2,
            "claim_limit": 5,
            "episode_limit": 2,
            "include_neighborhood": True,
            "depth": 1,
        }
        return _response(
            request,
            200,
            {
                "query": "memoria",
                "domain_hint": "Psicología",
                "status": "no_match",
                "primary_concepts": [],
                "relations": [],
                "claims": [],
                "episodes": [],
                "warnings": ["no_candidates_found"],
                "debug": {
                    "candidate_count": 0,
                    "selected_concept_uids": [],
                    "selection_reasons": [],
                },
            },
        )

    client = MCPBackendClient(_settings(), transport=httpx.MockTransport(handler))
    try:
        result = await client.get_learning_context(
            query="memoria",
            domain_hint="Psicología",
            candidate_limit=4,
            concept_limit=2,
            claim_limit=5,
            episode_limit=2,
            include_neighborhood=True,
            depth=1,
        )
    finally:
        await client.close()

    assert result["status"] == "no_match"
    assert result["debug"]["candidate_count"] == 0


@pytest.mark.anyio
async def test_create_concept_request_shape():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/concepts"
        assert json.loads(request.content.decode()) == {
            "canonical_name": "Sistema Binario",
            "aliases": ["sistema binario"],
            "domain": "Programacion Cobol",
            "description": "Base 2.",
        }
        return _response(request, 200, {"concept": {"uid": "cn_1"}, "created": True})

    client = MCPBackendClient(_settings(), transport=httpx.MockTransport(handler))
    try:
        result = await client.create_concept(
            canonical_name="Sistema Binario",
            aliases=["sistema binario"],
            domain="Programacion Cobol",
            description="Base 2.",
        )
    finally:
        await client.close()

    assert result["concept"]["uid"] == "cn_1"


@pytest.mark.anyio
async def test_attach_concept_evidence_request_shape():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/concepts/evidence"
        assert json.loads(request.content.decode()) == {
            "concept_ref": "cn_1",
            "episode_id": "ep_1",
            "link_episode_claims": True,
        }
        return _response(request, 200, {"status": "attached", "concept_uid": "cn_1", "episode_id": "ep_1", "linked_claim_count": 2})

    client = MCPBackendClient(_settings(), transport=httpx.MockTransport(handler))
    try:
        result = await client.attach_concept_evidence(
            concept_ref="cn_1",
            episode_id="ep_1",
            link_episode_claims=True,
        )
    finally:
        await client.close()

    assert result["linked_claim_count"] == 2


@pytest.mark.anyio
async def test_upsert_concept_sends_optional_uid():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PUT"
        assert request.url.path == "/v1/concepts/upsert"
        assert json.loads(request.content.decode()) == {
            "uid": "cn_existing",
            "canonical_name": "Sistema Binario",
            "aliases": ["sistema binario"],
            "domain": "Programacion Cobol",
            "description": "Base 2.",
        }
        return _response(request, 200, {"concept": {"uid": "cn_existing"}, "created": False})

    client = MCPBackendClient(_settings(), transport=httpx.MockTransport(handler))
    try:
        result = await client.upsert_concept(
            uid="cn_existing",
            canonical_name="Sistema Binario",
            aliases=["sistema binario"],
            domain="Programacion Cobol",
            description="Base 2.",
        )
    finally:
        await client.close()

    assert result["created"] is False


@pytest.mark.anyio
async def test_tutor_context_request_and_response_shape():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/search/tutor-context"
        assert json.loads(request.content.decode()) == {
            "query": None,
            "episode_id": "ep_1",
            "job_id": None,
            "depth": 1,
            "include_evidence": True,
        }
        return _response(
            request,
            200,
            {
                "resolved_reference": {
                    "input_type": "episode_id",
                    "input_value": "ep_1",
                    "resolved_concept_uid": None,
                    "resolved_concept_name": None,
                    "resolved_episode_id": "ep_1",
                    "resolved_job_id": None,
                    "resolution_reason": None,
                },
                "status": "failed",
                "concepts": [],
                "claims": [],
                "relations": [],
                "source_fragments": [],
                "evidence": [],
                "warnings": ["no_traceable_claims"],
                "failure_reason": "insufficient_traceable_evidence",
            },
        )

    client = MCPBackendClient(_settings(), transport=httpx.MockTransport(handler))
    try:
        result = await client.get_tutor_context(episode_id="ep_1")
    finally:
        await client.close()

    assert result["status"] == "failed"
    assert result["failure_reason"] == "insufficient_traceable_evidence"


@pytest.mark.anyio
async def test_spaced_repetition_request_shapes():
    calls: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/v1/sr/state":
            assert request.url.params["user_id"] == "user-1"
            assert request.url.params["concept_uid"] == "cn_1"
            assert request.url.params["dimension"] == "recall"
            return _response(request, 200, {"state": {"user_id": "user-1", "concept_uid": "cn_1", "dimension": "recall", "repetitions": 0, "ease_factor": 2.5, "interval_days": 0, "last_reviewed_at": None, "next_review_at": "2026-06-07T10:00:00+00:00", "propagation_relief_count": 0, "requires_direct_validation": False, "updated_at": "2026-06-07T10:00:00+00:00"}})
        if request.url.path == "/v1/sr/due":
            assert request.url.params["user_id"] == "user-1"
            return _response(request, 200, {"user_id": "user-1", "items": []})
        if request.url.path == "/v1/sr/update":
            assert json.loads(request.content.decode()) == {
                "user_id": "user-1",
                "concept_uid": "cn_1",
                "dimension": "recall",
                "block_verdict": "correct",
                "block_difficulty": "intermediate",
                "hint_used": False,
                "retry_used": False,
                "coverage": 1.0,
                "precision": 1.0,
                "was_direct_evaluation": True,
            }
            return _response(request, 200, {"state": {"user_id": "user-1", "concept_uid": "cn_1", "dimension": "recall", "repetitions": 1, "ease_factor": 2.6, "interval_days": 1, "last_reviewed_at": "2026-06-07T10:00:00+00:00", "next_review_at": "2026-06-08T10:00:00+00:00", "propagation_relief_count": 0, "requires_direct_validation": False, "updated_at": "2026-06-07T10:00:00+00:00"}, "sr_feedback": {"concept_uid": "cn_1", "dimension": "recall", "calculated_quality_q": 5, "rationale": "ok"}, "ef_bonus_applied": False})
        if request.url.path == "/v1/sr/relief":
            assert json.loads(request.content.decode()) == {
                "user_id": "user-1",
                "source_concept_uid": "cn_2",
                "source_dimension": "application",
                "quality_q": 4,
            }
            return _response(request, 200, {"updated_states": []})
        if request.url.path == "/v1/sr/stats":
            assert request.url.params["user_id"] == "user-1"
            return _response(request, 200, {"user_id": "user-1", "stats": {"total_items": 1, "due_items": 0, "forced_review_items": 0, "average_ease_factor": 2.6}})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = MCPBackendClient(_settings(), transport=httpx.MockTransport(handler))
    try:
        await client.get_sr_state(user_id="user-1", concept_uid="cn_1", dimension="recall")
        await client.get_due_sr_items(user_id="user-1")
        await client.update_sr_from_block_result(
            user_id="user-1",
            concept_uid="cn_1",
            dimension="recall",
            block_verdict="correct",
            block_difficulty="intermediate",
            coverage=1.0,
            precision=1.0,
        )
        await client.apply_prereq_relief(
            user_id="user-1",
            source_concept_uid="cn_2",
            source_dimension="application",
            quality_q=4,
        )
        stats = await client.get_sr_stats(user_id="user-1")
    finally:
        await client.close()

    assert stats["stats"]["average_ease_factor"] == 2.6
    assert calls == [
        ("GET", "/v1/sr/state"),
        ("GET", "/v1/sr/due"),
        ("POST", "/v1/sr/update"),
        ("POST", "/v1/sr/relief"),
        ("GET", "/v1/sr/stats"),
    ]
