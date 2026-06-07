from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.mcp_backend_client import MCPBackendError
from app.mcp_server import create_app, create_mcp_server


class FakeBackendClient:
    def __init__(self) -> None:
        self.ready = (True, {"status": "ready"})
        self.fragment_result = {
            "status": "completed",
            "episode_id": "ep_1",
            "job_id": "job_1",
            "result": {"episode_id": "ep_1", "domain": "Psicología"},
        }
        self.search_result = {"query": "memoria", "results": []}
        self.learning_context_result = {
            "query": "memoria",
            "domain_hint": "Psicología",
            "status": "sparse",
            "primary_concepts": [
                {
                    "uid": "cn_1",
                    "canonical_name": "Memoria episódica",
                    "domain": "Psicología",
                    "description": "Sistema de memoria autobiográfica.",
                    "retrieval_score": 0.98,
                    "retrieval_reason": "alias",
                    "quality_flags": [],
                }
            ],
            "relations": [],
            "claims": [],
            "episodes": [],
            "warnings": ["no_supporting_claims"],
            "debug": {
                "candidate_count": 1,
                "selected_concept_uids": ["cn_1"],
                "selection_reasons": ["cn_1:alias:0.98:exact_or_alias_match"],
            },
        }
        self.tutor_context_result = {
            "resolved_reference": {
                "input_type": "query",
                "input_value": "memoria",
                "resolved_concept_uid": "cn_1",
                "resolved_concept_name": "Memoria episódica",
                "resolved_episode_id": None,
                "resolved_job_id": None,
                "resolution_reason": "alias",
            },
            "status": "ok",
            "concepts": [
                {
                    "uid": "cn_1",
                    "canonical_name": "Memoria episódica",
                    "domain": "Psicología",
                    "description": "Sistema de memoria autobiográfica.",
                    "aliases": ["recuerdo autobiográfico"],
                }
            ],
            "claims": [
                {
                    "uid": "cl_1",
                    "text": "La memoria episódica recupera experiencias personales con contexto temporal.",
                    "confidence": 0.91,
                    "evidence_episode_ids": ["ep_1"],
                }
            ],
            "relations": [],
            "source_fragments": [
                {
                    "episode_id": "ep_1",
                    "text": "Fragmento base.",
                    "status": "processed",
                    "source_type": "manual_input",
                    "tags": ["Psicología"],
                    "language": "es",
                }
            ],
            "evidence": [{"subject_type": "claim", "subject_uid": "cl_1", "episode_id": "ep_1"}],
            "warnings": [],
            "failure_reason": None,
        }
        self.upsert_result = {"concept": {"uid": "cn_1"}, "created": True}
        self.create_result = {"concept": {"uid": "cn_strict"}, "created": True}
        self.attach_evidence_result = {"status": "attached", "concept_uid": "cn_1", "episode_id": "ep_1", "linked_claim_count": 2}
        self.link_result = {"status": "linked"}
        self.neighborhood_result = {"concept": {"uid": "cn_1"}, "nodes": [], "relations": [], "claims": [], "episodes": []}
        self.pedagogical_context_result = {
            "user_id": "user-1",
            "status": "ok",
            "concepts": [],
            "domains": [],
            "recent_evaluations": [],
            "warnings": [],
        }
        self.pedagogical_update_result = {
            "user_id": "user-1",
            "status": "ok",
            "context": self.pedagogical_context_result,
            "session_view": {
                "user_id": "user-1",
                "status": "ok",
                "summary": "Priorizar conceptos debiles.",
                "weak_concepts": [],
                "detected_gaps": [],
                "suggested_questions": [],
                "effective_depth_used": 3,
                "domain_focus": [],
                "recalculation_traces": [],
                "warnings": [],
            },
            "warnings": [],
        }
        self.pedagogical_session_view_result = self.pedagogical_update_result["session_view"]
        self.sr_state_result = {
            "state": {
                "user_id": "user-1",
                "concept_uid": "cn_1",
                "dimension": "recognition",
                "repetitions": 1,
                "ease_factor": 2.5,
                "interval_days": 1,
                "last_reviewed_at": "2026-06-06T10:00:00+00:00",
                "next_review_at": "2026-06-07T10:00:00+00:00",
                "propagation_relief_count": 0,
                "requires_direct_validation": False,
                "updated_at": "2026-06-06T10:00:00+00:00",
            }
        }
        self.due_sr_items_result = {"user_id": "user-1", "items": []}
        self.sr_update_result = {
            "state": self.sr_state_result["state"],
            "sr_feedback": {
                "concept_uid": "cn_1",
                "dimension": "recognition",
                "calculated_quality_q": 5,
                "rationale": "Respuesta correcta directa sin apoyo",
            },
            "ef_bonus_applied": False,
        }
        self.sr_relief_result = {"updated_states": []}
        self.sr_stats_result = {
            "user_id": "user-1",
            "stats": {
                "total_items": 1,
                "due_items": 0,
                "forced_review_items": 0,
                "average_ease_factor": 2.5,
            },
        }
        self.adaptive_session_result = {
            "session": {
                "session_id": "ads_1",
                "user_id": "user-1",
                "resolved_reference": self.tutor_context_result["resolved_reference"],
                "domain_hint": "Psicología",
                "language": "es",
                "constraints": {"max_items_per_block": 3, "max_blocks": 4, "allowed_question_types": ["multiple_choice_single", "multiple_choice_multi", "true_false", "cloze", "open"], "preferred_max_difficulty": None, "allow_scaffolding": True},
                "tutor_context": self.tutor_context_result,
                "current_block": {
                    "block_id": "blk_1",
                    "plan": {
                        "block_id": "blk_1",
                        "block_goal": "reinforce_weak",
                        "target_concept_uid": "cn_1",
                        "target_concept_name": "Memoria episódica",
                        "target_dimensions": ["recognition"],
                        "recommended_question_types": ["multiple_choice_single", "true_false"],
                        "difficulty": "intermediate",
                        "scaffolding": {
                            "allow_hint_after_error": True,
                            "allow_rephrase_retry": True,
                            "allow_difficulty_drop_next_item": False,
                            "show_corrective_explanation_at_end": True,
                        },
                        "success_criteria": {
                            "min_block_score": 0.7,
                            "min_dimension_signal": 0.65,
                            "max_supported_answers_ratio": 0.5,
                        },
                        "next_step_policy": {
                            "on_success": "shift_dimension_within_concept_then_reprioritize",
                            "on_partial_high": "stay_local_with_light_support",
                            "on_partial_low": "decrease_difficulty_and_return_to_basic_dimension",
                            "on_failure": "decrease_difficulty_and_move_to_related_or_prerequisite",
                        },
                        "planner_explanation": "priorizar reconocimiento",
                    },
                    "items": [],
                    "answer_keys": [],
                    "generated_at": "2026-06-06T10:00:00+00:00",
                },
                "block_history": [],
                "summary": {
                    "total_blocks": 4,
                    "completed_blocks": 0,
                    "latest_block_verdict": None,
                    "session_closed": False,
                    "closure_reason": None,
                },
                "status": "active",
                "opened_at": "2026-06-06T10:00:00+00:00",
                "updated_at": "2026-06-06T10:00:00+00:00",
            },
            "current_block": {
                "block_id": "blk_1",
                "plan": {
                    "block_id": "blk_1",
                    "block_goal": "reinforce_weak",
                    "target_concept_uid": "cn_1",
                    "target_concept_name": "Memoria episódica",
                    "target_dimensions": ["recognition"],
                    "recommended_question_types": ["multiple_choice_single", "true_false"],
                    "difficulty": "intermediate",
                    "scaffolding": {
                        "allow_hint_after_error": True,
                        "allow_rephrase_retry": True,
                        "allow_difficulty_drop_next_item": False,
                        "show_corrective_explanation_at_end": True,
                    },
                    "success_criteria": {
                        "min_block_score": 0.7,
                        "min_dimension_signal": 0.65,
                        "max_supported_answers_ratio": 0.5,
                    },
                    "next_step_policy": {
                        "on_success": "shift_dimension_within_concept_then_reprioritize",
                        "on_partial_high": "stay_local_with_light_support",
                        "on_partial_low": "decrease_difficulty_and_return_to_basic_dimension",
                        "on_failure": "decrease_difficulty_and_move_to_related_or_prerequisite",
                    },
                    "planner_explanation": "priorizar reconocimiento",
                },
                "items": [],
                "answer_keys": [],
                "generated_at": "2026-06-06T10:00:00+00:00",
            },
            "planner_explanation": "priorizar reconocimiento",
            "grounding_status": "ok",
        }
        self.adaptive_submit_result = {
            "session": {
                **self.adaptive_session_result["session"],
                "current_block": None,
                "block_history": [
                    {
                        "block_id": "blk_1",
                        "item_results": [],
                        "dimension_summary": {"recognition": 1.0},
                        "block_verdict": "correct",
                        "block_score": 1.0,
                        "recommended_next_action": "shift_dimension_within_concept_then_reprioritize",
                        "corrective_explanation": "explicacion",
                        "transition_explanation": "transicion",
                    }
                ],
                "summary": {
                    "total_blocks": 4,
                    "completed_blocks": 1,
                    "latest_block_verdict": "correct",
                    "session_closed": True,
                    "closure_reason": "coverage_sufficient",
                },
                "status": "closed",
            },
            "block_result": {
                "block_id": "blk_1",
                "item_results": [],
                "dimension_summary": {"recognition": 1.0},
                "block_verdict": "correct",
                "block_score": 1.0,
                "recommended_next_action": "shift_dimension_within_concept_then_reprioritize",
                "corrective_explanation": "explicacion",
                "transition_explanation": "transicion",
            },
            "updated_context": self.pedagogical_context_result,
            "next_action": "shift_dimension_within_concept_then_reprioritize",
            "next_block": None,
            "session_closed": True,
        }
        self.link_error: Exception | None = None

    async def ingest_fragment_and_wait(self, **_: Any) -> dict[str, Any]:
        return self.fragment_result

    async def search_candidates(self, **_: Any) -> dict[str, Any]:
        return self.search_result

    async def get_learning_context(self, **_: Any) -> dict[str, Any]:
        return self.learning_context_result

    async def get_tutor_context(self, **_: Any) -> dict[str, Any]:
        return self.tutor_context_result

    async def create_concept(self, **_: Any) -> dict[str, Any]:
        return self.create_result

    async def attach_concept_evidence(self, **_: Any) -> dict[str, Any]:
        return self.attach_evidence_result

    async def upsert_concept(self, **_: Any) -> dict[str, Any]:
        return self.upsert_result

    async def link_concepts(self, **_: Any) -> dict[str, Any]:
        if self.link_error:
            raise self.link_error
        return self.link_result

    async def get_neighborhood(self, **_: Any) -> dict[str, Any]:
        return self.neighborhood_result

    async def get_pedagogical_context(self, **_: Any) -> dict[str, Any]:
        return self.pedagogical_context_result

    async def update_pedagogical_context(self, **_: Any) -> dict[str, Any]:
        return self.pedagogical_update_result

    async def get_pedagogical_session_view(self, **_: Any) -> dict[str, Any]:
        return self.pedagogical_session_view_result

    async def get_sr_state(self, **_: Any) -> dict[str, Any]:
        return self.sr_state_result

    async def get_due_sr_items(self, **_: Any) -> dict[str, Any]:
        return self.due_sr_items_result

    async def update_sr_from_block_result(self, **_: Any) -> dict[str, Any]:
        return self.sr_update_result

    async def apply_prereq_relief(self, **_: Any) -> dict[str, Any]:
        return self.sr_relief_result

    async def get_sr_stats(self, **_: Any) -> dict[str, Any]:
        return self.sr_stats_result

    async def start_adaptive_session(self, **_: Any) -> dict[str, Any]:
        return self.adaptive_session_result

    async def submit_adaptive_block(self, **_: Any) -> dict[str, Any]:
        return self.adaptive_submit_result

    async def get_adaptive_session(self, **_: Any) -> dict[str, Any]:
        return self.adaptive_session_result["session"]

    async def check_ready(self) -> tuple[bool, dict[str, Any]]:
        return self.ready

    async def add_knowledge_fragment(self, **_: Any) -> dict[str, Any]:
        raise NotImplementedError

    async def close(self) -> None:
        return None


def _settings() -> Settings:
    return Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        MCP_BEARER_TOKEN="test-bearer-token",
        KG_API_KEY="internal-api-key",
        KG_API_BASE_URL="http://backend.test",
    )


def test_mcp_http_requires_bearer_token_for_mcp_only():
    backend = FakeBackendClient()
    app = create_app(settings=_settings(), backend_client=backend)

    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200

        missing = client.get("/mcp")
        assert missing.status_code == 401
        assert missing.json() == {"detail": "invalid bearer token"}

        invalid = client.post("/mcp", headers={"Authorization": "Bearer nope"}, json={})
        assert invalid.status_code == 401
        assert invalid.json() == {"detail": "invalid bearer token"}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client_session() -> AsyncGenerator[Any]:
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session

    server = create_mcp_server(settings=_settings(), backend_client=FakeBackendClient())
    async with create_connected_server_and_client_session(server, raise_exceptions=True) as session:
        yield session


@pytest.mark.anyio
async def test_mcp_server_exposes_exactly_twenty_tools(client_session):
    tools = await client_session.list_tools()

    assert sorted(tool.name for tool in tools.tools) == [
        "add_knowledge_fragment",
        "apply_prereq_relief",
        "attach_concept_evidence",
        "create_concept",
        "get_adaptive_session",
        "get_due_sr_items",
        "get_learning_context",
        "get_neighborhood",
        "get_pedagogical_context",
        "get_pedagogical_session_view",
        "get_sr_state",
        "get_sr_stats",
        "get_tutor_context",
        "link_concepts",
        "search_candidates",
        "start_adaptive_session",
        "submit_adaptive_block",
        "update_pedagogical_context",
        "update_sr_from_block_result",
        "upsert_concept",
    ]

    tool_map = {tool.name: tool for tool in tools.tools}
    assert tool_map["add_knowledge_fragment"].inputSchema["properties"]["text"]["type"] == "string"
    assert tool_map["search_candidates"].inputSchema["properties"]["limit"]["default"] == 10
    assert tool_map["get_learning_context"].inputSchema["properties"]["candidate_limit"]["default"] == 8
    assert tool_map["get_pedagogical_context"].inputSchema["properties"]["user_id"]["type"] == "string"
    assert tool_map["get_sr_state"].inputSchema["properties"]["dimension"]["type"] == "string"
    assert tool_map["get_tutor_context"].inputSchema["properties"]["include_evidence"]["default"] is True
    assert tool_map["start_adaptive_session"].inputSchema["properties"]["language"]["default"] == "es"
    assert "from" in tool_map["link_concepts"].inputSchema["properties"]
    assert "depth" in tool_map["get_neighborhood"].inputSchema["properties"]


@pytest.mark.anyio
async def test_mcp_server_translates_tool_results_and_errors():
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session

    backend = FakeBackendClient()
    server = create_mcp_server(settings=_settings(), backend_client=backend)
    async with create_connected_server_and_client_session(server, raise_exceptions=True) as session:
        fragment = await session.call_tool("add_knowledge_fragment", {"text": "Texto"})
        assert fragment.isError in {False, None}
        assert fragment.structuredContent["status"] == "completed"
        assert fragment.structuredContent["job_id"] == "job_1"

        learning_context = await session.call_tool(
            "get_learning_context",
            {"query": "memoria", "domain_hint": "Psicología"},
        )
        assert learning_context.isError in {False, None}
        assert learning_context.structuredContent["status"] == "sparse"
        assert learning_context.structuredContent["primary_concepts"][0]["uid"] == "cn_1"

        tutor_context = await session.call_tool(
            "get_tutor_context",
            {"query": "memoria"},
        )
        assert tutor_context.isError in {False, None}
        assert tutor_context.structuredContent["status"] == "ok"
        assert tutor_context.structuredContent["resolved_reference"]["resolved_concept_uid"] == "cn_1"

        adaptive_start = await session.call_tool(
            "start_adaptive_session",
            {"user_id": "user-1", "query": "memoria"},
        )
        assert adaptive_start.isError in {False, None}
        assert adaptive_start.structuredContent["grounding_status"] == "ok"

        adaptive_submit = await session.call_tool(
            "submit_adaptive_block",
            {"session_id": "ads_1", "block_id": "blk_1", "submissions": [{"item_id": "blk_1_item_1", "selected_choices": [0]}]},
        )
        assert adaptive_submit.isError in {False, None}
        assert adaptive_submit.structuredContent["session_closed"] is True

        pedagogical_context = await session.call_tool(
            "get_pedagogical_context",
            {"user_id": "user-1"},
        )
        assert pedagogical_context.isError in {False, None}
        assert pedagogical_context.structuredContent["user_id"] == "user-1"

        sr_state = await session.call_tool(
            "get_sr_state",
            {"user_id": "user-1", "concept_uid": "cn_1", "dimension": "recognition"},
        )
        assert sr_state.isError in {False, None}
        assert sr_state.structuredContent["state"]["dimension"] == "recognition"

        pedagogical_update = await session.call_tool(
            "update_pedagogical_context",
            {
                "user_id": "user-1",
                "evaluations": [{"concept_uid": "cn_1", "score_0_to_100": 72}],
            },
        )
        assert pedagogical_update.isError in {False, None}
        assert pedagogical_update.structuredContent["session_view"]["effective_depth_used"] == 3

        create_concept = await session.call_tool(
            "create_concept",
            {"canonical_name": "Memoria", "domain": "Psicología"},
        )
        assert create_concept.isError in {False, None}
        assert create_concept.structuredContent["concept"]["uid"] == "cn_strict"

        attach = await session.call_tool(
            "attach_concept_evidence",
            {"concept_ref": "cn_1", "episode_id": "ep_1", "link_episode_claims": True},
        )
        assert attach.isError in {False, None}
        assert attach.structuredContent["linked_claim_count"] == 2

        backend.link_error = MCPBackendError("source or target concept not found", status_code=404)
        link = await session.call_tool(
            "link_concepts",
            {"from": "cn_1", "relation": "RELATED_TO", "to": "cn_2"},
        )
        assert link.isError is True
        assert any(
            "source or target concept not found" in getattr(content, "text", "") for content in link.content
        )


@pytest.mark.anyio
async def test_mcp_streamable_http_client_works_with_documented_url():
    pytest.importorskip("mcp")
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    app = create_app(settings=_settings(), backend_client=FakeBackendClient())
    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"Authorization": "Bearer test-bearer-token"},
        ) as http_client:
            async with streamable_http_client(
                "http://testserver/mcp",
                http_client=http_client,
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await session.list_tools()

    assert sorted(tool.name for tool in tools.tools) == [
        "add_knowledge_fragment",
        "apply_prereq_relief",
        "attach_concept_evidence",
        "create_concept",
        "get_adaptive_session",
        "get_due_sr_items",
        "get_learning_context",
        "get_neighborhood",
        "get_pedagogical_context",
        "get_pedagogical_session_view",
        "get_sr_state",
        "get_sr_stats",
        "get_tutor_context",
        "link_concepts",
        "search_candidates",
        "start_adaptive_session",
        "submit_adaptive_block",
        "update_pedagogical_context",
        "update_sr_from_block_result",
        "upsert_concept",
    ]
