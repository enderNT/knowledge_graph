from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest

from app.agent_mcp_client import AgentMCPUpstreamClient
from app.agent_mcp_server import create_agent_app, create_agent_mcp_server
from app.config import Settings
from app.mcp_server import create_app as create_knowledge_mcp_app


class FakeKnowledgeBackendClient:
    def __init__(self) -> None:
        self.ready = (True, {"status": "ready"})
        self.last_link_call: dict[str, Any] | None = None

    async def ingest_fragment_and_wait(self, **_: Any) -> dict[str, Any]:
        return {"status": "completed", "episode_id": "ep_1", "job_id": "job_1", "result": {"episode_id": "ep_1"}}

    async def search_candidates(self, **kwargs: Any) -> dict[str, Any]:
        return {"query": kwargs["query"], "results": []}

    async def get_learning_context(self, **kwargs: Any) -> dict[str, Any]:
        query = kwargs["query"]
        if query == "tema inexistente":
            return {
                "query": query,
                "domain_hint": kwargs.get("domain_hint"),
                "status": "no_match",
                "primary_concepts": [],
                "relations": [],
                "claims": [],
                "episodes": [],
                "warnings": ["no_candidates_found"],
                "debug": {"candidate_count": 0, "selected_concept_uids": [], "selection_reasons": []},
            }
        if query == "memoria":
            return {
                "query": query,
                "domain_hint": kwargs.get("domain_hint"),
                "status": "sparse",
                "primary_concepts": [
                    {
                        "uid": "cn_mem_1",
                        "canonical_name": "Memoria episódica",
                        "domain": "Psicología",
                        "description": "Sistema de memoria autobiográfica con contexto temporal.",
                        "retrieval_score": 0.98,
                        "retrieval_reason": "alias",
                        "quality_flags": [],
                    }
                ],
                "relations": [],
                "claims": [
                    {
                        "uid": "cl_1",
                        "text": "La memoria episódica recupera experiencias personales con contexto temporal.",
                        "confidence": 0.93,
                    }
                ],
                "episodes": [],
                "warnings": ["no_source_episodes"],
                "debug": {"candidate_count": 1, "selected_concept_uids": ["cn_mem_1"], "selection_reasons": ["cn_mem_1"]},
            }
        return {
            "query": query,
            "domain_hint": kwargs.get("domain_hint"),
            "status": "ok",
            "primary_concepts": [
                {
                    "uid": "cn_1",
                    "canonical_name": "Condicionamiento clásico",
                    "domain": "Psicología",
                    "description": "Aprendizaje asociativo donde un estimulo adquiere valor predictivo.",
                    "retrieval_score": 1.0,
                    "retrieval_reason": "normalized_name",
                    "quality_flags": [],
                }
            ],
            "relations": [
                {
                    "from_uid": "cn_1",
                    "from_name": "Condicionamiento clásico",
                    "relation": "CONTRASTS_WITH",
                    "to_uid": "cn_2",
                    "to_name": "Condicionamiento operante",
                    "confidence": 0.95,
                    "evidence_episode_id": "ep_1",
                }
            ],
            "claims": [
                {
                    "uid": "cl_1",
                    "text": "El condicionamiento clásico asocia un estimulo condicionado con uno incondicionado.",
                    "confidence": 0.91,
                }
            ],
            "episodes": [
                {
                    "uid": "ep_1",
                    "text": "Pavlov observo que un estimulo neutro podia anticipar alimento tras repeticiones.",
                    "status": "processed",
                }
            ],
            "warnings": [],
            "debug": {"candidate_count": 1, "selected_concept_uids": ["cn_1"], "selection_reasons": ["cn_1"]},
        }

    async def get_tutor_context(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("episode_id") == "ep_sin_evidencia":
            return {
                "resolved_reference": {
                    "input_type": "episode_id",
                    "input_value": "ep_sin_evidencia",
                    "resolved_concept_uid": None,
                    "resolved_concept_name": None,
                    "resolved_episode_id": "ep_sin_evidencia",
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
            }
        return {
            "resolved_reference": {
                "input_type": "query" if kwargs.get("query") else "episode_id",
                "input_value": kwargs.get("query") or kwargs.get("episode_id"),
                "resolved_concept_uid": "cn_1",
                "resolved_concept_name": "Condicionamiento clásico",
                "resolved_episode_id": "ep_1",
                "resolved_job_id": kwargs.get("job_id"),
                "resolution_reason": "normalized_name",
            },
            "status": "ok",
            "concepts": [
                {
                    "uid": "cn_1",
                    "canonical_name": "Condicionamiento clásico",
                    "domain": "Psicología",
                    "description": "Aprendizaje asociativo donde un estimulo adquiere valor predictivo.",
                    "aliases": [],
                }
            ],
            "claims": [
                {
                    "uid": "cl_1",
                    "text": "El condicionamiento clásico asocia un estimulo condicionado con uno incondicionado.",
                    "confidence": 0.91,
                    "evidence_episode_ids": ["ep_1"],
                }
            ],
            "relations": [
                {
                    "uid": "cn_1|CONTRASTS_WITH|cn_2|ep_1",
                    "from_uid": "cn_1",
                    "from_name": "Condicionamiento clásico",
                    "relation": "CONTRASTS_WITH",
                    "to_uid": "cn_2",
                    "to_name": "Condicionamiento operante",
                    "confidence": 0.95,
                    "evidence_episode_ids": ["ep_1"],
                }
            ],
            "source_fragments": [
                {
                    "episode_id": "ep_1",
                    "text": "Pavlov observo que un estimulo neutro podia anticipar alimento tras repeticiones.",
                    "status": "processed",
                    "source_type": "manual_input",
                    "tags": ["Psicología"],
                    "language": "es",
                }
            ],
            "evidence": [
                {"subject_type": "claim", "subject_uid": "cl_1", "episode_id": "ep_1"},
                {"subject_type": "relation", "subject_uid": "cn_1|CONTRASTS_WITH|cn_2|ep_1", "episode_id": "ep_1"},
            ],
            "warnings": [],
            "failure_reason": None,
        }

    async def create_concept(self, **_: Any) -> dict[str, Any]:
        return {"concept": {"uid": "cn_strict"}, "created": True}

    async def attach_concept_evidence(self, **_: Any) -> dict[str, Any]:
        return {"status": "attached", "concept_uid": "cn_strict", "episode_id": "ep_1", "linked_claim_count": 1}

    async def upsert_concept(self, **_: Any) -> dict[str, Any]:
        return {"concept": {"uid": "cn_new"}, "created": True}

    async def link_concepts(self, **kwargs: Any) -> dict[str, Any]:
        self.last_link_call = kwargs
        return {"status": "linked"}

    async def get_neighborhood(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "concept": {
                "uid": kwargs["concept"],
                "type": "Concept",
                "name": "Memoria episódica" if kwargs["concept"] == "cn_mem_1" else "Condicionamiento clásico",
                "domain": "Psicología",
                "description": "Descripcion de apoyo",
            },
            "nodes": [],
            "relations": [],
            "claims": [
                {
                    "uid": "cl_nb_1",
                    "text": "La memoria episódica preserva experiencias autobiográficas situadas en tiempo y lugar.",
                    "confidence": 0.9,
                }
            ],
            "episodes": [
                {
                    "uid": "ep_nb_1",
                    "text": "Ejemplo recuperado desde la vecindad.",
                    "status": "processed",
                }
            ],
        }

    async def get_pedagogical_context(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "user_id": kwargs["user_id"],
            "status": "ok",
            "concepts": [],
            "domains": [],
            "recent_evaluations": [],
            "warnings": [],
        }

    async def update_pedagogical_context(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "user_id": kwargs["user_id"],
            "status": "ok",
            "context": {
                "user_id": kwargs["user_id"],
                "status": "ok",
                "concepts": [],
                "domains": [],
                "recent_evaluations": [],
                "warnings": [],
            },
            "session_view": {
                "user_id": kwargs["user_id"],
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

    async def get_pedagogical_session_view(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "user_id": kwargs["user_id"],
            "status": "ok",
            "summary": "Priorizar conceptos debiles.",
            "weak_concepts": [],
            "detected_gaps": [],
            "suggested_questions": [],
            "effective_depth_used": 3,
            "domain_focus": [],
            "recalculation_traces": [],
            "warnings": [],
        }

    async def get_sr_state(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "state": {
                "user_id": kwargs["user_id"],
                "concept_uid": kwargs["concept_uid"],
                "dimension": kwargs["dimension"],
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

    async def get_due_sr_items(self, **kwargs: Any) -> dict[str, Any]:
        return {"user_id": kwargs["user_id"], "items": []}

    async def update_sr_from_block_result(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "state": {
                "user_id": kwargs["user_id"],
                "concept_uid": kwargs["concept_uid"],
                "dimension": kwargs["dimension"],
                "repetitions": 1,
                "ease_factor": 2.5,
                "interval_days": 1,
                "last_reviewed_at": "2026-06-06T10:00:00+00:00",
                "next_review_at": "2026-06-07T10:00:00+00:00",
                "propagation_relief_count": 0,
                "requires_direct_validation": False,
                "updated_at": "2026-06-06T10:00:00+00:00",
            },
            "sr_feedback": {
                "concept_uid": kwargs["concept_uid"],
                "dimension": kwargs["dimension"],
                "calculated_quality_q": 5,
                "rationale": "Respuesta correcta directa sin apoyo",
            },
            "ef_bonus_applied": False,
        }

    async def apply_prereq_relief(self, **_: Any) -> dict[str, Any]:
        return {"updated_states": []}

    async def get_sr_stats(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "user_id": kwargs["user_id"],
            "stats": {"total_items": 1, "due_items": 0, "forced_review_items": 0, "average_ease_factor": 2.5},
        }

    async def start_adaptive_session(self, **kwargs: Any) -> dict[str, Any]:
        tutor_context = await self.get_tutor_context(
            query=kwargs.get("query"),
            episode_id=kwargs.get("episode_id"),
            job_id=kwargs.get("job_id"),
        )
        block = {
            "block_id": "blk_1",
            "plan": {
                "block_id": "blk_1",
                "block_goal": "reinforce_weak",
                "target_concept_uid": "cn_1",
                "target_concept_name": "Condicionamiento clásico",
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
        }
        return {
            "session": {
                "session_id": "ads_1",
                "user_id": kwargs["user_id"],
                "resolved_reference": tutor_context["resolved_reference"],
                "domain_hint": kwargs.get("domain_hint"),
                "language": kwargs.get("language", "es"),
                "constraints": {
                    "max_items_per_block": 3,
                    "max_blocks": 4,
                    "allowed_question_types": ["multiple_choice_single", "multiple_choice_multi", "true_false", "cloze", "open"],
                    "preferred_max_difficulty": None,
                    "allow_scaffolding": True,
                },
                "tutor_context": tutor_context,
                "current_block": block,
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
            "current_block": block,
            "planner_explanation": "priorizar reconocimiento",
            "grounding_status": "ok",
        }

    async def submit_adaptive_block(self, **_: Any) -> dict[str, Any]:
        return {
            "session": {
                "session_id": "ads_1",
                "user_id": "user-1",
                "resolved_reference": {
                    "input_type": "query",
                    "input_value": "condicionamiento clásico",
                    "resolved_concept_uid": "cn_1",
                    "resolved_concept_name": "Condicionamiento clásico",
                    "resolved_episode_id": "ep_1",
                    "resolved_job_id": None,
                    "resolution_reason": "normalized_name",
                },
                "domain_hint": None,
                "language": "es",
                "constraints": {
                    "max_items_per_block": 3,
                    "max_blocks": 4,
                    "allowed_question_types": ["multiple_choice_single", "multiple_choice_multi", "true_false", "cloze", "open"],
                    "preferred_max_difficulty": None,
                    "allow_scaffolding": True,
                },
                "tutor_context": await self.get_tutor_context(query="condicionamiento clásico"),
                "current_block": None,
                "block_history": [],
                "summary": {
                    "total_blocks": 4,
                    "completed_blocks": 1,
                    "latest_block_verdict": "correct",
                    "session_closed": True,
                    "closure_reason": "coverage_sufficient",
                },
                "status": "closed",
                "opened_at": "2026-06-06T10:00:00+00:00",
                "updated_at": "2026-06-06T10:05:00+00:00",
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
            "updated_context": await self.get_pedagogical_context(user_id="user-1"),
            "next_action": "shift_dimension_within_concept_then_reprioritize",
            "next_block": None,
            "session_closed": True,
        }

    async def get_adaptive_session(self, **_: Any) -> dict[str, Any]:
        return (await self.start_adaptive_session(user_id="user-1", query="condicionamiento clásico"))["session"]

    async def check_ready(self) -> tuple[bool, dict[str, Any]]:
        return self.ready

    async def add_knowledge_fragment(self, **_: Any) -> dict[str, Any]:
        return {"episode_id": "ep_1", "job_id": "job_1", "status": "queued"}

    async def close(self) -> None:
        return None


def _knowledge_settings() -> Settings:
    return Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        MCP_BEARER_TOKEN="knowledge-token",
        KG_API_KEY="internal-api-key",
        KG_API_BASE_URL="http://backend.test",
    )


def _agent_settings() -> Settings:
    return Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        AGENT_MCP_BEARER_TOKEN="agent-token",
        KNOWLEDGE_MCP_BASE_URL="http://knowledge.test",
        KNOWLEDGE_MCP_BEARER_TOKEN="knowledge-token",
        MCP_BEARER_TOKEN="knowledge-token",
        KG_API_KEY="internal-api-key",
        KG_API_BASE_URL="http://backend.test",
    )


def test_agent_upstream_client_default_timeout_covers_ingestion_window():
    settings = _agent_settings()
    client = AgentMCPUpstreamClient(settings)
    try:
        assert client._client.timeout.read >= settings.mcp_ingestion_timeout_seconds
        assert client._client.timeout.connect >= 20.0
    finally:
        import asyncio

        asyncio.run(client.close())


@pytest.fixture
def knowledge_backend() -> FakeKnowledgeBackendClient:
    return FakeKnowledgeBackendClient()


@pytest.fixture
async def upstream_client(knowledge_backend: FakeKnowledgeBackendClient) -> AsyncGenerator[AgentMCPUpstreamClient]:
    knowledge_app = create_knowledge_mcp_app(settings=_knowledge_settings(), backend_client=knowledge_backend)
    transport = httpx.ASGITransport(app=knowledge_app)
    async with knowledge_app.router.lifespan_context(knowledge_app):
        client = AgentMCPUpstreamClient(_agent_settings(), transport=transport)
        try:
            yield client
        finally:
            await client.close()


@pytest.mark.anyio
async def test_agent_mcp_http_requires_bearer_and_reports_ready(upstream_client: AgentMCPUpstreamClient):
    app = create_agent_app(settings=_agent_settings(), upstream_client=upstream_client)
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://agent.test") as client:
            assert (await client.get("/health/live")).status_code == 200
            assert (await client.get("/health/ready")).status_code == 200
            assert (await client.get("/mcp")).status_code == 401
            assert (await client.post("/mcp", headers={"Authorization": "Bearer nope"}, json={})).status_code == 401


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client_session(
    upstream_client: AgentMCPUpstreamClient,
) -> AsyncGenerator[Any]:
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session

    server = create_agent_mcp_server(settings=_agent_settings(), upstream_client=upstream_client)
    async with create_connected_server_and_client_session(server, raise_exceptions=True) as session:
        yield session


@pytest.mark.anyio
async def test_agent_mcp_exposes_exactly_twenty_three_tools(client_session):
    tools = await client_session.list_tools()
    assert sorted(tool.name for tool in tools.tools) == [
        "evaluate_answer",
        "explain_topic",
        "get_adaptive_session",
        "generate_quiz",
        "kg_add_knowledge_fragment",
        "kg_attach_concept_evidence",
        "kg_create_concept",
        "kg_get_learning_context",
        "kg_get_neighborhood",
        "kg_get_pedagogical_context",
        "kg_get_pedagogical_session_view",
        "kg_get_tutor_context",
        "kg_link_concepts",
        "kg_search_candidates",
        "kg_sr_apply_relief",
        "kg_sr_get_due_items",
        "kg_sr_get_state",
        "kg_sr_get_stats",
        "kg_sr_update_from_block",
        "kg_update_pedagogical_context",
        "kg_upsert_concept",
        "start_adaptive_session",
        "submit_adaptive_block",
    ]

    tool_map = {tool.name: tool for tool in tools.tools}
    assert tool_map["generate_quiz"].inputSchema["properties"]["question_count"]["default"] == 5
    assert tool_map["explain_topic"].inputSchema["properties"]["audience"]["default"] == "intermediate"
    assert tool_map["kg_get_tutor_context"].inputSchema["properties"]["depth"]["default"] == 1
    assert tool_map["kg_get_pedagogical_context"].inputSchema["properties"]["user_id"]["type"] == "string"
    assert tool_map["kg_sr_get_state"].inputSchema["properties"]["concept_uid"]["type"] == "string"
    assert tool_map["start_adaptive_session"].inputSchema["properties"]["language"]["default"] == "es"
    assert "from" in tool_map["kg_link_concepts"].inputSchema["properties"]
    assert tool_map["kg_upsert_concept"].inputSchema["properties"]["uid"]["type"] == "string"


@pytest.mark.anyio
async def test_agent_high_level_tools_cover_ok_sparse_and_no_match(client_session):
    explain_ok = await client_session.call_tool(
        "explain_topic",
        {"query": "condicionamiento clásico", "domain_hint": "Psicología"},
    )
    assert explain_ok.isError in {False, None}
    assert explain_ok.structuredContent["status"] == "ok"
    assert explain_ok.structuredContent["source_concept_uids"] == ["cn_1"]
    assert explain_ok.structuredContent["debug"]["generation_mode"] in {"stub", "structured_llm"}

    explain_sparse = await client_session.call_tool(
        "explain_topic",
        {"query": "memoria", "domain_hint": "Psicología"},
    )
    assert explain_sparse.isError in {False, None}
    assert explain_sparse.structuredContent["status"] == "sparse"
    assert explain_sparse.structuredContent["debug"]["used_neighborhood"] is True
    assert "no_source_episodes" in explain_sparse.structuredContent["warnings"]

    quiz = await client_session.call_tool(
        "generate_quiz",
        {"query": "condicionamiento clásico", "question_count": 3, "question_type": "mixed"},
    )
    assert quiz.isError in {False, None}
    assert quiz.structuredContent["status"] == "ok"
    assert len(quiz.structuredContent["questions"]) == 3
    assert len(quiz.structuredContent["answer_key"]) == 3

    no_match = await client_session.call_tool(
        "generate_quiz",
        {"query": "tema inexistente", "domain_hint": "Psicología"},
    )
    assert no_match.isError in {False, None}
    assert no_match.structuredContent["status"] == "no_match"
    assert no_match.structuredContent["questions"] == []


@pytest.mark.anyio
async def test_agent_evaluate_answer_returns_correct_partial_and_incorrect(client_session):
    correct = await client_session.call_tool(
        "evaluate_answer",
        {
            "query": "memoria",
            "question": "¿Qué hace la memoria episódica?",
            "learner_answer": "La memoria episódica recupera experiencias personales con contexto temporal.",
            "expected_answer": "La memoria episódica recupera experiencias personales con contexto temporal.",
        },
    )
    assert correct.structuredContent["verdict"] == "correct"

    partial = await client_session.call_tool(
        "evaluate_answer",
        {
            "query": "memoria",
            "question": "¿Qué hace la memoria episódica?",
            "learner_answer": "Recupera experiencias personales.",
            "expected_answer": "La memoria episódica recupera experiencias personales con contexto temporal.",
        },
    )
    assert partial.structuredContent["verdict"] == "partial"

    incorrect = await client_session.call_tool(
        "evaluate_answer",
        {
            "query": "memoria",
            "question": "¿Qué hace la memoria episódica?",
            "learner_answer": "Controla el sistema digestivo.",
            "expected_answer": "La memoria episódica recupera experiencias personales con contexto temporal.",
        },
    )
    assert incorrect.structuredContent["verdict"] == "incorrect"


@pytest.mark.anyio
async def test_agent_passthrough_tools_preserve_shape_and_from_alias(
    client_session,
    knowledge_backend: FakeKnowledgeBackendClient,
):
    link = await client_session.call_tool(
        "kg_link_concepts",
        {"from": "cn_1", "relation": "RELATED_TO", "to": "cn_2"},
    )
    assert link.isError in {False, None}
    assert link.structuredContent == {"status": "linked"}
    assert knowledge_backend.last_link_call == {
        "from_ref": "cn_1",
        "relation": "RELATED_TO",
        "to_ref": "cn_2",
        "evidence_episode_id": None,
    }

    learning_context = await client_session.call_tool(
        "kg_get_learning_context",
        {"query": "memoria", "domain_hint": "Psicología"},
    )
    assert learning_context.structuredContent["status"] == "sparse"

    tutor_context = await client_session.call_tool(
        "kg_get_tutor_context",
        {"query": "condicionamiento clásico"},
    )
    assert tutor_context.structuredContent["status"] == "ok"
    assert tutor_context.structuredContent["resolved_reference"]["resolved_concept_uid"] == "cn_1"

    pedagogical_context = await client_session.call_tool(
        "kg_get_pedagogical_context",
        {"user_id": "user-1"},
    )
    assert pedagogical_context.structuredContent["user_id"] == "user-1"

    sr_state = await client_session.call_tool(
        "kg_sr_get_state",
        {"user_id": "user-1", "concept_uid": "cn_1", "dimension": "recognition"},
    )
    assert sr_state.structuredContent["state"]["concept_uid"] == "cn_1"

    pedagogical_update = await client_session.call_tool(
        "kg_update_pedagogical_context",
        {"user_id": "user-1", "evaluations": [{"concept_uid": "cn_1", "score_0_to_100": 68}]},
    )
    assert pedagogical_update.structuredContent["session_view"]["effective_depth_used"] == 3

    adaptive_start = await client_session.call_tool(
        "start_adaptive_session",
        {"user_id": "user-1", "query": "condicionamiento clásico"},
    )
    assert adaptive_start.structuredContent["grounding_status"] == "ok"

    adaptive_submit = await client_session.call_tool(
        "submit_adaptive_block",
        {"session_id": "ads_1", "block_id": "blk_1", "submissions": [{"item_id": "blk_1_item_1", "selected_choices": [0]}]},
    )
    assert adaptive_submit.structuredContent["session_closed"] is True


@pytest.mark.anyio
async def test_agent_streamable_http_client_works_with_documented_url(
    upstream_client: AgentMCPUpstreamClient,
):
    pytest.importorskip("mcp")
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    app = create_agent_app(settings=_agent_settings(), upstream_client=upstream_client)
    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://agent.test",
            headers={"Authorization": "Bearer agent-token"},
        ) as http_client:
            async with streamable_http_client(
                "http://agent.test/mcp",
                http_client=http_client,
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await session.list_tools()

    assert len(tools.tools) == 18
