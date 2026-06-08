from __future__ import annotations

from typing import Any

from app.agent_content import AgentContentGenerator
from app.agent_mcp_client import AgentMCPUpstreamClient
from app.schemas import (
    EvaluateAnswerResponse,
    ExplainTopicResponse,
    GenerateQuizResponse,
    TutorContextResponse,
)


class AgentProxyService:
    def __init__(
        self,
        *,
        upstream_client: AgentMCPUpstreamClient,
        content_generator: AgentContentGenerator,
    ) -> None:
        self._upstream = upstream_client
        self._content_generator = content_generator

    async def explain_topic(
        self,
        *,
        query: str,
        domain_hint: str | None = None,
        audience: str = "intermediate",
        focus: str | None = None,
        include_examples: bool = True,
    ) -> ExplainTopicResponse:
        context = await self._retrieve_grounding(query=query)
        if context.status != "ok":
            return ExplainTopicResponse(
                query=query,
                domain_hint=domain_hint,
                status="blocked",
                explanation_markdown=(
                    "No se encontró evidencia pedagógica aprobada suficiente para explicar este tema sin inventar contenido."
                ),
                key_points=[],
                examples=[],
                source_concept_uids=[],
                warnings=list(context.warnings),
                debug={"retrieval_status": "no_match", "used_neighborhood": False, "generation_mode": "skipped", "source_concept_uids": [], "source_claim_uids": []},
                failure_reason=context.failure_reason or "insufficient_pedagogical_evidence",
            )
        return await self._content_generator.explain_topic(
            query=query,
            domain_hint=domain_hint,
            audience=audience,
            focus=focus,
            include_examples=include_examples,
            tutor_context=context,
        )

    async def generate_quiz(
        self,
        *,
        query: str,
        domain_hint: str | None = None,
        difficulty: str = "intermediate",
        question_count: int = 5,
        question_type: str = "mixed",
    ) -> GenerateQuizResponse:
        context = await self._retrieve_grounding(query=query)
        if context.status != "ok":
            return GenerateQuizResponse(
                query=query,
                domain_hint=domain_hint,
                status="blocked",
                questions=[],
                answer_key=[],
                coverage_summary="No se pudo generar un quiz grounded porque no hubo evidencia pedagógica aprobada.",
                warnings=list(context.warnings),
                debug={"retrieval_status": "no_match", "used_neighborhood": False, "generation_mode": "skipped", "source_concept_uids": [], "source_claim_uids": []},
                failure_reason=context.failure_reason or "insufficient_pedagogical_evidence",
            )
        return await self._content_generator.generate_quiz(
            query=query,
            domain_hint=domain_hint,
            difficulty=difficulty,
            question_count=question_count,
            question_type=question_type,
            tutor_context=context,
        )

    async def evaluate_answer(
        self,
        *,
        query: str,
        question: str,
        learner_answer: str,
        domain_hint: str | None = None,
        expected_answer: str | None = None,
    ) -> EvaluateAnswerResponse:
        context = await self._retrieve_grounding(query=query)
        if context.status != "ok":
            return EvaluateAnswerResponse(
                query=query,
                status="blocked",
                verdict="unsupported",
                score_0_to_1=0.0,
                feedback_markdown=(
                    "No se puede evaluar la respuesta de forma grounded porque no hay evidencia pedagógica aprobada suficiente."
                ),
                matched_points=[],
                missing_points=[],
                warnings=list(context.warnings),
                debug={"retrieval_status": "no_match", "used_neighborhood": False, "generation_mode": "skipped", "source_concept_uids": [], "source_claim_uids": []},
                failure_reason=context.failure_reason or "insufficient_pedagogical_evidence",
            )
        return await self._content_generator.evaluate_answer(
            query=query,
            question=question,
            learner_answer=learner_answer,
            domain_hint=domain_hint,
            expected_answer=expected_answer,
            tutor_context=context,
        )

    async def passthrough(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._upstream.call_tool(tool_name, arguments)

    async def _retrieve_grounding(self, *, query: str) -> TutorContextResponse:
        return TutorContextResponse.model_validate(await self._upstream.get_tutor_context(query=query))
