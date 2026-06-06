from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Literal

import httpx

from app.config import Settings
from app.schemas import (
    AgentToolDebug,
    EvaluateAnswerResponse,
    ExplainTopicResponse,
    GenerateQuizResponse,
    LearningContextResponse,
    NeighborhoodResponse,
    QuizAnswerKeyItem,
    QuizQuestion,
)
from app.utils import dedupe_preserve_order, normalize_text


class AgentContentGenerator(ABC):
    mode: Literal["structured_llm", "stub"]

    @abstractmethod
    async def explain_topic(
        self,
        *,
        query: str,
        domain_hint: str | None,
        audience: str,
        focus: str | None,
        include_examples: bool,
        context: LearningContextResponse,
        neighborhood: NeighborhoodResponse | None,
    ) -> ExplainTopicResponse:
        raise NotImplementedError

    @abstractmethod
    async def generate_quiz(
        self,
        *,
        query: str,
        domain_hint: str | None,
        difficulty: str,
        question_count: int,
        question_type: str,
        context: LearningContextResponse,
        neighborhood: NeighborhoodResponse | None,
    ) -> GenerateQuizResponse:
        raise NotImplementedError

    @abstractmethod
    async def evaluate_answer(
        self,
        *,
        query: str,
        question: str,
        learner_answer: str,
        domain_hint: str | None,
        expected_answer: str | None,
        context: LearningContextResponse,
        neighborhood: NeighborhoodResponse | None,
    ) -> EvaluateAnswerResponse:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class StubAgentContentGenerator(AgentContentGenerator):
    mode: Literal["structured_llm", "stub"] = "stub"

    async def explain_topic(
        self,
        *,
        query: str,
        domain_hint: str | None,
        audience: str,
        focus: str | None,
        include_examples: bool,
        context: LearningContextResponse,
        neighborhood: NeighborhoodResponse | None,
    ) -> ExplainTopicResponse:
        evidence = _collect_evidence(context, neighborhood)
        concepts = [item.canonical_name for item in context.primary_concepts]
        examples = list(context.episodes[:3])
        example_texts = [episode.text for episode in examples] if include_examples else []
        explanation_parts = [
            f"### {query}",
            f"Audiencia objetivo: `{audience}`.",
        ]
        if concepts:
            explanation_parts.append(f"Conceptos base: {', '.join(concepts)}.")
        if focus:
            explanation_parts.append(f"Foco solicitado: {focus}.")
        if evidence:
            explanation_parts.append("Puntos respaldados por el knowledge graph:")
            explanation_parts.extend(f"- {item}" for item in evidence[:4])
        if context.status == "sparse":
            explanation_parts.append("Advertencia: el contexto recuperado es parcial y puede requerir curación.")
        return ExplainTopicResponse(
            query=query,
            domain_hint=domain_hint,
            status=context.status,
            explanation_markdown="\n".join(explanation_parts),
            key_points=evidence[:5],
            examples=example_texts,
            source_concept_uids=[item.uid for item in context.primary_concepts],
            warnings=list(context.warnings),
            debug=_build_debug(context, neighborhood is not None, self.mode),
        )

    async def generate_quiz(
        self,
        *,
        query: str,
        domain_hint: str | None,
        difficulty: str,
        question_count: int,
        question_type: str,
        context: LearningContextResponse,
        neighborhood: NeighborhoodResponse | None,
    ) -> GenerateQuizResponse:
        evidence = _collect_evidence(context, neighborhood)
        questions: list[QuizQuestion] = []
        answer_key: list[QuizAnswerKeyItem] = []
        if not evidence:
            evidence = [f"No hay evidencia suficiente para evaluar {query} con confianza."]

        for index in range(question_count):
            fact = evidence[index % len(evidence)]
            question_kind = _resolve_question_type(question_type, index)
            question_id = f"q_{index + 1}"
            if question_kind == "multiple_choice":
                distractors = _build_distractors(fact, evidence)
                choices = dedupe_preserve_order([fact, *distractors])[:4]
                questions.append(
                    QuizQuestion(
                        id=question_id,
                        type="multiple_choice",
                        prompt=f"¿Cuál afirmación está respaldada sobre {query}?",
                        choices=choices,
                    )
                )
                rationale = f"La opción correcta resume evidencia recuperada para {query}."
                answer_key.append(
                    QuizAnswerKeyItem(question_id=question_id, correct_answer=fact, rationale=rationale)
                )
            else:
                prompt = f"Explica con tus palabras este punto sobre {query}: {fact}"
                questions.append(QuizQuestion(id=question_id, type="open", prompt=prompt, choices=[]))
                answer_key.append(
                    QuizAnswerKeyItem(
                        question_id=question_id,
                        correct_answer=fact,
                        rationale="La respuesta debe cubrir la idea central recuperada del grafo.",
                    )
                )

        return GenerateQuizResponse(
            query=query,
            domain_hint=domain_hint,
            status=context.status,
            questions=questions,
            answer_key=answer_key,
            coverage_summary=(
                f"Quiz basado en {len(context.primary_concepts)} conceptos, "
                f"{len(context.claims)} claims y dificultad `{difficulty}`."
            ),
            warnings=list(context.warnings),
            debug=_build_debug(context, neighborhood is not None, self.mode),
        )

    async def evaluate_answer(
        self,
        *,
        query: str,
        question: str,
        learner_answer: str,
        domain_hint: str | None,
        expected_answer: str | None,
        context: LearningContextResponse,
        neighborhood: NeighborhoodResponse | None,
    ) -> EvaluateAnswerResponse:
        target_points = [expected_answer] if expected_answer else _collect_evidence(context, neighborhood)[:3]
        target_points = [item for item in target_points if item]
        if not target_points:
            return EvaluateAnswerResponse(
                query=query,
                status=context.status,
                verdict="unsupported",
                score_0_to_1=0.0,
                feedback_markdown="No hay evidencia suficiente para evaluar esta respuesta con confianza.",
                matched_points=[],
                missing_points=[],
                warnings=list(context.warnings),
                debug=_build_debug(context, neighborhood is not None, "skipped"),
            )

        point_scores = {point: _point_match_score(learner_answer, point) for point in target_points}
        matched = [point for point, score in point_scores.items() if score >= 0.5]
        missing = [point for point in target_points if point not in matched]
        if len(target_points) == 1:
            score = next(iter(point_scores.values()))
        else:
            score = sum(point_scores.values()) / len(target_points)
        if score >= 0.75:
            verdict = "correct"
        elif score >= 0.34:
            verdict = "partial"
        else:
            verdict = "incorrect"

        feedback = [
            f"### Evaluacion de la respuesta sobre {query}",
            f"Pregunta: {question}",
            f"Veredicto: `{verdict}`.",
        ]
        if matched:
            feedback.append("Puntos cubiertos:")
            feedback.extend(f"- {point}" for point in matched)
        if missing:
            feedback.append("Puntos faltantes:")
            feedback.extend(f"- {point}" for point in missing)

        return EvaluateAnswerResponse(
            query=query,
            status=context.status,
            verdict=verdict,
            score_0_to_1=round(score, 2),
            feedback_markdown="\n".join(feedback),
            matched_points=matched,
            missing_points=missing,
            warnings=list(context.warnings),
            debug=_build_debug(context, neighborhood is not None, self.mode),
        )


class OpenAICompatibleAgentContentGenerator(AgentContentGenerator):
    mode: Literal["structured_llm", "stub"] = "structured_llm"

    def __init__(self, settings: Settings) -> None:
        if not settings.resolved_agent_openai_api_key:
            raise ValueError("agent generation requires AGENT_OPENAI_API_KEY or OPENAI_API_KEY")
        self._settings = settings
        self._fallback = StubAgentContentGenerator()
        self._client = httpx.AsyncClient(
            base_url=settings.resolved_agent_openai_base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {settings.resolved_agent_openai_api_key}"},
            timeout=60.0,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def explain_topic(
        self,
        *,
        query: str,
        domain_hint: str | None,
        audience: str,
        focus: str | None,
        include_examples: bool,
        context: LearningContextResponse,
        neighborhood: NeighborhoodResponse | None,
    ) -> ExplainTopicResponse:
        if context.status == "no_match":
            return await self._fallback.explain_topic(
                query=query,
                domain_hint=domain_hint,
                audience=audience,
                focus=focus,
                include_examples=include_examples,
                context=context,
                neighborhood=neighborhood,
            )
        prompt = {
            "task": "explain_topic",
            "query": query,
            "domain_hint": domain_hint,
            "audience": audience,
            "focus": focus,
            "include_examples": include_examples,
            "context": context.model_dump(),
            "support_neighborhood": neighborhood.model_dump() if neighborhood else None,
        }
        response = await self._generate_json(prompt, ExplainTopicResponse)
        return response.model_copy(
            update={
                "debug": _build_debug(context, neighborhood is not None, self.mode),
                "warnings": dedupe_preserve_order([*context.warnings, *response.warnings]),
                "source_concept_uids": response.source_concept_uids or [item.uid for item in context.primary_concepts],
            }
        )

    async def generate_quiz(
        self,
        *,
        query: str,
        domain_hint: str | None,
        difficulty: str,
        question_count: int,
        question_type: str,
        context: LearningContextResponse,
        neighborhood: NeighborhoodResponse | None,
    ) -> GenerateQuizResponse:
        if context.status == "no_match":
            return await self._fallback.generate_quiz(
                query=query,
                domain_hint=domain_hint,
                difficulty=difficulty,
                question_count=question_count,
                question_type=question_type,
                context=context,
                neighborhood=neighborhood,
            )
        prompt = {
            "task": "generate_quiz",
            "query": query,
            "domain_hint": domain_hint,
            "difficulty": difficulty,
            "question_count": question_count,
            "question_type": question_type,
            "context": context.model_dump(),
            "support_neighborhood": neighborhood.model_dump() if neighborhood else None,
        }
        response = await self._generate_json(prompt, GenerateQuizResponse)
        return response.model_copy(
            update={
                "debug": _build_debug(context, neighborhood is not None, self.mode),
                "warnings": dedupe_preserve_order([*context.warnings, *response.warnings]),
            }
        )

    async def evaluate_answer(
        self,
        *,
        query: str,
        question: str,
        learner_answer: str,
        domain_hint: str | None,
        expected_answer: str | None,
        context: LearningContextResponse,
        neighborhood: NeighborhoodResponse | None,
    ) -> EvaluateAnswerResponse:
        if context.status == "no_match":
            return await self._fallback.evaluate_answer(
                query=query,
                question=question,
                learner_answer=learner_answer,
                domain_hint=domain_hint,
                expected_answer=expected_answer,
                context=context,
                neighborhood=neighborhood,
            )
        prompt = {
            "task": "evaluate_answer",
            "query": query,
            "question": question,
            "learner_answer": learner_answer,
            "domain_hint": domain_hint,
            "expected_answer": expected_answer,
            "context": context.model_dump(),
            "support_neighborhood": neighborhood.model_dump() if neighborhood else None,
        }
        response = await self._generate_json(prompt, EvaluateAnswerResponse)
        return response.model_copy(
            update={
                "debug": _build_debug(context, neighborhood is not None, self.mode),
                "warnings": dedupe_preserve_order([*context.warnings, *response.warnings]),
            }
        )

    async def _generate_json(self, payload: dict[str, Any], schema: type[Any]):
        system_prompt = (
            "You are a grounded teaching assistant that can only use the retrieved evidence provided by the caller. "
            "Do not invent facts. If the context is sparse, preserve uncertainty in warnings and phrasing. "
            "Use external knowledge only to improve pedagogy and formatting, never to add unsupported factual claims. "
            "Return only JSON that matches the requested schema."
        )
        try:
            response = await self._client.post(
                "chat/completions",
                json={
                    "model": self._settings.resolved_agent_openai_chat_model,
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
                    ],
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return schema.model_validate(json.loads(content))
        except (httpx.HTTPError, KeyError, ValueError, json.JSONDecodeError):
            task = payload.get("task")
            context = LearningContextResponse.model_validate(payload["context"])
            neighborhood_data = payload.get("support_neighborhood")
            neighborhood = NeighborhoodResponse.model_validate(neighborhood_data) if neighborhood_data else None
            if task == "explain_topic":
                return await self._fallback.explain_topic(
                    query=str(payload["query"]),
                    domain_hint=payload.get("domain_hint"),
                    audience=str(payload["audience"]),
                    focus=payload.get("focus"),
                    include_examples=bool(payload.get("include_examples", True)),
                    context=context,
                    neighborhood=neighborhood,
                )
            if task == "generate_quiz":
                return await self._fallback.generate_quiz(
                    query=str(payload["query"]),
                    domain_hint=payload.get("domain_hint"),
                    difficulty=str(payload["difficulty"]),
                    question_count=int(payload["question_count"]),
                    question_type=str(payload["question_type"]),
                    context=context,
                    neighborhood=neighborhood,
                )
            return await self._fallback.evaluate_answer(
                query=str(payload["query"]),
                question=str(payload["question"]),
                learner_answer=str(payload["learner_answer"]),
                domain_hint=payload.get("domain_hint"),
                expected_answer=payload.get("expected_answer"),
                context=context,
                neighborhood=neighborhood,
            )


def build_agent_content_generator(settings: Settings) -> AgentContentGenerator:
    if settings.resolved_agent_openai_api_key:
        return OpenAICompatibleAgentContentGenerator(settings)
    return StubAgentContentGenerator()


def _collect_evidence(
    context: LearningContextResponse,
    neighborhood: NeighborhoodResponse | None,
) -> list[str]:
    evidence: list[str] = []
    for concept in context.primary_concepts:
        if concept.description:
            evidence.append(f"{concept.canonical_name}: {concept.description}")
        else:
            evidence.append(concept.canonical_name)
    for claim in context.claims:
        if claim.text:
            evidence.append(claim.text)
    for relation in context.relations:
        evidence.append(f"{relation.from_name} {relation.relation} {relation.to_name}")
    if neighborhood:
        for claim in neighborhood.claims:
            text = str(claim.get("text") or "").strip()
            if text:
                evidence.append(text)
    return dedupe_preserve_order([item for item in evidence if item.strip()])


def _build_debug(
    context: LearningContextResponse,
    used_neighborhood: bool,
    generation_mode: Literal["structured_llm", "stub", "skipped"],
) -> AgentToolDebug:
    return AgentToolDebug(
        retrieval_status=context.status,
        used_neighborhood=used_neighborhood,
        generation_mode=generation_mode,
        source_concept_uids=[item.uid for item in context.primary_concepts],
        source_claim_uids=[item.uid for item in context.claims],
    )


def _resolve_question_type(question_type: str, index: int) -> Literal["multiple_choice", "open"]:
    if question_type == "multiple_choice":
        return "multiple_choice"
    if question_type == "open":
        return "open"
    return "multiple_choice" if index % 2 == 0 else "open"


def _build_distractors(correct_fact: str, evidence: list[str]) -> list[str]:
    distractors = [item for item in evidence if item != correct_fact]
    distractors.extend(
        [
            "No hay evidencia suficiente para sostener esta afirmacion.",
            "El knowledge graph no recupero una relacion verificable para este punto.",
            "La fuente recuperada contradice esta afirmacion.",
        ]
    )
    return distractors


def _matches_point(answer: str, point: str) -> bool:
    return _point_match_score(answer, point) >= 0.5


def _point_match_score(answer: str, point: str) -> float:
    normalized_answer = normalize_text(answer)
    normalized_point = normalize_text(point)
    if normalized_point and normalized_point in normalized_answer:
        return 1.0
    answer_tokens = {token for token in re.findall(r"[a-z0-9]+", normalized_answer) if len(token) > 3}
    point_tokens = {token for token in re.findall(r"[a-z0-9]+", normalized_point) if len(token) > 3}
    if not point_tokens:
        return 0.0
    overlap = len(answer_tokens.intersection(point_tokens))
    return overlap / len(point_tokens)
