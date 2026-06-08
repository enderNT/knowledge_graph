from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.schemas import (
    AdaptiveBlockAnswerKey,
    AdaptiveBlockItem,
    AgentToolDebug,
    EvaluateAnswerResponse,
    ExplainTopicResponse,
    GenerateQuizResponse,
    QuizAnswerKeyItem,
    QuizQuestion,
    TutorContextPedagogicalEvidence,
    TutorContextResponse,
)
from app.utils import dedupe_preserve_order, normalize_text


QUESTION_CONTRACT_VERSION = "pedagogical-v2"


@dataclass(frozen=True)
class EvidenceUnit:
    uid: str
    concept_uid: str
    concept_name: str
    statement: str
    supporting_quote: str
    episode_id: str
    kind: str


class PedagogicalAssessmentService:
    def select_evidence(
        self,
        *,
        tutor_context: TutorContextResponse,
        concept_uid: str | None = None,
        limit: int = 3,
    ) -> list[EvidenceUnit]:
        items: list[EvidenceUnit] = []
        for item in tutor_context.pedagogical_evidence:
            if item.status != "approved":
                continue
            if concept_uid and item.concept_uid != concept_uid:
                continue
            items.append(
                EvidenceUnit(
                    uid=item.uid,
                    concept_uid=item.concept_uid,
                    concept_name=item.concept_name,
                    statement=item.statement.strip(),
                    supporting_quote=item.supporting_quote.strip(),
                    episode_id=item.episode_id,
                    kind=item.kind,
                )
            )
        deduped: list[EvidenceUnit] = []
        seen: set[str] = set()
        for item in items:
            key = normalize_text(item.statement)
            if not key or key in seen:
                continue
            deduped.append(item)
            seen.add(key)
        return deduped[:limit]

    def blocked_reason(self, tutor_context: TutorContextResponse) -> str | None:
        return tutor_context.failure_reason

    def build_explanation(
        self,
        *,
        query: str,
        domain_hint: str | None,
        audience: str,
        focus: str | None,
        include_examples: bool,
        tutor_context: TutorContextResponse,
        debug: AgentToolDebug,
    ) -> ExplainTopicResponse:
        evidence = self.select_evidence(tutor_context=tutor_context, limit=4)
        blocked = self.blocked_reason(tutor_context)
        if blocked:
            return ExplainTopicResponse(
                query=query,
                domain_hint=domain_hint,
                status="blocked",
                explanation_markdown="No hay evidencia pedagógica aprobada suficiente para explicar este tema con seguridad.",
                key_points=[],
                examples=[],
                source_concept_uids=[item.uid for item in tutor_context.concepts],
                warnings=list(tutor_context.warnings),
                debug=debug,
                failure_reason=blocked or "pedagogical_quality_insufficient",
            )
        if not evidence:
            return ExplainTopicResponse(
                query=query,
                domain_hint=domain_hint,
                status="sparse",
                explanation_markdown="No se encontraron unidades pedagógicas seleccionables para esta explicación.",
                key_points=[],
                examples=[],
                source_concept_uids=[item.uid for item in tutor_context.concepts],
                warnings=list(tutor_context.warnings),
                debug=debug,
                failure_reason=None,
            )
        key_points = [item.statement for item in evidence]
        examples = [item.supporting_quote for item in evidence[:2]] if include_examples else []
        parts = [f"### {query}", f"Audiencia objetivo: `{audience}`."]
        if focus:
            parts.append(f"Foco: {focus}.")
        parts.append("Puntos clave:")
        parts.extend(f"- {point}" for point in key_points)
        return ExplainTopicResponse(
            query=query,
            domain_hint=domain_hint,
            status="ok",
            explanation_markdown="\n".join(parts),
            key_points=key_points,
            examples=examples,
            source_concept_uids=dedupe_preserve_order([item.concept_uid for item in evidence]),
            warnings=list(tutor_context.warnings),
            debug=debug,
            failure_reason=None,
        )

    def build_quiz(
        self,
        *,
        query: str,
        domain_hint: str | None,
        difficulty: str,
        question_count: int,
        question_type: str,
        tutor_context: TutorContextResponse,
        debug: AgentToolDebug,
    ) -> GenerateQuizResponse:
        evidence = self.select_evidence(tutor_context=tutor_context, limit=max(question_count, 3))
        blocked = self.blocked_reason(tutor_context)
        if blocked:
            return GenerateQuizResponse(
                query=query,
                domain_hint=domain_hint,
                status="blocked",
                questions=[],
                answer_key=[],
                coverage_summary="La generación se bloqueó por falta de evidencia pedagógica aprobada.",
                warnings=list(tutor_context.warnings),
                debug=debug,
                failure_reason=blocked or "pedagogical_quality_insufficient",
            )
        if not evidence:
            return GenerateQuizResponse(
                query=query,
                domain_hint=domain_hint,
                status="sparse",
                questions=[],
                answer_key=[],
                coverage_summary="No hubo unidades pedagógicas seleccionables para generar el quiz.",
                warnings=list(tutor_context.warnings),
                debug=debug,
                failure_reason=None,
            )
        questions: list[QuizQuestion] = []
        answer_key: list[QuizAnswerKeyItem] = []
        for index in range(question_count):
            unit = evidence[index % len(evidence)]
            if question_type == "open" or (question_type == "mixed" and index % 2 == 1):
                questions.append(
                    QuizQuestion(
                        id=f"q_{index + 1}",
                        type="open",
                        prompt=f"Explica con tus palabras esta idea sobre {unit.concept_name}: {unit.statement}",
                        choices=[],
                    )
                )
                answer_key.append(
                    QuizAnswerKeyItem(
                        question_id=f"q_{index + 1}",
                        correct_answer=unit.statement,
                        rationale="La respuesta debe cubrir la afirmación aprobada y su idea central.",
                        evidence_unit_ids=[unit.uid],
                        grading_rubric=self._rubric_for_statement(unit.statement),
                        validator_notes=[],
                    )
                )
                continue
            distractors = self._plausible_distractors(unit, evidence)
            if len(distractors) < 3:
                return GenerateQuizResponse(
                    query=query,
                    domain_hint=domain_hint,
                    status="blocked",
                    questions=[],
                    answer_key=[],
                    coverage_summary="La generación se bloqueó porque no se logró un set válido de distractores.",
                    warnings=[*tutor_context.warnings, "invalid_distractor_set"],
                    debug=debug,
                    failure_reason="pedagogical_quality_insufficient",
                )
            choices = [unit.statement, *distractors[:3]]
            questions.append(
                QuizQuestion(
                    id=f"q_{index + 1}",
                    type="multiple_choice",
                    prompt=f"¿Cuál afirmación describe correctamente {unit.concept_name}?",
                    choices=choices,
                )
            )
            answer_key.append(
                QuizAnswerKeyItem(
                    question_id=f"q_{index + 1}",
                    correct_answer=unit.statement,
                    rationale="La opción correcta corresponde a evidencia pedagógica aprobada.",
                    evidence_unit_ids=[unit.uid],
                    grading_rubric=self._rubric_for_statement(unit.statement),
                    validator_notes=[],
                )
            )
        return GenerateQuizResponse(
            query=query,
            domain_hint=domain_hint,
            status="ok",
            questions=questions,
            answer_key=answer_key,
            coverage_summary=f"Quiz grounded con {len(evidence)} unidades aprobadas y dificultad `{difficulty}`.",
            warnings=list(tutor_context.warnings),
            debug=debug,
            failure_reason=None,
        )

    def evaluate_answer(
        self,
        *,
        query: str,
        question: str,
        learner_answer: str,
        expected_answer: str | None,
        tutor_context: TutorContextResponse,
        debug: AgentToolDebug,
    ) -> EvaluateAnswerResponse:
        blocked = self.blocked_reason(tutor_context)
        if blocked:
            return EvaluateAnswerResponse(
                query=query,
                status="blocked",
                verdict="unsupported",
                score_0_to_1=0.0,
                feedback_markdown="No se puede justificar una evaluación grounded con la evidencia pedagógica disponible.",
                matched_points=[],
                missing_points=[],
                warnings=list(tutor_context.warnings),
                debug=debug,
                failure_reason=blocked,
            )
        evidence = self.select_evidence(tutor_context=tutor_context, limit=3)
        target = expected_answer or (evidence[0].statement if evidence else "")
        if not target:
            return EvaluateAnswerResponse(
                query=query,
                status="sparse",
                verdict="unsupported",
                score_0_to_1=0.0,
                feedback_markdown="No existe una unidad pedagógica aprobada para esta evaluación.",
                matched_points=[],
                missing_points=[],
                warnings=list(tutor_context.warnings),
                debug=debug,
                failure_reason=None,
            )
        rubric = self._rubric_for_statement(target)
        matched, missing = self._score_rubric(learner_answer, rubric)
        score = len(matched) / max(len(rubric["points"]), 1)
        verdict: Literal["correct", "partial", "incorrect", "unsupported"]
        if score >= 0.99:
            verdict = "correct"
        elif score >= 0.4:
            verdict = "partial"
        else:
            verdict = "incorrect"
        feedback = [
            f"### Evaluación de la respuesta sobre {query}",
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
            status="ok",
            verdict=verdict,
            score_0_to_1=round(score, 2),
            feedback_markdown="\n".join(feedback),
            matched_points=matched,
            missing_points=missing,
            warnings=list(tutor_context.warnings),
            debug=debug,
            failure_reason=None,
        )

    def build_adaptive_item(
        self,
        *,
        item_id: str,
        question_type: str,
        concept_uid: str,
        concept_name: str,
        difficulty: str,
        dimension: str,
        unit: EvidenceUnit,
        alternative_units: list[EvidenceUnit],
    ) -> tuple[AdaptiveBlockItem, AdaptiveBlockAnswerKey]:
        rubric = self._rubric_for_statement(unit.statement)
        if question_type == "multiple_choice_single":
            distractors = self._plausible_distractors(unit, alternative_units)
            choices = [unit.statement, *distractors[:3]]
            return (
                AdaptiveBlockItem(
                    item_id=item_id,
                    question_type="multiple_choice_single",
                    concept_uid=concept_uid,
                    target_dimension=dimension,  # type: ignore[arg-type]
                    difficulty=difficulty,  # type: ignore[arg-type]
                    prompt=f"¿Cuál afirmación describe correctamente {concept_name}?",
                    choices=choices,
                    rubric={"main_signal": "recognition"},
                    grounding_refs=[unit.uid],
                ),
                AdaptiveBlockAnswerKey(
                    item_id=item_id,
                    grading_mode="deterministic_choice",
                    expected=[unit.statement],
                    correct_choice_indexes=[0],
                    rationale="La opción correcta coincide con evidencia pedagógica aprobada.",
                    evidence_unit_ids=[unit.uid],
                    grading_rubric=rubric,
                ),
            )
        if question_type == "true_false":
            return (
                AdaptiveBlockItem(
                    item_id=item_id,
                    question_type="true_false",
                    concept_uid=concept_uid,
                    target_dimension=dimension,  # type: ignore[arg-type]
                    difficulty=difficulty,  # type: ignore[arg-type]
                    prompt=f"Verdadero o falso: {unit.statement}",
                    choices=["Verdadero", "Falso"],
                    rubric={"main_signal": "recognition"},
                    grounding_refs=[unit.uid],
                ),
                AdaptiveBlockAnswerKey(
                    item_id=item_id,
                    grading_mode="deterministic_boolean",
                    expected=[unit.statement],
                    boolean_answer=True,
                    rationale="La afirmación está aprobada como evidencia pedagógica.",
                    evidence_unit_ids=[unit.uid],
                    grading_rubric=rubric,
                ),
            )
        if question_type == "cloze":
            expected = rubric["points"][0] if rubric["points"] else unit.statement
            prompt = unit.statement.replace(expected, "_____ ", 1) if expected in unit.statement else f"Completa: {unit.statement}"
            return (
                AdaptiveBlockItem(
                    item_id=item_id,
                    question_type="cloze",
                    concept_uid=concept_uid,
                    target_dimension=dimension,  # type: ignore[arg-type]
                    difficulty=difficulty,  # type: ignore[arg-type]
                    prompt=prompt,
                    choices=[],
                    rubric={"main_signal": "recall"},
                    grounding_refs=[unit.uid],
                ),
                AdaptiveBlockAnswerKey(
                    item_id=item_id,
                    grading_mode="exact_text",
                    expected=[expected],
                    rationale="La respuesta correcta recupera el punto central omitido.",
                    evidence_unit_ids=[unit.uid],
                    grading_rubric=rubric,
                ),
            )
        if question_type == "multiple_choice_multi" and len(alternative_units) >= 2:
            secondary = alternative_units[1]
            distractors = self._plausible_distractors(unit, alternative_units)
            choices = [unit.statement, secondary.statement, *distractors[:2]]
            return (
                AdaptiveBlockItem(
                    item_id=item_id,
                    question_type="multiple_choice_multi",
                    concept_uid=concept_uid,
                    target_dimension=dimension,  # type: ignore[arg-type]
                    difficulty=difficulty,  # type: ignore[arg-type]
                    prompt=f"Selecciona todas las afirmaciones correctas sobre {concept_name}.",
                    choices=choices,
                    rubric={"main_signal": "application", "secondary_signal": "coverage"},
                    grounding_refs=[unit.uid, secondary.uid],
                ),
                AdaptiveBlockAnswerKey(
                    item_id=item_id,
                    grading_mode="partial_multi_choice",
                    expected=[unit.statement, secondary.statement],
                    correct_choice_indexes=[0, 1],
                    rationale="Las respuestas correctas coinciden con evidencia aprobada del mismo concepto.",
                    evidence_unit_ids=[unit.uid, secondary.uid],
                    grading_rubric=rubric,
                ),
            )
        return (
            AdaptiveBlockItem(
                item_id=item_id,
                question_type="open",
                concept_uid=concept_uid,
                target_dimension=dimension,  # type: ignore[arg-type]
                difficulty=difficulty,  # type: ignore[arg-type]
                prompt=f"Explica con tus palabras esta idea sobre {concept_name}: {unit.statement}",
                choices=[],
                rubric={"main_signal": dimension, "secondary_signal": "coverage"},
                grounding_refs=[unit.uid],
            ),
            AdaptiveBlockAnswerKey(
                item_id=item_id,
                grading_mode="rubric_structured",
                expected=[unit.statement],
                rationale="La respuesta debe cubrir los puntos de la rúbrica grounded.",
                evidence_unit_ids=[unit.uid],
                grading_rubric=rubric,
            ),
        )

    def grade_open_response(self, learner_answer: str, answer_key: AdaptiveBlockAnswerKey) -> tuple[float, float, float]:
        if not answer_key.grading_rubric and answer_key.expected:
            expected = normalize_text(answer_key.expected[0])
            answer = normalize_text(learner_answer)
            expected_tokens = {token for token in expected.split() if len(token) > 3}
            answer_tokens = {token for token in answer.split() if len(token) > 3}
            overlap = len(expected_tokens & answer_tokens) / max(len(expected_tokens), 1)
            return overlap, overlap, overlap
        matched, _ = self._score_rubric(learner_answer, answer_key.grading_rubric)
        total = max(len(answer_key.grading_rubric.get("points", [])), 1)
        coverage = len(matched) / total
        precision = coverage
        return coverage, precision, coverage

    def _rubric_for_statement(self, statement: str) -> dict[str, list[str] | str]:
        normalized = normalize_text(statement)
        points = [chunk.strip() for chunk in statement.replace(";", ".").split(".") if chunk.strip()]
        if not points:
            points = [statement.strip()]
        keywords = [token for token in normalized.split() if len(token) > 4][:6]
        return {"summary": statement.strip(), "points": points[:3], "keywords": keywords}

    def _score_rubric(self, learner_answer: str, rubric: dict[str, list[str] | str]) -> tuple[list[str], list[str]]:
        points = [str(item) for item in rubric.get("points", [])]
        keywords = {str(item) for item in rubric.get("keywords", [])}
        normalized_answer = normalize_text(learner_answer)
        matched: list[str] = []
        missing: list[str] = []
        answer_tokens = {token for token in normalized_answer.split() if len(token) > 3}
        for point in points:
            normalized_point = normalize_text(point)
            point_tokens = {token for token in normalized_point.split() if len(token) > 3}
            coverage = len(point_tokens & answer_tokens) / max(len(point_tokens), 1)
            if normalized_point in normalized_answer or coverage >= 0.45 or (keywords and len(keywords & answer_tokens) >= 2):
                matched.append(point)
            else:
                missing.append(point)
        return matched, missing

    def _plausible_distractors(self, unit: EvidenceUnit, evidence: list[EvidenceUnit]) -> list[str]:
        distractors: list[str] = []
        for other in evidence:
            if other.uid == unit.uid:
                continue
            if other.concept_uid == unit.concept_uid:
                distractors.append(other.statement.replace(other.concept_name, unit.concept_name))
            else:
                distractors.append(other.statement)
        deduped = []
        seen: set[str] = set()
        for item in distractors:
            normalized = normalize_text(item)
            if not normalized or normalized == normalize_text(unit.statement) or normalized in seen:
                continue
            deduped.append(item)
            seen.add(normalized)
        return deduped
