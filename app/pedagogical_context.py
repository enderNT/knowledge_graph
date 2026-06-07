from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from app.ai_provider import AIProvider
from app.config import Settings
from app.schemas import (
    GetPedagogicalContextRequest,
    GetPedagogicalContextResponse,
    PEDAGOGICAL_RELATIONS,
    PedagogicalConceptState,
    PedagogicalDimensionState,
    PedagogicalDimensionStates,
    PedagogicalContextSnapshot,
    PedagogicalDomainState,
    PedagogicalEvaluationEvent,
    PedagogicalEvaluationInput,
    PedagogicalRecentStats,
    PedagogicalRecalculationTrace,
    PedagogicalSessionFocusItem,
    PedagogicalSessionView,
    PedagogicalSessionViewRequest,
    PedagogicalSessionViewResponse,
    UpdatePedagogicalContextRequest,
    UpdatePedagogicalContextResponse,
)
from app.store import KnowledgeStore
from app.utils import utcnow_iso


_RELATION_WEIGHTS = {
    "PREREQUISITE_FOR": 0.35,
    "PART_OF": 0.22,
    "IS_A": 0.15,
}


@dataclass
class PedagogicalRelationNeighbor:
    concept_uid: str
    relation: str
    depth: int


class PedagogicalContextBuilder:
    def __init__(
        self,
        *,
        settings: Settings,
        store: KnowledgeStore,
        ai_provider: AIProvider,
    ) -> None:
        self._settings = settings
        self._store = store
        self._ai_provider = ai_provider

    async def load_context(self, payload: GetPedagogicalContextRequest) -> GetPedagogicalContextResponse:
        context = await self._store.get_pedagogical_context(user_id=payload.user_id)
        filtered = self._filter_context(
            context,
            domain=payload.domain,
            concept_uids=set(payload.concept_uids),
        )
        return GetPedagogicalContextResponse.model_validate(filtered.model_dump())

    async def apply_evaluation_results(
        self,
        payload: UpdatePedagogicalContextRequest,
    ) -> UpdatePedagogicalContextResponse:
        existing = await self._store.get_pedagogical_context(user_id=payload.user_id)
        warnings = list(existing.warnings)
        event_time = payload.session_closed_at or utcnow_iso()
        trace_buffer: list[PedagogicalRecalculationTrace] = []

        concept_states = {state.concept_uid: state for state in existing.concepts}
        domain_names: set[str] = {state.domain for state in existing.domains}
        new_events: list[PedagogicalEvaluationEvent] = []

        for item in payload.evaluations:
            concept = await self._store.get_concept(item.concept_uid)
            if concept is None:
                warnings.append(f"unknown_concept:{item.concept_uid}")
                continue
            domain_names.add(concept.domain)
            event = PedagogicalEvaluationEvent(
                user_id=payload.user_id,
                concept_uid=concept.uid,
                concept_name=concept.canonical_name,
                domain=concept.domain,
                score_0_to_100=item.score_0_to_100,
                recorded_at=item.recorded_at or event_time,
            )
            new_events.append(event)
            updated = self._recalculate_concept_state(
                existing=concept_states.get(concept.uid),
                event=event,
                concept_name=concept.canonical_name,
            )
            concept_states[concept.uid] = updated
            await self._store.upsert_pedagogical_concept_state(updated)
            await self._store.append_pedagogical_evaluation_event(event)
            trace_buffer.extend(updated.recalculation_traces)

            neighbors = await self._store.get_pedagogical_related_concepts(
                concept_uid=concept.uid,
                max_depth=updated.effective_depth_used,
                allowed_relations=PEDAGOGICAL_RELATIONS,
            )
            propagated = await self._apply_propagation(
                user_id=payload.user_id,
                source_state=updated,
                neighbors=neighbors,
                concept_states=concept_states,
                event_time=event.recorded_at,
            )
            trace_buffer.extend(propagated)

        for domain_name in sorted(domain_names):
            domain_state = await self._recalculate_domain_state(
                user_id=payload.user_id,
                domain=domain_name,
                concept_states=concept_states,
            )
            if domain_state is None:
                continue
            await self._store.upsert_pedagogical_domain_state(domain_state)

        context = await self._store.get_pedagogical_context(user_id=payload.user_id)
        merged_warnings = self._dedupe_strings([*context.warnings, *warnings])
        if self._context_status(context) != "not_found":
            merged_warnings = [item for item in merged_warnings if item != "empty_user_context"]
        context = context.model_copy(
            update={
                "warnings": merged_warnings,
                "status": self._context_status(context),
            }
        )
        session_view = await self.build_session_view(
            PedagogicalSessionViewRequest(
                user_id=payload.user_id,
                domain_hint=payload.domain_hint,
                concept_uids=[item.concept_uid for item in payload.evaluations],
            )
        )
        return UpdatePedagogicalContextResponse(
            user_id=payload.user_id,
            status=context.status,
            context=context,
            session_view=session_view,
            warnings=self._dedupe_strings([*warnings, *session_view.warnings]),
        )

    async def build_session_view(
        self,
        payload: PedagogicalSessionViewRequest,
    ) -> PedagogicalSessionViewResponse:
        context = await self._store.get_pedagogical_context(user_id=payload.user_id)
        filtered = self._filter_context(
            context,
            domain=payload.domain_hint,
            concept_uids=set(payload.concept_uids),
        )
        concepts = sorted(
            filtered.concepts,
            key=lambda item: (-item.priority_score, item.dimensions.weakest_dimension()[1].score_0_to_100, item.mastery_score_0_to_100),
        )
        weak_items = [
            PedagogicalSessionFocusItem(
                concept_uid=item.concept_uid,
                concept_name=item.concept_name,
                domain=item.domain,
                mastery_score_0_to_100=item.mastery_score_0_to_100,
                mastery_label=item.mastery_label,
                reason="low_dimension_mastery" if item.dimensions.weakest_dimension()[1].score_0_to_100 < 60 else "review",
            )
            for item in concepts[:5]
        ]
        suggested_questions = self._dedupe_strings(
            [question for item in concepts[:5] for question in item.suggested_questions]
        )[:5]
        detected_gaps = self._dedupe_strings(
            [gap for item in concepts[:5] for gap in item.detected_gaps]
        )[:5]
        traces = [trace for item in concepts[:5] for trace in item.recalculation_traces][:12]
        effective_depth = max((item.effective_depth_used for item in concepts[:5]), default=3)
        domains = [item.domain for item in concepts[:5]]
        status = self._context_status(filtered)

        if not concepts:
            return PedagogicalSessionViewResponse(
                user_id=payload.user_id,
                status="not_found",
                summary="No existe todavia contexto pedagogico persistido para este usuario.",
                weak_concepts=[],
                detected_gaps=[],
                suggested_questions=[],
                effective_depth_used=3,
                domain_focus=[],
                recalculation_traces=[],
                warnings=["empty_user_context"],
            )

        summary = (
            f"Priorizar {len(weak_items)} conceptos con menor dominio y usar profundidad {effective_depth} "
            f"para repaso dirigido."
        )
        return PedagogicalSessionViewResponse(
            user_id=payload.user_id,
            status=status,
            summary=summary,
            weak_concepts=weak_items,
            detected_gaps=detected_gaps,
            suggested_questions=suggested_questions,
            effective_depth_used=effective_depth,
            domain_focus=self._dedupe_strings(domains),
            recalculation_traces=traces,
            warnings=list(filtered.warnings),
        )

    async def _apply_propagation(
        self,
        *,
        user_id: str,
        source_state: PedagogicalConceptState,
        neighbors: list[dict[str, str | int]],
        concept_states: dict[str, PedagogicalConceptState],
        event_time: str,
    ) -> list[PedagogicalRecalculationTrace]:
        propagated_traces: list[PedagogicalRecalculationTrace] = []
        baseline = source_state.recent_stats.recent_average
        source_delta = source_state.mastery_score_0_to_100 - baseline
        if abs(source_delta) < 0.01:
            return propagated_traces

        for raw_neighbor in neighbors:
            relation = str(raw_neighbor.get("relation") or "")
            related_uid = str(raw_neighbor.get("concept_uid") or "")
            depth = int(raw_neighbor.get("depth") or 1)
            if not related_uid or related_uid == source_state.concept_uid:
                continue
            weight = _RELATION_WEIGHTS.get(relation)
            if weight is None:
                continue

            related_concept = await self._store.get_concept(related_uid)
            if related_concept is None:
                continue

            attenuation = weight * (0.6 ** max(depth - 1, 0))
            propagation_delta = source_delta * attenuation
            previous = concept_states.get(related_uid)
            previous_score = previous.mastery_score_0_to_100 if previous else 50.0
            new_score = self._clamp_score(previous_score + propagation_delta)
            existing_history = previous.recent_history if previous else []
            stats = self._build_recent_stats(existing_history)
            carried_dimensions = previous.dimensions if previous else PedagogicalDimensionStates()
            weakest_name, weakest_state = carried_dimensions.weakest_dimension()
            propagated_dimensions = carried_dimensions.model_copy(
                update={
                    weakest_name: weakest_state.model_copy(
                        update={"score_0_to_100": new_score, "last_evaluated_at": event_time}
                    )
                }
            )
            trace = PedagogicalRecalculationTrace(
                kind="propagation",
                message=f"{relation} depth={depth} delta={propagation_delta:.2f}",
                concept_uid=source_state.concept_uid,
                related_concept_uid=related_uid,
                domain=related_concept.domain,
            )
            updated = PedagogicalConceptState(
                user_id=user_id,
                concept_uid=related_uid,
                concept_name=related_concept.canonical_name,
                domain=related_concept.domain,
                mastery_score_0_to_100=new_score,
                mastery_label=self._label_for_score(new_score),
                dimensions=propagated_dimensions,
                confidence_0_to_1=previous.confidence_0_to_1 if previous else 0.2,
                trend=stats.trend,
                priority_score=self._priority_for_state(propagated_dimensions, previous.confidence_0_to_1 if previous else 0.2),
                last_block_id=previous.last_block_id if previous else None,
                recent_history=existing_history,
                recent_stats=stats,
                weaknesses=self._weaknesses_for_score(new_score, related_concept.canonical_name),
                detected_gaps=self._gaps_for_score(new_score, related_concept.canonical_name),
                suggested_questions=self._suggested_questions(related_concept.canonical_name, new_score),
                effective_depth_used=self._effective_depth_for_score(new_score),
                last_evaluated_at=previous.last_evaluated_at if previous else event_time,
                updated_at=event_time,
                recalculation_traces=[*(previous.recalculation_traces if previous else [])[-4:], trace],
            )
            concept_states[related_uid] = updated
            await self._store.upsert_pedagogical_concept_state(updated)
            propagated_traces.append(trace)
        return propagated_traces

    def _recalculate_concept_state(
        self,
        *,
        existing: PedagogicalConceptState | None,
        event: PedagogicalEvaluationEvent,
        concept_name: str,
    ) -> PedagogicalConceptState:
        previous_history = list(existing.recent_history) if existing else []
        fresh_history = [*previous_history, event][-5:]
        recent_average = sum(item.score_0_to_100 for item in fresh_history) / len(fresh_history)
        previous_score = existing.mastery_score_0_to_100 if existing else recent_average
        blended_score = (
            event.score_0_to_100 * 0.6
            + recent_average * 0.25
            + previous_score * 0.15
        )
        decayed_score = self._apply_decay(blended_score, event.recorded_at)
        recent_stats = self._build_recent_stats(fresh_history)
        dimensions = self._dimensions_from_legacy_event(
            existing=existing.dimensions if existing else None,
            score=decayed_score,
            recorded_at=event.recorded_at,
        )
        confidence = self._confidence_for_history(fresh_history)
        trace = PedagogicalRecalculationTrace(
            kind="concept_recalculation",
            message=(
                f"recent={event.score_0_to_100:.2f} average={recent_average:.2f} "
                f"previous={previous_score:.2f} decayed={decayed_score:.2f}"
            ),
            concept_uid=event.concept_uid,
            domain=event.domain,
        )
        return PedagogicalConceptState(
            user_id=event.user_id,
            concept_uid=event.concept_uid,
            concept_name=concept_name,
            domain=event.domain,
            mastery_score_0_to_100=decayed_score,
            mastery_label=self._label_for_score(decayed_score),
            dimensions=dimensions,
            confidence_0_to_1=confidence,
            trend=recent_stats.trend,
            priority_score=self._priority_for_state(dimensions, confidence),
            last_block_id=existing.last_block_id if existing else None,
            recent_history=fresh_history,
            recent_stats=recent_stats,
            weaknesses=self._weaknesses_for_score(decayed_score, concept_name),
            detected_gaps=self._gaps_for_score(decayed_score, concept_name),
            suggested_questions=self._suggested_questions(concept_name, decayed_score),
            effective_depth_used=self._effective_depth_for_score(decayed_score),
            last_evaluated_at=event.recorded_at,
            updated_at=event.recorded_at,
            recalculation_traces=[*(existing.recalculation_traces if existing else [])[-4:], trace],
        )

    async def _recalculate_domain_state(
        self,
        *,
        user_id: str,
        domain: str,
        concept_states: dict[str, PedagogicalConceptState],
    ) -> PedagogicalDomainState | None:
        domain_concepts = [item for item in concept_states.values() if item.domain == domain]
        if not domain_concepts:
            return None

        weighted_total = 0.0
        weight_sum = 0.0
        domain_events: list[PedagogicalEvaluationEvent] = []
        traces: list[PedagogicalRecalculationTrace] = []
        weak_concept_uids: list[str] = []
        for item in domain_concepts:
            neighbors = await self._store.get_pedagogical_related_concepts(
                concept_uid=item.concept_uid,
                max_depth=1,
                allowed_relations=PEDAGOGICAL_RELATIONS,
            )
            structural_weight = 1.0 + sum(
                0.2 for neighbor in neighbors if str(neighbor.get("relation")) == "PREREQUISITE_FOR"
            )
            weighted_total += item.mastery_score_0_to_100 * structural_weight
            weight_sum += structural_weight
            domain_events.extend(item.recent_history)
            traces.extend(item.recalculation_traces[-2:])
            if item.mastery_score_0_to_100 < 60:
                weak_concept_uids.append(item.concept_uid)

        score = self._clamp_score(weighted_total / weight_sum if weight_sum else 0.0)
        recent_stats = self._build_recent_stats(domain_events[-5:])
        trace = PedagogicalRecalculationTrace(
            kind="domain_aggregation",
            message=f"aggregated {len(domain_concepts)} concepts into {score:.2f}",
            domain=domain,
        )
        return PedagogicalDomainState(
            user_id=user_id,
            domain=domain,
            mastery_score_0_to_100=score,
            mastery_label=self._label_for_score(score),
            concept_count=len(domain_concepts),
            weak_concept_uids=weak_concept_uids[:5],
            recent_stats=recent_stats,
            updated_at=utcnow_iso(),
            recalculation_traces=[*traces[-5:], trace],
        )

    def _filter_context(
        self,
        context: PedagogicalContextSnapshot,
        *,
        domain: str | None,
        concept_uids: set[str],
    ) -> PedagogicalContextSnapshot:
        concepts = list(context.concepts)
        domains = list(context.domains)
        if domain:
            concepts = [item for item in concepts if item.domain == domain]
            domains = [item for item in domains if item.domain == domain]
        if concept_uids:
            concepts = [item for item in concepts if item.concept_uid in concept_uids]
            allowed_domains = {item.domain for item in concepts}
            domains = [item for item in domains if item.domain in allowed_domains]
        return PedagogicalContextSnapshot(
            user_id=context.user_id,
            status=self._context_status_from_lists(concepts, domains),
            concepts=concepts,
            domains=domains,
            recent_evaluations=list(context.recent_evaluations),
            warnings=list(context.warnings),
        )

    def _context_status(self, context: PedagogicalContextSnapshot) -> str:
        return self._context_status_from_lists(context.concepts, context.domains)

    @staticmethod
    def _context_status_from_lists(
        concepts: list[PedagogicalConceptState],
        domains: list[PedagogicalDomainState],
    ) -> str:
        if not concepts and not domains:
            return "not_found"
        if not concepts or not domains:
            return "sparse"
        return "ok"

    def _apply_decay(self, score: float, reference_time: str) -> float:
        now = self._parse_iso(utcnow_iso())
        then = self._parse_iso(reference_time)
        if now is None or then is None:
            return self._clamp_score(score)
        days = max((now - then).total_seconds() / 86400.0, 0.0)
        if days <= 7:
            return self._clamp_score(score)
        penalty = min((days - 7) * 0.35, 10.0)
        return self._clamp_score(score - penalty)

    def _build_recent_stats(self, history: list[PedagogicalEvaluationEvent]) -> PedagogicalRecentStats:
        if not history:
            return PedagogicalRecentStats(
                recent_average=0.0,
                trend="insufficient_data",
                deviation=0.0,
                last_evaluated_at=None,
            )
        values = [item.score_0_to_100 for item in history]
        average = sum(values) / len(values)
        if len(values) >= 2:
            delta = values[-1] - values[0]
            if delta > 5:
                trend = "improving"
            elif delta < -5:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        deviation = math.sqrt(sum((value - average) ** 2 for value in values) / len(values))
        return PedagogicalRecentStats(
            recent_average=self._clamp_score(average),
            trend=trend,
            deviation=deviation,
            last_evaluated_at=history[-1].recorded_at,
        )

    @staticmethod
    def _label_for_score(score: float) -> str:
        if score < 20:
            return "muy bajo"
        if score < 40:
            return "bajo"
        if score < 60:
            return "medio"
        if score < 80:
            return "alto"
        return "muy alto"

    @staticmethod
    def _effective_depth_for_score(score: float) -> int:
        if score >= 85:
            return 5
        if score >= 70:
            return 4
        if score >= 35:
            return 3
        return 2

    @staticmethod
    def _weaknesses_for_score(score: float, concept_name: str) -> list[str]:
        if score >= 60:
            return []
        if score >= 40:
            return [f"Necesita consolidar {concept_name} con mas practica guiada."]
        return [f"Presenta dominio fragil en {concept_name} y requiere repaso prioritario."]

    @staticmethod
    def _gaps_for_score(score: float, concept_name: str) -> list[str]:
        if score >= 60:
            return []
        if score >= 40:
            return [f"Aun no conecta {concept_name} con sus relaciones principales."]
        return [f"No demuestra dominio funcional de {concept_name} en evaluacion formal."]

    @staticmethod
    def _suggested_questions(concept_name: str, score: float) -> list[str]:
        if score >= 80:
            return [f"Explica {concept_name} y contrasta un caso limite."]
        if score >= 60:
            return [f"Aplica {concept_name} en un ejemplo nuevo."]
        return [
            f"Define {concept_name} con tus propias palabras.",
            f"Da un ejemplo correcto de {concept_name}.",
        ]

    @staticmethod
    def _clamp_score(value: float) -> float:
        return max(0.0, min(100.0, round(value, 2)))

    @staticmethod
    def _dimensions_from_legacy_event(
        *,
        existing: PedagogicalDimensionStates | None,
        score: float,
        recorded_at: str,
    ) -> PedagogicalDimensionStates:
        if existing is None:
            return PedagogicalDimensionStates(
                recognition=PedagogicalDimensionState(score_0_to_100=score, last_evaluated_at=recorded_at),
                recall=PedagogicalDimensionState(score_0_to_100=score, last_evaluated_at=recorded_at),
                explanation=PedagogicalDimensionState(score_0_to_100=score, last_evaluated_at=recorded_at),
                application=PedagogicalDimensionState(score_0_to_100=score, last_evaluated_at=recorded_at),
            )
        weakest_name, weakest_state = existing.weakest_dimension()
        return existing.model_copy(
            update={
                weakest_name: weakest_state.model_copy(
                    update={"score_0_to_100": score, "last_evaluated_at": recorded_at}
                )
            }
        )

    @staticmethod
    def _confidence_for_history(history: list[PedagogicalEvaluationEvent]) -> float:
        if not history:
            return 0.25
        return min(1.0, 0.25 + len(history) * 0.15)

    @staticmethod
    def _priority_for_state(dimensions: PedagogicalDimensionStates, confidence_0_to_1: float) -> float:
        weakest_score = dimensions.weakest_dimension()[1].score_0_to_100
        need = 1.0 - (weakest_score / 100.0)
        confidence_gap = 1.0 - confidence_0_to_1
        return round(max(0.0, min(1.0, (need * 0.7) + (confidence_gap * 0.3))), 2)

    @staticmethod
    def _dedupe_strings(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    @staticmethod
    def _parse_iso(value: str) -> datetime | None:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            return None
