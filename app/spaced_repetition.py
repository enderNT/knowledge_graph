from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.config import Settings
from app.schemas import (
    ApplyPrereqReliefRequest,
    ApplyPrereqReliefResponse,
    DueSRItem,
    GetDueSRItemsResponse,
    GetSRStateRequest,
    GetSRStateResponse,
    GetSRStatsResponse,
    PedagogicalDimension,
    SRFeedback,
    SRStats,
    SpacedRepetitionState,
    UpdateSRFromBlockRequest,
    UpdateSRFromBlockResponse,
)
from app.store import KnowledgeStore
from app.utils import utcnow_iso


class SpacedRepetitionService:
    def __init__(
        self,
        *,
        settings: Settings,
        store: KnowledgeStore,
    ) -> None:
        self._settings = settings
        self._store = store

    async def get_state(self, payload: GetSRStateRequest) -> GetSRStateResponse:
        state = await self._get_or_create_state(
            user_id=payload.user_id,
            concept_uid=payload.concept_uid,
            dimension=payload.dimension,
        )
        return GetSRStateResponse(state=state)

    async def get_due_items(self, *, user_id: str) -> GetDueSRItemsResponse:
        return GetDueSRItemsResponse(
            user_id=user_id,
            items=await self._store.list_due_spaced_repetition_items(user_id=user_id, now_iso=utcnow_iso()),
        )

    async def update_from_block_result(
        self,
        payload: UpdateSRFromBlockRequest,
    ) -> UpdateSRFromBlockResponse:
        state = await self._get_or_create_state(
            user_id=payload.user_id,
            concept_uid=payload.concept_uid,
            dimension=payload.dimension,
        )
        sr_feedback = self._map_quality(payload)
        now_iso = utcnow_iso()
        updated_state, ef_bonus_applied = self._apply_sm2_update(
            state=state,
            quality_q=sr_feedback.calculated_quality_q,
            now_iso=now_iso,
            was_direct_evaluation=payload.was_direct_evaluation,
        )
        await self._store.upsert_spaced_repetition_state(updated_state)
        return UpdateSRFromBlockResponse(
            state=updated_state,
            sr_feedback=sr_feedback,
            ef_bonus_applied=ef_bonus_applied,
        )

    async def apply_prereq_relief(
        self,
        payload: ApplyPrereqReliefRequest,
    ) -> ApplyPrereqReliefResponse:
        if payload.quality_q < 4 or payload.source_dimension not in {"application", "explanation"}:
            return ApplyPrereqReliefResponse(updated_states=[])

        updated_states: list[SpacedRepetitionState] = []
        parent_dimensions = self._parent_dimensions_for_source(payload.source_dimension)
        for parent_uid in await self._store.get_prerequisite_parent_concepts(concept_uid=payload.source_concept_uid):
            for dimension in parent_dimensions:
                state = await self._get_or_create_state(
                    user_id=payload.user_id,
                    concept_uid=parent_uid,
                    dimension=dimension,
                )
                if state.requires_direct_validation or state.propagation_relief_count >= 2:
                    continue
                updated = self._apply_relief_to_state(state)
                await self._store.upsert_spaced_repetition_state(updated)
                updated_states.append(updated)
        return ApplyPrereqReliefResponse(updated_states=updated_states)

    async def get_stats(self, *, user_id: str) -> GetSRStatsResponse:
        states = await self._store.list_spaced_repetition_states(user_id=user_id)
        now_iso = utcnow_iso()
        due_count = len([state for state in states if state.next_review_at <= now_iso])
        forced_count = len([state for state in states if state.requires_direct_validation])
        average_ef = round(sum(state.ease_factor for state in states) / len(states), 2) if states else 0.0
        return GetSRStatsResponse(
            user_id=user_id,
            stats=SRStats(
                total_items=len(states),
                due_items=due_count,
                forced_review_items=forced_count,
                average_ease_factor=average_ef,
            ),
        )

    async def list_review_candidates_for_planner(
        self,
        *,
        user_id: str,
        domain_hint: str | None = None,
        recovery_only: bool = False,
    ) -> list[DueSRItem]:
        due_items = await self._store.list_due_spaced_repetition_items(user_id=user_id, now_iso=utcnow_iso())
        due_keys = {(item.concept_uid, item.dimension) for item in due_items}
        context = await self._store.get_pedagogical_context(user_id=user_id)
        context_map = {state.concept_uid: state for state in context.concepts}
        forced_items: list[DueSRItem] = []
        for state in await self._store.list_forced_spaced_repetition_states(user_id=user_id):
            if (state.concept_uid, state.dimension) in due_keys:
                continue
            concept_state = context_map.get(state.concept_uid)
            concept_name = concept_state.concept_name if concept_state else ""
            domain = concept_state.domain if concept_state else ""
            if not concept_name or not domain:
                concept = await self._store.get_concept(state.concept_uid)
                if concept is not None:
                    concept_name = concept_name or concept.canonical_name
                    domain = domain or concept.domain
            forced_items.append(
                DueSRItem(
                    concept_uid=state.concept_uid,
                    dimension=state.dimension,
                    next_review_at=state.next_review_at,
                    days_overdue=0,
                    concept_name=concept_name,
                    domain=domain,
                    ease_factor=state.ease_factor,
                    interval_days=state.interval_days,
                    mastery_score_0_to_100=concept_state.mastery_score_0_to_100 if concept_state else 50.0,
                    confidence_0_to_1=concept_state.confidence_0_to_1 if concept_state else 0.25,
                    requires_direct_validation=True,
                )
            )
        forced_items.sort(key=lambda item: (item.mastery_score_0_to_100, item.confidence_0_to_1))
        candidates = [*forced_items, *due_items]
        if domain_hint is not None:
            candidates = [item for item in candidates if item.domain == domain_hint]
        if recovery_only:
            candidates = [item for item in candidates if item.interval_days <= 3 or item.ease_factor <= 1.7]
        return candidates

    async def _get_or_create_state(
        self,
        *,
        user_id: str,
        concept_uid: str,
        dimension: PedagogicalDimension,
    ) -> SpacedRepetitionState:
        existing = await self._store.get_spaced_repetition_state(
            user_id=user_id,
            concept_uid=concept_uid,
            dimension=dimension,
        )
        if existing is not None:
            return existing
        now_iso = utcnow_iso()
        created = SpacedRepetitionState(
            user_id=user_id,
            concept_uid=concept_uid,
            dimension=dimension,
            repetitions=0,
            ease_factor=2.5,
            interval_days=0,
            last_reviewed_at=None,
            next_review_at=now_iso,
            propagation_relief_count=0,
            requires_direct_validation=False,
            updated_at=now_iso,
        )
        await self._store.upsert_spaced_repetition_state(created)
        return created

    def _map_quality(self, payload: UpdateSRFromBlockRequest) -> SRFeedback:
        if payload.block_verdict == "correct":
            quality = 4 if payload.hint_used or payload.retry_used else 5
            rationale = "Respuesta correcta con apoyo" if quality == 4 else "Respuesta correcta directa sin apoyo"
        elif payload.block_verdict == "partial_high":
            if payload.block_difficulty == "advanced":
                quality = 4
            elif payload.block_difficulty == "intermediate":
                quality = 3
            else:
                quality = 2
            rationale = f"Respuesta partial_high en dificultad {payload.block_difficulty}"
        elif payload.block_verdict == "partial_low":
            quality = 2
            rationale = "Respuesta partial_low: requiere repaso a corto plazo"
        else:
            quality = 1 if payload.coverage >= 0.20 or payload.precision >= 0.20 else 0
            rationale = (
                "Respuesta incorrect con estructura parcial recuperable"
                if quality == 1
                else "Respuesta incorrect sin estructura recuperable"
            )
        return SRFeedback(
            concept_uid=payload.concept_uid,
            dimension=payload.dimension,
            calculated_quality_q=quality,
            rationale=rationale,
        )

    def _apply_sm2_update(
        self,
        *,
        state: SpacedRepetitionState,
        quality_q: int,
        now_iso: str,
        was_direct_evaluation: bool,
    ) -> tuple[SpacedRepetitionState, bool]:
        ef_bonus_applied = False
        previous_interval = max(state.interval_days, 1)
        if quality_q < 3:
            updated = state.model_copy(
                update={
                    "repetitions": 1,
                    "interval_days": 1,
                    "last_reviewed_at": now_iso if was_direct_evaluation else state.last_reviewed_at,
                    "next_review_at": self._shift_days(now_iso, 1),
                    "propagation_relief_count": 0 if was_direct_evaluation else state.propagation_relief_count,
                    "requires_direct_validation": False if was_direct_evaluation else state.requires_direct_validation,
                    "updated_at": now_iso,
                }
            )
            return updated, ef_bonus_applied

        repetitions = state.repetitions + 1
        if repetitions == 1:
            interval_days = 1
        elif repetitions == 2:
            interval_days = 6
        else:
            interval_days = max(1, int(round(previous_interval * state.ease_factor)))

        if was_direct_evaluation and quality_q == 5 and abs(state.ease_factor - 1.3) < 1e-9:
            ease_factor = 2.0
            ef_bonus_applied = True
        else:
            ease_factor = max(
                1.3,
                round(
                    state.ease_factor
                    + (0.1 - (5 - quality_q) * (0.08 + (5 - quality_q) * 0.02)),
                    2,
                ),
            )

        updated = state.model_copy(
            update={
                "repetitions": repetitions,
                "ease_factor": ease_factor,
                "interval_days": interval_days,
                "last_reviewed_at": now_iso if was_direct_evaluation else state.last_reviewed_at,
                "next_review_at": self._shift_days(now_iso, interval_days),
                "propagation_relief_count": 0 if was_direct_evaluation else state.propagation_relief_count,
                "requires_direct_validation": False if was_direct_evaluation else state.requires_direct_validation,
                "updated_at": now_iso,
            }
        )
        return updated, ef_bonus_applied

    def _apply_relief_to_state(self, state: SpacedRepetitionState) -> SpacedRepetitionState:
        now_iso = utcnow_iso()
        effective_interval = max(state.interval_days, 1)
        next_relief_count = state.propagation_relief_count + 1
        return state.model_copy(
            update={
                "next_review_at": self._shift_days(now_iso, effective_interval * 1.25),
                "propagation_relief_count": next_relief_count,
                "requires_direct_validation": next_relief_count >= 2,
                "updated_at": now_iso,
            }
        )

    @staticmethod
    def _parent_dimensions_for_source(source_dimension: PedagogicalDimension) -> list[PedagogicalDimension]:
        if source_dimension == "application":
            return ["application", "recall"]
        if source_dimension == "explanation":
            return ["explanation", "recall"]
        return []

    @staticmethod
    def _shift_days(now_iso: str, days: float) -> str:
        base = datetime.fromisoformat(now_iso)
        if base.tzinfo is None:
            base = base.replace(tzinfo=UTC)
        return (base + timedelta(days=days)).replace(microsecond=0).isoformat()
