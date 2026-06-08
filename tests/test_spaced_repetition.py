from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from app.ai_provider import StubAIProvider
from app.schemas import (
    ApplyPrereqReliefRequest,
    CreateConceptRequest,
    GetSRStateRequest,
    PedagogicalConceptState,
    PedagogicalDimensionState,
    PedagogicalDimensionStates,
    PedagogicalRecentStats,
    SpacedRepetitionState,
    UpdateSRFromBlockRequest,
)
from app.spaced_repetition import SpacedRepetitionService


def _make_service(settings, store) -> SpacedRepetitionService:
    return SpacedRepetitionService(settings=settings, store=store)


def _seed_concept(store, provider, *, canonical_name: str, domain: str) -> str:
    async def run() -> str:
        concept = await store.create_concept(
            CreateConceptRequest(
                canonical_name=canonical_name,
                aliases=[],
                domain=domain,
                description=f"Descripcion de {canonical_name}.",
            ),
            embedding=await provider.embed(canonical_name),
            source_confidence=1.0,
        )
        return concept.uid

    return asyncio.run(run())


def _pedagogical_state(*, user_id: str, concept_uid: str, concept_name: str, domain: str, mastery: float, confidence: float) -> PedagogicalConceptState:
    return PedagogicalConceptState(
        user_id=user_id,
        concept_uid=concept_uid,
        concept_name=concept_name,
        domain=domain,
        mastery_score_0_to_100=mastery,
        mastery_label="medio",
        dimensions=PedagogicalDimensionStates(
            recognition=PedagogicalDimensionState(score_0_to_100=mastery),
            recall=PedagogicalDimensionState(score_0_to_100=mastery),
            explanation=PedagogicalDimensionState(score_0_to_100=mastery),
            application=PedagogicalDimensionState(score_0_to_100=mastery),
        ),
        confidence_0_to_1=confidence,
        trend="stable",
        priority_score=0.5,
        last_block_id=None,
        recent_history=[],
        recent_stats=PedagogicalRecentStats(recent_average=mastery, trend="stable", deviation=0.0, last_evaluated_at=None),
        weaknesses=[],
        detected_gaps=[],
        suggested_questions=[],
        effective_depth_used=3,
        last_evaluated_at=None,
        updated_at="2026-06-06T10:00:00+00:00",
        recalculation_traces=[],
    )


def test_sr_update_maps_quality_and_applies_bonus(settings, store):
    service = _make_service(settings, store)

    async def run():
        initial = await service.get_state(
            GetSRStateRequest(user_id="user-1", concept_uid="cn_1", dimension="recall")
        )
        assert initial.state.repetitions == 0
        assert initial.state.ease_factor == 2.5

        incorrect = await service.update_from_block_result(
            UpdateSRFromBlockRequest(
                user_id="user-1",
                concept_uid="cn_1",
                dimension="recall",
                block_verdict="incorrect",
                block_difficulty="intermediate",
                coverage=0.25,
                precision=0.1,
            )
        )
        assert incorrect.sr_feedback.calculated_quality_q == 1
        assert incorrect.state.repetitions == 1
        assert incorrect.state.interval_days == 1

        await store.upsert_spaced_repetition_state(
            SpacedRepetitionState(
                user_id="user-1",
                concept_uid="cn_1",
                dimension="recall",
                repetitions=3,
                ease_factor=1.3,
                interval_days=6,
                last_reviewed_at="2026-06-01T10:00:00+00:00",
                next_review_at="2026-06-02T10:00:00+00:00",
                propagation_relief_count=2,
                requires_direct_validation=True,
                updated_at="2026-06-02T10:00:00+00:00",
            )
        )

        recovered = await service.update_from_block_result(
            UpdateSRFromBlockRequest(
                user_id="user-1",
                concept_uid="cn_1",
                dimension="recall",
                block_verdict="correct",
                block_difficulty="intermediate",
                hint_used=False,
                retry_used=False,
                coverage=1.0,
                precision=1.0,
            )
        )
        assert recovered.sr_feedback.calculated_quality_q == 5
        assert recovered.ef_bonus_applied is True
        assert recovered.state.ease_factor == 2.0
        assert recovered.state.propagation_relief_count == 0
        assert recovered.state.requires_direct_validation is False

    asyncio.run(run())


def test_prereq_relief_caps_after_second_relief_and_direct_update_resets(settings, store):
    provider = StubAIProvider(settings)
    parent_uid = _seed_concept(store, provider, canonical_name="Promesas", domain="JavaScript")
    child_uid = _seed_concept(store, provider, canonical_name="async/await", domain="JavaScript")
    service = _make_service(settings, store)

    async def run():
        await store.create_relation(from_ref=parent_uid, relation="PREREQUISITE_FOR", to_ref=child_uid, evidence_episode_id=None)

        first = await service.apply_prereq_relief(
            ApplyPrereqReliefRequest(
                user_id="user-1",
                source_concept_uid=child_uid,
                source_dimension="application",
                quality_q=4,
            )
        )
        assert len(first.updated_states) == 2
        assert all(state.propagation_relief_count == 1 for state in first.updated_states)
        assert all(state.requires_direct_validation is False for state in first.updated_states)

        second = await service.apply_prereq_relief(
            ApplyPrereqReliefRequest(
                user_id="user-1",
                source_concept_uid=child_uid,
                source_dimension="application",
                quality_q=4,
            )
        )
        assert len(second.updated_states) == 2
        assert all(state.propagation_relief_count == 2 for state in second.updated_states)
        assert all(state.requires_direct_validation is True for state in second.updated_states)

        direct = await service.update_from_block_result(
            UpdateSRFromBlockRequest(
                user_id="user-1",
                concept_uid=parent_uid,
                dimension="recall",
                block_verdict="correct",
                block_difficulty="intermediate",
                coverage=1.0,
                precision=1.0,
            )
        )
        assert direct.state.propagation_relief_count == 0
        assert direct.state.requires_direct_validation is False

    asyncio.run(run())


def test_due_items_sort_by_overdue_then_mastery_then_confidence(settings, store):
    service = _make_service(settings, store)
    now = datetime.now(UTC).replace(microsecond=0)

    async def run():
        await store.upsert_pedagogical_concept_state(
            _pedagogical_state(
                user_id="user-1",
                concept_uid="cn_a",
                concept_name="Concepto A",
                domain="Psicología",
                mastery=55.0,
                confidence=0.5,
            )
        )
        await store.upsert_pedagogical_concept_state(
            _pedagogical_state(
                user_id="user-1",
                concept_uid="cn_b",
                concept_name="Concepto B",
                domain="Psicología",
                mastery=40.0,
                confidence=0.6,
            )
        )
        await store.upsert_pedagogical_concept_state(
            _pedagogical_state(
                user_id="user-1",
                concept_uid="cn_c",
                concept_name="Concepto C",
                domain="Psicología",
                mastery=40.0,
                confidence=0.2,
            )
        )

        await store.upsert_spaced_repetition_state(
            SpacedRepetitionState(
                user_id="user-1",
                concept_uid="cn_a",
                dimension="recall",
                repetitions=1,
                ease_factor=2.5,
                interval_days=1,
                last_reviewed_at=None,
                next_review_at=(now - timedelta(days=3)).isoformat(),
                propagation_relief_count=0,
                requires_direct_validation=False,
                updated_at=now.isoformat(),
            )
        )
        await store.upsert_spaced_repetition_state(
            SpacedRepetitionState(
                user_id="user-1",
                concept_uid="cn_b",
                dimension="recall",
                repetitions=1,
                ease_factor=2.5,
                interval_days=1,
                last_reviewed_at=None,
                next_review_at=(now - timedelta(days=3)).isoformat(),
                propagation_relief_count=0,
                requires_direct_validation=False,
                updated_at=now.isoformat(),
            )
        )
        await store.upsert_spaced_repetition_state(
            SpacedRepetitionState(
                user_id="user-1",
                concept_uid="cn_c",
                dimension="recall",
                repetitions=1,
                ease_factor=2.5,
                interval_days=1,
                last_reviewed_at=None,
                next_review_at=(now - timedelta(days=1)).isoformat(),
                propagation_relief_count=0,
                requires_direct_validation=False,
                updated_at=now.isoformat(),
            )
        )

        due = await service.get_due_items(user_id="user-1")
        assert [item.concept_uid for item in due.items] == ["cn_b", "cn_a", "cn_c"]

    asyncio.run(run())


def test_list_review_candidates_for_planner_filters_by_domain_and_recovery(settings, store):
    provider = StubAIProvider(settings)
    service = _make_service(settings, store)
    js_uid = _seed_concept(store, provider, canonical_name="Promesas", domain="JavaScript")
    psych_uid = _seed_concept(store, provider, canonical_name="Memoria", domain="Psicología")

    async def run():
        await store.upsert_pedagogical_concept_state(
            _pedagogical_state(
                user_id="user-1",
                concept_uid=js_uid,
                concept_name="Promesas",
                domain="JavaScript",
                mastery=50.0,
                confidence=0.4,
            )
        )
        await store.upsert_pedagogical_concept_state(
            _pedagogical_state(
                user_id="user-1",
                concept_uid=psych_uid,
                concept_name="Memoria",
                domain="Psicología",
                mastery=45.0,
                confidence=0.3,
            )
        )
        await store.upsert_spaced_repetition_state(
            SpacedRepetitionState(
                user_id="user-1",
                concept_uid=js_uid,
                dimension="recall",
                repetitions=2,
                ease_factor=2.5,
                interval_days=7,
                last_reviewed_at="2026-06-01T10:00:00+00:00",
                next_review_at="2026-06-02T10:00:00+00:00",
                propagation_relief_count=0,
                requires_direct_validation=False,
                updated_at="2026-06-06T10:00:00+00:00",
            )
        )
        await store.upsert_spaced_repetition_state(
            SpacedRepetitionState(
                user_id="user-1",
                concept_uid=psych_uid,
                dimension="recall",
                repetitions=2,
                ease_factor=1.6,
                interval_days=6,
                last_reviewed_at="2026-06-01T10:00:00+00:00",
                next_review_at="2026-06-02T10:00:00+00:00",
                propagation_relief_count=0,
                requires_direct_validation=False,
                updated_at="2026-06-06T10:00:00+00:00",
            )
        )

        js_candidates = await service.list_review_candidates_for_planner(
            user_id="user-1",
            domain_hint="JavaScript",
        )
        assert [item.concept_uid for item in js_candidates] == [js_uid]
        assert js_candidates[0].interval_days == 7
        assert js_candidates[0].ease_factor == 2.5

        recovery_candidates = await service.list_review_candidates_for_planner(
            user_id="user-1",
            recovery_only=True,
        )
        assert [item.concept_uid for item in recovery_candidates] == [psych_uid]

    asyncio.run(run())
