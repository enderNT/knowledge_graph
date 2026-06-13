from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass

from fastapi import HTTPException

from app.ai_provider import AIProvider
from app.config import Settings
from app.learning_context import TutorContextBuilder

logger = logging.getLogger(__name__)
from app.pedagogical_assessment import PedagogicalAssessmentService
from app.pedagogical_context import PedagogicalContextBuilder
from app.schemas import (
    ApplyPrereqReliefRequest,
    AdaptiveBlockPlan,
    AdaptiveBlockPurpose,
    AdaptiveBlockResponse,
    AdaptiveBlockResult,
    AdaptiveBlockSubmissionRequest,
    AdaptiveBlockSubmissionResponse,
    AdaptiveDifficulty,
    AdaptiveInteractionEvent,
    AdaptiveItemResult,
    AdaptiveQuestionType,
    AdaptiveSessionSnapshot,
    AdaptiveSessionStartRequest,
    AdaptiveSessionStartResponse,
    AdaptiveSessionSummary,
    AdaptiveSuccessCriteria,
    AdaptiveNextStepPolicy,
    AdaptiveScaffoldingPolicy,
    AdaptiveVerdict,
    GeneratedAdaptiveBlock,
    PedagogicalConceptState,
    PedagogicalContextSnapshot,
    PedagogicalDimension,
    PedagogicalDimensionState,
    PedagogicalDimensionStates,
    PedagogicalDomainState,
    PedagogicalEvaluationEvent,
    PedagogicalRecentStats,
    PedagogicalRecalculationTrace,
    TutorContextRequest,
    TutorContextResolvedReference,
    TutorContextResponse,
    AdaptiveBlockAnswerKey,
    AdaptiveBlockItem,
    AdaptiveItemSubmission,
    AdaptiveSessionConstraints,
    DueSRItem,
    StudyMode,
    UpdateSRFromBlockRequest,
)
from app.spaced_repetition import SpacedRepetitionService
from app.store import KnowledgeStore
from app.trace import bind
from app.utils import make_prefixed_id, normalize_text, utcnow_iso


_SIGNIFICANT_STOPWORDS = {
    "a",
    "al",
    "como",
    "con",
    "de",
    "del",
    "el",
    "en",
    "es",
    "la",
    "las",
    "los",
    "para",
    "por",
    "que",
    "se",
    "su",
    "un",
    "una",
    "y",
}
_RELATION_WEIGHTS = {
    "PREREQUISITE_FOR": 0.35,
    "PART_OF": 0.22,
    "IS_A": 0.15,
}


@dataclass
class AdaptivePlannerDecision:
    concept: PedagogicalConceptState
    block_purpose: AdaptiveBlockPurpose
    block_goal: str
    difficulty: AdaptiveDifficulty
    primary_dimension: PedagogicalDimension
    question_types: list[AdaptiveQuestionType]
    scaffolding: AdaptiveScaffoldingPolicy
    explanation: str
    due_item: DueSRItem | None = None
    corrective_explanation_seed: str = ""


class AdaptiveLearningService:
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
        self._sr_service = SpacedRepetitionService(settings=settings, store=store)
        self._assessment = PedagogicalAssessmentService()

    async def start_session(
        self,
        payload: AdaptiveSessionStartRequest,
    ) -> AdaptiveSessionStartResponse:
        bind(step="start_session")
        t0 = time.monotonic()
        logger.info(
            "adaptive session starting",
            extra={"user_id": payload.user_id, "study_mode": payload.study_mode, "domain_hint": payload.domain_hint},
        )
        tutor_context = await self._resolve_tutor_context(payload)
        pedagogical_context = await self._store.get_pedagogical_context(user_id=payload.user_id)
        review_candidates = await self._review_candidates_for_mode(
            user_id=payload.user_id,
            study_mode=payload.study_mode,
            domain_hint=payload.domain_hint,
        )
        review_target, new_target = self._session_targets(
            study_mode=payload.study_mode,
            review_candidate_count=len(review_candidates),
            max_blocks=payload.constraints.max_blocks,
        )
        if payload.study_mode in {"backlog", "recovery"} and not review_candidates:
            raise HTTPException(status_code=422, detail="no_review_candidates_for_mode")
        block = await self._build_next_block(
            user_id=payload.user_id,
            constraints=payload.constraints,
            route_tutor_context=tutor_context,
            pedagogical_context=pedagogical_context,
            previous_result=None,
            study_mode=payload.study_mode,
            domain_hint=payload.domain_hint,
            review_blocks_completed=0,
            review_blocks_target=review_target,
            review_candidates=review_candidates,
            served_evidence_uids=[],
            language=payload.language,
        )
        initial_served_uids = [uid for key in block.answer_keys for uid in key.evidence_unit_ids]
        session = AdaptiveSessionSnapshot(
            session_id=make_prefixed_id("ads"),
            user_id=payload.user_id,
            study_mode=payload.study_mode,
            resolved_reference=tutor_context.resolved_reference,
            domain_hint=payload.domain_hint,
            language=payload.language,
            constraints=payload.constraints,
            tutor_context=tutor_context,
            current_block=block,
            block_history=[],
            served_evidence_uids=initial_served_uids,
            summary=AdaptiveSessionSummary(
                total_blocks=payload.constraints.max_blocks,
                completed_blocks=0,
                review_blocks_target=review_target,
                new_blocks_target=new_target,
                review_blocks_completed=0,
                new_blocks_completed=0,
                latest_block_verdict=None,
                session_closed=False,
                closure_reason=None,
            ),
            status="active",
            opened_at=utcnow_iso(),
            updated_at=utcnow_iso(),
        )
        await self._store.upsert_adaptive_session(session)
        await self._store.append_adaptive_block_attempt(
            session_id=session.session_id,
            block=block,
            answer_keys=block.answer_keys,
            submissions=[],
            interaction_events=[],
            block_result=None,
        )
        bind(run_id=session.session_id)
        logger.info(
            "adaptive session created",
            extra={
                "session_id": session.session_id,
                "user_id": session.user_id,
                "study_mode": session.study_mode,
                "block_id": block.block_id,
                "target_concept": block.plan.target_concept_name,
                "duration_ms": int((time.monotonic() - t0) * 1000),
            },
        )
        return AdaptiveSessionStartResponse(
            session=session,
            current_block=block,
            planner_explanation=block.plan.planner_explanation,
            grounding_status="ok",
        )

    async def get_session(self, session_id: str) -> AdaptiveSessionSnapshot:
        session = await self._store.get_adaptive_session(session_id=session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="adaptive session not found")
        return session

    async def submit_block(
        self,
        *,
        session_id: str,
        payload: AdaptiveBlockSubmissionRequest,
    ) -> AdaptiveBlockSubmissionResponse:
        bind(run_id=session_id, step="submit_block")
        t0 = time.monotonic()
        logger.info(
            "block submission received",
            extra={"session_id": session_id, "block_id": payload.block_id, "items_count": len(payload.submissions)},
        )
        session = await self.get_session(session_id)
        current_block = session.current_block
        if current_block is None or current_block.block_id != payload.block_id:
            logger.warning(
                "block mismatch",
                extra={"session_id": session_id, "submitted": payload.block_id, "current": current_block.block_id if current_block else None},
            )
            raise HTTPException(status_code=409, detail="current adaptive block mismatch")

        item_map = {item.item_id: item for item in current_block.items}
        key_map = {item.item_id: item for item in current_block.answer_keys}
        event_map = {item.item_id: item for item in payload.interaction_events}
        submission_map = {item.item_id: item for item in payload.submissions}

        item_results: list[AdaptiveItemResult] = []
        dimension_scores: dict[PedagogicalDimension, list[float]] = defaultdict(list)
        for item in current_block.items:
            submission = submission_map.get(item.item_id)
            if submission is None:
                raise HTTPException(status_code=422, detail=f"missing submission for {item.item_id}")
            result = await self._evaluate_item(
                item=item,
                answer_key=key_map[item.item_id],
                submission=submission,
                interaction=event_map.get(item.item_id, AdaptiveInteractionEvent(item_id=item.item_id)),
                concept_name=current_block.plan.target_concept_name,
                language=session.language,
            )
            item_results.append(result)
            dimension_scores[item.target_dimension].append(result.score_0_to_1)

        dimension_summary = {
            dimension: round(sum(scores) / len(scores), 2)
            for dimension, scores in dimension_scores.items()
        }
        block_score = round(sum(result.score_0_to_1 for result in item_results) / len(item_results), 2)
        block_verdict = self._band_verdict(block_score)
        next_action = self._next_action_from_verdict(block_verdict)
        avg_coverage = round(
            sum(result.signals.get("coverage", 0.0) for result in item_results) / max(len(item_results), 1),
            2,
        )
        avg_precision = round(
            sum(result.signals.get("precision", 0.0) for result in item_results) / max(len(item_results), 1),
            2,
        )
        sr_update = await self._sr_service.update_from_block_result(
            UpdateSRFromBlockRequest(
                user_id=session.user_id,
                concept_uid=current_block.plan.target_concept_uid,
                dimension=current_block.plan.target_dimensions[0],
                block_verdict=block_verdict,
                block_difficulty=current_block.plan.difficulty,
                hint_used=any(item.used_hint for item in item_results),
                retry_used=any(item.used_retry for item in item_results),
                coverage=avg_coverage,
                precision=avg_precision,
                was_direct_evaluation=True,
            )
        )
        await self._sr_service.apply_prereq_relief(
            ApplyPrereqReliefRequest(
                user_id=session.user_id,
                source_concept_uid=current_block.plan.target_concept_uid,
                source_dimension=current_block.plan.target_dimensions[0],
                quality_q=sr_update.sr_feedback.calculated_quality_q,
            )
        )
        concept_state = await self._upsert_state_from_block(
            session=session,
            block=current_block,
            item_results=item_results,
            dimension_summary=dimension_summary,
            block_score=block_score,
        )
        updated_context = await self._store.get_pedagogical_context(user_id=session.user_id)
        block_result = AdaptiveBlockResult(
            block_id=current_block.block_id,
            item_results=item_results,
            dimension_summary=dimension_summary,
            block_verdict=block_verdict,
            block_score=block_score,
            sr_feedback=sr_update.sr_feedback,
            recommended_next_action=next_action,
            corrective_explanation=current_block.plan.corrective_explanation_seed or self._corrective_explanation(session.tutor_context),
            transition_explanation=self._transition_explanation(block_verdict, concept_state.concept_name),
        )
        await self._store.append_adaptive_block_attempt(
            session_id=session_id,
            block=current_block,
            answer_keys=current_block.answer_keys,
            submissions=payload.submissions,
            interaction_events=payload.interaction_events,
            block_result=block_result,
        )

        review_blocks_completed = session.summary.review_blocks_completed + (
            1 if current_block.plan.block_purpose == "spaced_repetition_review" else 0
        )
        new_blocks_completed = session.summary.new_blocks_completed + (
            1 if current_block.plan.block_purpose == "new_content" else 0
        )
        session_closed = (
            len(session.block_history) + 1 >= session.constraints.max_blocks
            or (
                block_verdict == "correct"
                and concept_state.dimensions.weakest_dimension()[1].score_0_to_100 >= 70
            )
        )
        next_block = None
        if not session_closed:
            review_candidates = await self._review_candidates_for_mode(
                user_id=session.user_id,
                study_mode=session.study_mode,
                domain_hint=session.domain_hint,
            )
            next_block = await self._build_next_block(
                user_id=session.user_id,
                constraints=session.constraints,
                route_tutor_context=session.tutor_context,
                pedagogical_context=updated_context,
                previous_result=block_result,
                study_mode=session.study_mode,
                domain_hint=session.domain_hint,
                review_blocks_completed=review_blocks_completed,
                review_blocks_target=session.summary.review_blocks_target,
                review_candidates=review_candidates,
                served_evidence_uids=session.served_evidence_uids,
                language=session.language,
            )
            await self._store.append_adaptive_block_attempt(
                session_id=session_id,
                block=next_block,
                answer_keys=next_block.answer_keys,
                submissions=[],
                interaction_events=[],
                block_result=None,
            )

        new_served_uids = (
            [*session.served_evidence_uids, *[uid for key in next_block.answer_keys for uid in key.evidence_unit_ids]]
            if next_block is not None
            else session.served_evidence_uids
        )
        updated_session = session.model_copy(
            update={
                "current_block": next_block,
                "block_history": [*session.block_history, block_result],
                "served_evidence_uids": new_served_uids,
                "summary": session.summary.model_copy(
                    update={
                        "completed_blocks": session.summary.completed_blocks + 1,
                        "review_blocks_completed": review_blocks_completed,
                        "new_blocks_completed": new_blocks_completed,
                        "latest_block_verdict": block_verdict,
                        "session_closed": session_closed,
                        "closure_reason": "coverage_sufficient"
                        if session_closed and block_verdict == "correct"
                        else ("max_blocks_reached" if session_closed else None),
                    }
                ),
                "status": "closed" if session_closed else "active",
                "updated_at": utcnow_iso(),
            }
        )
        await self._store.upsert_adaptive_session(updated_session)
        logger.info(
            "block submission complete",
            extra={
                "session_id": session_id,
                "block_id": payload.block_id,
                "verdict": block_verdict,
                "score": block_score,
                "session_status": updated_session.status,
                "session_closed": session_closed,
                "completed_blocks": updated_session.summary.completed_blocks,
                "duration_ms": int((time.monotonic() - t0) * 1000),
            },
        )
        return AdaptiveBlockSubmissionResponse(
            session=updated_session,
            block_result=block_result,
            updated_context=updated_context,
            next_action=next_action,
            next_block=next_block,
            session_closed=session_closed,
        )

    async def _resolve_tutor_context(self, payload: AdaptiveSessionStartRequest) -> TutorContextResponse:
        if payload.study_mode == "auto" and not any([payload.query, payload.episode_id, payload.job_id]):
            return await self._auto_resolve_tutor_context(
                user_id=payload.user_id,
                domain_hint=payload.domain_hint,
            )
        builder = TutorContextBuilder(
            settings=self._settings,
            store=self._store,
            ai_provider=self._ai_provider,
        )
        tutor_context = await builder.build(
            TutorContextRequest(
                query=payload.query,
                episode_id=payload.episode_id,
                job_id=payload.job_id,
                depth=1,
                include_evidence=True,
            )
        )
        if tutor_context.status != "ok":
            raise HTTPException(
                status_code=422,
                detail=tutor_context.failure_reason or "insufficient_pedagogical_evidence",
            )
        return tutor_context

    async def _auto_resolve_tutor_context(
        self,
        *,
        user_id: str,
        domain_hint: str | None,
    ) -> TutorContextResponse:
        due_items = await self._sr_service.list_review_candidates_for_planner(
            user_id=user_id,
            domain_hint=domain_hint,
            recovery_only=False,
        )
        candidates: list[tuple[str, str]] = []
        for item in due_items:
            candidates.append((item.concept_uid, item.concept_name or item.concept_uid))
        if not candidates:
            pedagogical_context = await self._store.get_pedagogical_context(user_id=user_id)
            states = pedagogical_context.concepts
            if domain_hint:
                states = [s for s in states if s.domain == domain_hint]
            states.sort(key=lambda s: (-s.priority_score, s.dimensions.weakest_dimension()[1].score_0_to_100))
            candidates = [(s.concept_uid, s.concept_name) for s in states]
        if not candidates:
            raise HTTPException(status_code=422, detail="no_content_available_for_auto_mode")
        for concept_uid, concept_name in candidates:
            tutor_context = await self._resolve_tutor_context_for_concept(
                concept_uid=concept_uid,
                concept_name=concept_name,
            )
            if tutor_context is not None:
                return tutor_context
        raise HTTPException(status_code=422, detail="no_traceable_evidence_found_for_auto_mode")

    async def _select_target_concept(
        self,
        *,
        user_id: str,
        tutor_context: TutorContextResponse,
        pedagogical_context: PedagogicalContextSnapshot,
        domain_hint: str | None = None,
    ) -> PedagogicalConceptState:
        state_map = {state.concept_uid: state for state in pedagogical_context.concepts}
        candidates: list[PedagogicalConceptState] = []
        for concept in tutor_context.concepts:
            if domain_hint is not None and concept.domain != domain_hint:
                continue
            existing = state_map.get(concept.uid)
            if existing is not None:
                candidates.append(existing)
                continue
            candidates.append(
                self._default_concept_state(
                    user_id=user_id,
                    concept_uid=concept.uid,
                    concept_name=concept.canonical_name,
                    domain=concept.domain,
                )
            )
        if not candidates:
            if domain_hint is not None:
                raise HTTPException(status_code=422, detail="no_concepts_available_for_domain")
            raise HTTPException(status_code=422, detail="tutor context returned no concepts for adaptive planning")
        candidates.sort(
            key=lambda item: (
                -item.priority_score,
                item.dimensions.weakest_dimension()[1].score_0_to_100,
                item.confidence_0_to_1,
            )
        )
        return candidates[0]

    async def _build_next_block(
        self,
        *,
        user_id: str,
        constraints: AdaptiveSessionConstraints,
        route_tutor_context: TutorContextResponse,
        pedagogical_context: PedagogicalContextSnapshot,
        previous_result: AdaptiveBlockResult | None,
        study_mode: StudyMode,
        domain_hint: str | None,
        review_blocks_completed: int,
        review_blocks_target: int,
        review_candidates: list[DueSRItem],
        served_evidence_uids: list[str],
        language: str = "es",
    ) -> AdaptiveBlockResponse:
        if review_blocks_completed < review_blocks_target:
            for due_item in review_candidates:
                review_block = await self._build_review_block(
                    user_id=user_id,
                    constraints=constraints,
                    pedagogical_context=pedagogical_context,
                    due_item=due_item,
                    served_evidence_uids=served_evidence_uids,
                    language=language,
                )
                if review_block is not None:
                    return review_block
            if study_mode in {"backlog", "recovery"}:
                raise HTTPException(status_code=422, detail="no_review_candidates_for_mode")

        concept_state = await self._select_target_concept(
            user_id=user_id,
            tutor_context=route_tutor_context,
            pedagogical_context=pedagogical_context,
            domain_hint=domain_hint if study_mode == "isolated" else None,
        )
        decision = self._plan_block(
            concept_state=concept_state,
            constraints=constraints,
            previous_result=previous_result,
            block_purpose="new_content",
        )
        return await self._generate_block(
            tutor_context=route_tutor_context,
            concept_state=concept_state,
            constraints=constraints,
            decision=decision,
            served_evidence_uids=served_evidence_uids,
            language=language,
        )

    async def _review_candidates_for_mode(
        self,
        *,
        user_id: str,
        study_mode: StudyMode,
        domain_hint: str | None,
    ) -> list[DueSRItem]:
        return await self._sr_service.list_review_candidates_for_planner(
            user_id=user_id,
            domain_hint=domain_hint if study_mode in {"isolated", "auto"} else None,
            recovery_only=study_mode == "recovery",
        )

    async def _build_review_block(
        self,
        *,
        user_id: str,
        constraints: AdaptiveSessionConstraints,
        pedagogical_context: PedagogicalContextSnapshot,
        due_item: DueSRItem,
        served_evidence_uids: list[str],
        language: str = "es",
    ) -> AdaptiveBlockResponse | None:
        tutor_context = await self._resolve_tutor_context_for_concept(
            concept_uid=due_item.concept_uid,
            concept_name=due_item.concept_name,
        )
        if tutor_context is None:
            return None
        concept_state = self._concept_state_for_due_item(
            user_id=user_id,
            pedagogical_context=pedagogical_context,
            tutor_context=tutor_context,
            due_item=due_item,
        )
        decision = self._plan_block(
            concept_state=concept_state,
            constraints=constraints,
            previous_result=None,
            block_purpose="spaced_repetition_review",
            forced_dimension=due_item.dimension,
            due_item=due_item,
        )
        return await self._generate_block(
            tutor_context=tutor_context,
            concept_state=concept_state,
            constraints=constraints,
            decision=decision,
            served_evidence_uids=served_evidence_uids,
            language=language,
        )

    async def _resolve_tutor_context_for_concept(
        self,
        *,
        concept_uid: str,
        concept_name: str,
    ) -> TutorContextResponse | None:
        builder = TutorContextBuilder(
            settings=self._settings,
            store=self._store,
            ai_provider=self._ai_provider,
        )
        bundle = await self._store.get_tutor_context_for_concept(concept_uid, depth=1)
        response = builder._finalize_response(
            resolved=TutorContextResolvedReference(
                input_type="query",
                input_value=concept_name or concept_uid,
                resolved_concept_uid=concept_uid,
                resolved_concept_name=concept_name or None,
            ),
            bundle=bundle,
            include_evidence=True,
        )
        return response if response.status == "ok" else None

    def _concept_state_for_due_item(
        self,
        *,
        user_id: str,
        pedagogical_context: PedagogicalContextSnapshot,
        tutor_context: TutorContextResponse,
        due_item: DueSRItem,
    ) -> PedagogicalConceptState:
        existing = next(
            (item for item in pedagogical_context.concepts if item.concept_uid == due_item.concept_uid),
            None,
        )
        if existing is not None:
            return existing
        concept = next((item for item in tutor_context.concepts if item.uid == due_item.concept_uid), None)
        concept_name = due_item.concept_name or (concept.canonical_name if concept else due_item.concept_uid)
        domain = due_item.domain or (concept.domain if concept else "General")
        return self._default_concept_state(
            user_id=user_id,
            concept_uid=due_item.concept_uid,
            concept_name=concept_name,
            domain=domain,
        )

    @staticmethod
    def _session_mix(*, review_candidate_count: int, max_blocks: int) -> tuple[int, int]:
        if review_candidate_count <= 0:
            return 0, max_blocks
        ratio = 0.75 if review_candidate_count >= 4 else 0.5
        review_blocks = round(max_blocks * ratio)
        if max_blocks > 1:
            review_blocks = max(1, min(max_blocks - 1, review_blocks))
        else:
            review_blocks = 1
        review_blocks = min(review_blocks, review_candidate_count)
        return review_blocks, max_blocks - review_blocks

    def _session_targets(
        self,
        *,
        study_mode: StudyMode,
        review_candidate_count: int,
        max_blocks: int,
    ) -> tuple[int, int]:
        if study_mode in {"backlog", "recovery"}:
            return max_blocks, 0
        return self._session_mix(
            review_candidate_count=review_candidate_count,
            max_blocks=max_blocks,
        )

    def _plan_block(
        self,
        *,
        concept_state: PedagogicalConceptState,
        constraints: AdaptiveSessionConstraints,
        previous_result: AdaptiveBlockResult | None,
        block_purpose: AdaptiveBlockPurpose,
        forced_dimension: PedagogicalDimension | None = None,
        due_item: DueSRItem | None = None,
    ) -> AdaptivePlannerDecision:
        primary_dimension = forced_dimension or concept_state.dimensions.weakest_dimension()[0]
        weakest_score = getattr(concept_state.dimensions, primary_dimension).score_0_to_100
        if forced_dimension is None and previous_result is not None:
            primary_dimension = self._follow_up_dimension(previous_result, concept_state)
            weakest_score = getattr(concept_state.dimensions, primary_dimension).score_0_to_100
        if concept_state.confidence_0_to_1 < 0.45:
            block_goal = "diagnose_uncertain"
        elif weakest_score < 55:
            block_goal = "reinforce_weak"
        elif previous_result and previous_result.block_verdict == "correct" and block_purpose == "new_content":
            block_goal = "test_depth"
        else:
            block_goal = "confirm_improvement"
        difficulty = self._difficulty_for_score(weakest_score)
        if constraints.preferred_max_difficulty:
            difficulty = min(
                [difficulty, constraints.preferred_max_difficulty],
                key=lambda item: ["introductory", "intermediate", "advanced"].index(item),
            )
        question_types = self._question_types_for_dimension(primary_dimension, constraints)
        scaffolding = self._scaffolding_for_score(weakest_score, constraints.allow_scaffolding)
        if block_purpose == "spaced_repetition_review" and due_item is not None:
            explanation = (
                f"Se prioriza repaso SR de {concept_state.concept_name} en {primary_dimension} "
                f"porque está vencido {due_item.days_overdue} dias."
            )
        else:
            explanation = (
                f"Se prioriza {concept_state.concept_name} porque su prioridad es {concept_state.priority_score:.2f}, "
                f"la dimension objetivo es {primary_dimension} y su senal actual es {weakest_score:.1f}."
            )
        return AdaptivePlannerDecision(
            concept=concept_state,
            block_purpose=block_purpose,
            block_goal=block_goal,
            difficulty=difficulty,
            primary_dimension=primary_dimension,
            question_types=question_types,
            scaffolding=scaffolding,
            explanation=explanation,
            due_item=due_item,
            corrective_explanation_seed="",
        )

    async def _generate_block(
        self,
        *,
        tutor_context: TutorContextResponse,
        concept_state: PedagogicalConceptState,
        constraints: AdaptiveSessionConstraints,
        decision: AdaptivePlannerDecision,
        served_evidence_uids: list[str],
        language: str = "es",
    ) -> AdaptiveBlockResponse:
        block_id = make_prefixed_id("blk")
        plan = AdaptiveBlockPlan(
            block_id=block_id,
            block_purpose=decision.block_purpose,
            block_goal=decision.block_goal,
            target_concept_uid=concept_state.concept_uid,
            target_concept_name=concept_state.concept_name,
            concept_domain=concept_state.domain,
            target_dimensions=[decision.primary_dimension],
            recommended_question_types=decision.question_types,
            difficulty=decision.difficulty,
            due_item=decision.due_item,
            corrective_explanation_seed=decision.corrective_explanation_seed or self._corrective_explanation(tutor_context),
            scaffolding=decision.scaffolding,
            success_criteria=AdaptiveSuccessCriteria(),
            next_step_policy=AdaptiveNextStepPolicy(
                on_success="shift_dimension_within_concept_then_reprioritize",
                on_partial_high="stay_local_with_light_support",
                on_partial_low="decrease_difficulty_and_return_to_basic_dimension",
                on_failure="decrease_difficulty_and_move_to_related_or_prerequisite",
            ),
            planner_explanation=decision.explanation,
        )
        item_count = constraints.max_items_per_block
        evidence = self._assessment.select_evidence(
            tutor_context=tutor_context,
            concept_uid=concept_state.concept_uid,
            limit=max(item_count, 3),
        )
        if not evidence:
            raise RuntimeError("validated tutor context did not provide pedagogical evidence")

        evidence_units_for_llm = [
            {"uid": unit.uid, "statement": unit.statement, "supporting_quote": unit.supporting_quote}
            for unit in evidence
        ]
        t0 = time.monotonic()
        llm_block = await self._ai_provider.generate_adaptive_block(
            evidence_units=evidence_units_for_llm,
            target_dimension=decision.primary_dimension,
            difficulty=decision.difficulty,
            question_types=[qt for qt in decision.question_types],
            served_evidence_uids=served_evidence_uids,
            item_count=item_count,
            concept_name=concept_state.concept_name,
            language=language,
        )
        logger.debug(
            "llm block generation",
            extra={
                "block_id": block_id,
                "items_generated": len(llm_block.items),
                "duration_ms": int((time.monotonic() - t0) * 1000),
            },
        )

        evidence_by_uid = {unit.uid: unit for unit in evidence}
        all_evidence_statements = [unit.statement for unit in evidence]
        _grading_mode_for_type = {
            "multiple_choice_single": "deterministic_choice",
            "multiple_choice_multi": "partial_multi_choice",
            "true_false": "deterministic_boolean",
            "cloze": "exact_text",
            "open": "rubric_structured",
        }

        items: list[AdaptiveBlockItem] = []
        answer_keys: list[AdaptiveBlockAnswerKey] = []
        for index in range(item_count):
            question_type = decision.question_types[index % len(decision.question_types)]
            use_llm_item = False
            if index < len(llm_block.items):
                gen = llm_block.items[index]
                unit_for_gen = evidence_by_uid.get(gen.evidence_unit_id) or evidence[index % len(evidence)]
                valid = bool(gen.prompt.strip())
                if valid and question_type in {"multiple_choice_single", "multiple_choice_multi"} and gen.choices:
                    ok, warns = self._assessment.validate_choice_set(
                        choices=gen.choices,
                        correct_indexes=gen.correct_choice_indexes,
                        correct_statement=gen.expected[0] if gen.expected else unit_for_gen.statement,
                        evidence_context=all_evidence_statements,
                    )
                    if not ok:
                        logger.warning(
                            "llm mcq validation failed, falling back to deterministic",
                            extra={"block_id": block_id, "item_index": index, "reasons": warns},
                        )
                        valid = False
                elif valid and question_type in {"multiple_choice_single", "multiple_choice_multi"} and not gen.choices:
                    valid = False
                if valid:
                    use_llm_item = True
                    rubric = self._assessment._rubric_for_statement(
                        gen.expected[0] if gen.expected else unit_for_gen.statement
                    )
                    evidence_uid = gen.evidence_unit_id if gen.evidence_unit_id else unit_for_gen.uid
                    item = AdaptiveBlockItem(
                        item_id=f"{block_id}_item_{index + 1}",
                        question_type=question_type,
                        concept_uid=concept_state.concept_uid,
                        target_dimension=decision.primary_dimension,
                        difficulty=decision.difficulty,
                        prompt=gen.prompt,
                        choices=gen.choices,
                        rubric={"main_signal": decision.primary_dimension},
                        grounding_refs=[evidence_uid],
                        metadata={"generated_by": "llm"},
                    )
                    answer_key = AdaptiveBlockAnswerKey(
                        item_id=f"{block_id}_item_{index + 1}",
                        grading_mode=_grading_mode_for_type.get(question_type, "rubric_structured"),
                        expected=gen.expected if gen.expected else [unit_for_gen.statement],
                        correct_choice_indexes=gen.correct_choice_indexes,
                        boolean_answer=gen.boolean_answer,
                        rationale=gen.rationale,
                        evidence_unit_ids=[evidence_uid],
                        grading_rubric=rubric,
                    )
                    items.append(item)
                    answer_keys.append(answer_key)

            if not use_llm_item:
                item, answer_key = self._assessment.build_adaptive_item(
                    item_id=f"{block_id}_item_{index + 1}",
                    question_type=question_type,
                    concept_uid=concept_state.concept_uid,
                    concept_name=concept_state.concept_name,
                    difficulty=decision.difficulty,
                    dimension=decision.primary_dimension,
                    unit=evidence[index % len(evidence)],
                    alternative_units=evidence,
                )
                items.append(item)
                answer_keys.append(answer_key)

        return AdaptiveBlockResponse(
            block_id=block_id,
            plan=plan,
            items=items,
            answer_keys=answer_keys,
            generated_at=utcnow_iso(),
        )

    @staticmethod
    def _warn_missing_submission_field(item: AdaptiveBlockItem, submission: AdaptiveItemSubmission) -> None:
        """Surface the silent score-to-zero case where the agent sent a submission
        that lacks the field required by the item's question type."""
        qt = item.question_type
        missing: str | None = None
        if qt in {"multiple_choice_single", "multiple_choice_multi"} and not submission.selected_choices:
            missing = "selected_choices"
        elif qt == "true_false" and submission.boolean_answer is None:
            missing = "boolean_answer"
        elif qt in {"open", "cloze"} and not (submission.response_text or "").strip():
            missing = "response_text"
        if missing:
            logger.warning(
                "submission missing expected field for question type",
                extra={"item_id": item.item_id, "question_type": qt, "missing_field": missing},
            )

    @staticmethod
    def _resolve_choice_indexes(raw: list[int | str], choices: list[str]) -> list[int]:
        """Convert choice submissions (index or text) to integer indexes."""
        resolved: list[int] = []
        for value in raw:
            if isinstance(value, int):
                resolved.append(value)
            else:
                normalized = value.strip().lower()
                for idx, choice in enumerate(choices):
                    if choice.strip().lower() == normalized:
                        resolved.append(idx)
                        break
                else:
                    logger.warning("submitted choice text not found in item choices", extra={"value": value[:80]})
        return resolved

    async def _evaluate_item(
        self,
        *,
        item: AdaptiveBlockItem,
        answer_key: AdaptiveBlockAnswerKey,
        submission: AdaptiveItemSubmission,
        interaction: AdaptiveInteractionEvent,
        concept_name: str = "",
        language: str = "es",
    ) -> AdaptiveItemResult:
        selected_indexes = self._resolve_choice_indexes(submission.selected_choices, item.choices)
        self._warn_missing_submission_field(item, submission)
        base_score = 0.0
        coverage = 0.0
        precision = 0.0
        error_signal = 0.15
        used_llm = False
        if item.question_type == "multiple_choice_single":
            base_score = 1.0 if selected_indexes[:1] == answer_key.correct_choice_indexes[:1] else 0.0
            coverage = precision = base_score
            error_signal = 1.0 if base_score else 0.15
        elif item.question_type == "true_false":
            base_score = 1.0 if submission.boolean_answer == answer_key.boolean_answer else 0.0
            coverage = precision = base_score
            error_signal = 1.0 if base_score else 0.15
        elif item.question_type == "multiple_choice_multi":
            expected = set(answer_key.correct_choice_indexes)
            selected = set(selected_indexes)
            true_positive = len(expected & selected)
            false_positive = len(selected - expected)
            coverage = true_positive / max(len(expected), 1)
            precision = true_positive / max(len(selected), 1) if selected else 0.0
            base_score = max(0.0, (coverage * 0.7) + (precision * 0.3) - (false_positive * 0.15))
            error_signal = 1.0 if false_positive == 0 else (0.45 if false_positive == 1 else 0.15)
        else:
            learner_text = submission.response_text or ""
            expected_statement = answer_key.expected[0] if answer_key.expected else ""
            pedagogical_evidence = [str(p) for p in answer_key.grading_rubric.get("points", [])]
            grading = await self._ai_provider.grade_learner_response(
                learner_response=learner_text,
                expected_statement=expected_statement,
                pedagogical_evidence=pedagogical_evidence,
                concept_name=concept_name,
                question_type=item.question_type,
                language=language,
            )
            coverage = grading.coverage
            precision = grading.precision
            base_score = grading.score_0_to_1
            used_llm = grading.used_llm
            if grading.reasoning:
                logger.debug(
                    "llm grading reasoning",
                    extra={"item_id": item.item_id, "used_llm": used_llm, "reasoning": grading.reasoning[:200]},
                )
            if item.question_type == "open":
                support_signal = self._support_signal(interaction)
                stability_signal = 1.0 if not interaction.retry_used else 0.6
                error_signal = 1.0 if coverage >= 0.75 else (0.45 if coverage >= 0.4 else 0.15)
                base_score = (
                    0.45 * coverage
                    + 0.20 * precision
                    + 0.20 * support_signal
                    + 0.15 * stability_signal
                )
        score = self._apply_support_penalty(base_score, interaction)
        verdict = self._band_verdict(score)
        feedback = self._feedback_for_verdict(verdict, item.target_dimension, answer_key.expected[:1])
        return AdaptiveItemResult(
            item_id=item.item_id,
            verdict=verdict,
            score_0_to_1=round(score, 2),
            target_dimension=item.target_dimension,
            used_hint=interaction.hint_used,
            used_retry=interaction.retry_used,
            feedback=feedback,
            signals={
                "coverage": round(coverage, 2),
                "precision": round(precision, 2),
                "support": round(self._support_signal(interaction), 2),
                "error_type": round(error_signal, 2),
                "stability": 1.0 if not interaction.retry_used else 0.6,
                "used_llm": 1.0 if used_llm else 0.0,
            },
        )

    async def _upsert_state_from_block(
        self,
        *,
        session: AdaptiveSessionSnapshot,
        block: AdaptiveBlockResponse,
        item_results: list[AdaptiveItemResult],
        dimension_summary: dict[PedagogicalDimension, float],
        block_score: float,
    ) -> PedagogicalConceptState:
        concept_uid = block.plan.target_concept_uid
        concept_name = block.plan.target_concept_name
        domain = block.plan.concept_domain or next(
            (concept.domain for concept in session.tutor_context.concepts if concept.uid == concept_uid),
            session.domain_hint or "General",
        )
        existing_context = await self._store.get_pedagogical_context(user_id=session.user_id)
        existing_map = {item.concept_uid: item for item in existing_context.concepts}
        existing = existing_map.get(concept_uid) or self._default_concept_state(
            user_id=session.user_id,
            concept_uid=concept_uid,
            concept_name=concept_name,
            domain=domain,
        )
        dimensions = existing.dimensions
        for dimension, score in dimension_summary.items():
            previous = getattr(dimensions, dimension)
            next_score = round((previous.score_0_to_100 * 0.35) + ((score * 100.0) * 0.65), 2)
            dimensions = dimensions.model_copy(
                update={
                    dimension: previous.model_copy(
                        update={"score_0_to_100": next_score, "last_evaluated_at": utcnow_iso()}
                    )
                }
            )
        mastery = dimensions.average_score()
        support_ratio = sum(1 for item in item_results if item.used_hint or item.used_retry) / max(len(item_results), 1)
        confidence = round(max(0.0, min(1.0, existing.confidence_0_to_1 + 0.18 - (support_ratio * 0.08))), 2)
        trace = PedagogicalRecalculationTrace(
            kind="adaptive_block",
            message=f"block={block.block_id} score={block_score:.2f}",
            concept_uid=concept_uid,
            domain=domain,
        )
        event = PedagogicalEvaluationEvent(
            user_id=session.user_id,
            concept_uid=concept_uid,
            concept_name=concept_name,
            domain=domain,
            score_0_to_100=round(block_score * 100.0, 2),
            recorded_at=utcnow_iso(),
        )
        history = [*existing.recent_history, event][-5:]
        recent_stats = PedagogicalContextBuilder(
            settings=self._settings,
            store=self._store,
            ai_provider=self._ai_provider,
        )._build_recent_stats(history)
        updated = PedagogicalConceptState(
            user_id=session.user_id,
            concept_uid=concept_uid,
            concept_name=concept_name,
            domain=domain,
            mastery_score_0_to_100=mastery,
            mastery_label=PedagogicalContextBuilder._label_for_score(mastery),
            dimensions=dimensions,
            confidence_0_to_1=confidence,
            trend=self._trend_from_delta(existing.mastery_score_0_to_100, mastery),
            priority_score=self._priority_for_state(dimensions, confidence),
            last_block_id=block.block_id,
            recent_history=history,
            recent_stats=recent_stats,
            weaknesses=PedagogicalContextBuilder._weaknesses_for_score(mastery, concept_name),
            detected_gaps=PedagogicalContextBuilder._gaps_for_score(mastery, concept_name),
            suggested_questions=PedagogicalContextBuilder._suggested_questions(concept_name, mastery),
            effective_depth_used=PedagogicalContextBuilder._effective_depth_for_score(mastery),
            last_evaluated_at=event.recorded_at,
            updated_at=event.recorded_at,
            recalculation_traces=[*existing.recalculation_traces[-4:], trace],
        )
        await self._store.upsert_pedagogical_concept_state(updated)
        await self._store.append_pedagogical_evaluation_event(event)
        await self._propagate_related_dimensions(source_state=updated, measured_dimensions=list(dimension_summary))
        await self._recalculate_domain_states(user_id=session.user_id)
        return updated

    async def _propagate_related_dimensions(
        self,
        *,
        source_state: PedagogicalConceptState,
        measured_dimensions: list[PedagogicalDimension],
    ) -> None:
        neighbors = await self._store.get_pedagogical_related_concepts(
            concept_uid=source_state.concept_uid,
            max_depth=2,
            allowed_relations={"PREREQUISITE_FOR", "PART_OF", "IS_A"},
        )
        for neighbor in neighbors:
            relation = str(neighbor.get("relation") or "")
            related_uid = str(neighbor.get("concept_uid") or "")
            depth = int(neighbor.get("depth") or 1)
            if not related_uid:
                continue
            concept = await self._store.get_concept(related_uid)
            if concept is None:
                continue
            context = await self._store.get_pedagogical_context(user_id=source_state.user_id)
            existing = next((item for item in context.concepts if item.concept_uid == related_uid), None)
            related_state = existing or self._default_concept_state(
                user_id=source_state.user_id,
                concept_uid=related_uid,
                concept_name=concept.canonical_name,
                domain=concept.domain,
            )
            attenuation = _RELATION_WEIGHTS.get(relation, 0.1) * (0.6 ** max(depth - 1, 0))
            dimensions = related_state.dimensions
            for dimension in measured_dimensions:
                source_dimension = getattr(source_state.dimensions, dimension).score_0_to_100
                previous = getattr(dimensions, dimension)
                propagated_score = round(previous.score_0_to_100 + ((source_dimension - previous.score_0_to_100) * attenuation), 2)
                dimensions = dimensions.model_copy(
                    update={
                        dimension: previous.model_copy(
                            update={"score_0_to_100": propagated_score, "last_evaluated_at": source_state.last_evaluated_at}
                        )
                    }
                )
            mastery = dimensions.average_score()
            propagated = related_state.model_copy(
                update={
                    "dimensions": dimensions,
                    "mastery_score_0_to_100": mastery,
                    "mastery_label": PedagogicalContextBuilder._label_for_score(mastery),
                    "priority_score": self._priority_for_state(dimensions, related_state.confidence_0_to_1),
                    "updated_at": source_state.updated_at,
                    "recalculation_traces": [
                        *related_state.recalculation_traces[-4:],
                        PedagogicalRecalculationTrace(
                            kind="adaptive_propagation",
                            message=f"{relation} depth={depth}",
                            concept_uid=source_state.concept_uid,
                            related_concept_uid=related_uid,
                            domain=concept.domain,
                        ),
                    ],
                }
            )
            await self._store.upsert_pedagogical_concept_state(propagated)

    async def _recalculate_domain_states(self, *, user_id: str) -> None:
        builder = PedagogicalContextBuilder(
            settings=self._settings,
            store=self._store,
            ai_provider=self._ai_provider,
        )
        context = await self._store.get_pedagogical_context(user_id=user_id)
        concept_states = {item.concept_uid: item for item in context.concepts}
        for domain in {item.domain for item in context.concepts}:
            state = await builder._recalculate_domain_state(
                user_id=user_id,
                domain=domain,
                concept_states=concept_states,
            )
            if state is not None:
                await self._store.upsert_pedagogical_domain_state(state)

    @staticmethod
    def _default_concept_state(
        *,
        user_id: str,
        concept_uid: str,
        concept_name: str,
        domain: str,
    ) -> PedagogicalConceptState:
        dimensions = PedagogicalDimensionStates()
        return PedagogicalConceptState(
            user_id=user_id,
            concept_uid=concept_uid,
            concept_name=concept_name,
            domain=domain,
            mastery_score_0_to_100=dimensions.average_score(),
            mastery_label="medio",
            dimensions=dimensions,
            confidence_0_to_1=0.25,
            trend="insufficient_data",
            priority_score=0.5,
            last_block_id=None,
            recent_history=[],
            recent_stats=PedagogicalRecentStats(
                recent_average=0.0,
                trend="insufficient_data",
                deviation=0.0,
                last_evaluated_at=None,
            ),
            weaknesses=[],
            detected_gaps=[],
            suggested_questions=[],
            effective_depth_used=3,
            last_evaluated_at=None,
            updated_at=utcnow_iso(),
            recalculation_traces=[],
        )

    @staticmethod
    def _difficulty_for_score(score_0_to_100: float) -> AdaptiveDifficulty:
        if score_0_to_100 < 45:
            return "introductory"
        if score_0_to_100 < 75:
            return "intermediate"
        return "advanced"

    @staticmethod
    def _question_types_for_dimension(
        dimension: PedagogicalDimension,
        constraints: AdaptiveSessionConstraints,
    ) -> list[AdaptiveQuestionType]:
        preferred: dict[PedagogicalDimension, list[AdaptiveQuestionType]] = {
            "recognition": ["multiple_choice_single", "true_false"],
            "recall": ["cloze", "multiple_choice_single"],
            "explanation": ["open"],
            "application": ["multiple_choice_multi", "open"],
        }
        allowed = [item for item in preferred[dimension] if item in constraints.allowed_question_types]
        return allowed or [constraints.allowed_question_types[0]]

    @staticmethod
    def _scaffolding_for_score(score_0_to_100: float, allow_scaffolding: bool) -> AdaptiveScaffoldingPolicy:
        if not allow_scaffolding:
            return AdaptiveScaffoldingPolicy()
        if score_0_to_100 < 45:
            return AdaptiveScaffoldingPolicy(
                allow_hint_after_error=True,
                allow_rephrase_retry=True,
                allow_difficulty_drop_next_item=True,
                show_corrective_explanation_at_end=True,
            )
        if score_0_to_100 < 75:
            return AdaptiveScaffoldingPolicy(
                allow_hint_after_error=True,
                allow_rephrase_retry=True,
                allow_difficulty_drop_next_item=False,
                show_corrective_explanation_at_end=True,
            )
        return AdaptiveScaffoldingPolicy(
            allow_hint_after_error=False,
            allow_rephrase_retry=False,
            allow_difficulty_drop_next_item=False,
            show_corrective_explanation_at_end=True,
        )

    def _follow_up_dimension(
        self,
        previous_result: AdaptiveBlockResult,
        concept_state: PedagogicalConceptState,
    ) -> PedagogicalDimension:
        if previous_result.block_verdict == "correct":
            for dimension in ("recognition", "recall", "explanation", "application"):
                if dimension != concept_state.dimensions.weakest_dimension()[0]:
                    return dimension
        if previous_result.block_verdict == "partial_low":
            return "recognition"
        if previous_result.block_verdict == "incorrect":
            return "recognition"
        return concept_state.dimensions.weakest_dimension()[0]

    @staticmethod
    def _collect_evidence(
        tutor_context: TutorContextResponse,
        concept_uid: str,
    ) -> list[tuple[str, str]]:
        evidence: list[tuple[str, str]] = []
        for claim in tutor_context.claims:
            evidence.append((claim.text, claim.uid))
        for relation in tutor_context.relations:
            if relation.from_uid == concept_uid or relation.to_uid == concept_uid:
                evidence.append((f"{relation.from_name} {relation.relation.lower()} {relation.to_name}.", relation.uid))
        if not evidence:
            for fragment in tutor_context.source_fragments:
                evidence.append((fragment.text, fragment.episode_id))
        return evidence or [("No hubo evidencia suficiente para generar el bloque.", "fallback")]

    @staticmethod
    def _single_choice_distractors(
        concept_name: str,
        statement: str,
        tutor_context: TutorContextResponse,
    ) -> list[str]:
        distractors: list[str] = [
            f"{concept_name} niega la evidencia recuperada y descarta toda relación con el tema.",
            f"{concept_name} solo existe si no hay fragmentos de apoyo trazables.",
        ]
        for concept in tutor_context.concepts:
            if concept.canonical_name == concept_name:
                continue
            distractors.append(f"{concept_name} es exactamente lo mismo que {concept.canonical_name}.")
        if statement not in distractors:
            distractors.append(f"{statement} pero sin apoyo verificable.")
        return distractors

    @staticmethod
    def _completion_phrase(statement: str, concept_name: str) -> str:
        tokens = statement.split()
        if len(tokens) <= 4:
            return statement
        if concept_name in statement:
            parts = statement.split(concept_name, 1)
            candidate = parts[-1].strip(" .,:;")
            if candidate:
                return candidate
        return " ".join(tokens[-4:])

    @staticmethod
    def _semantic_overlap(left: str, right: str) -> float:
        left_tokens = {token for token in normalize_text(left).split() if token and token not in _SIGNIFICANT_STOPWORDS}
        right_tokens = {token for token in normalize_text(right).split() if token and token not in _SIGNIFICANT_STOPWORDS}
        if not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(right_tokens)

    @staticmethod
    def _precision_signal(left: str, right: str) -> float:
        left_norm = normalize_text(left)
        right_norm = normalize_text(right)
        if not left_norm or not right_norm:
            return 0.0
        if left_norm == right_norm:
            return 1.0
        if left_norm in right_norm or right_norm in left_norm:
            return 0.8
        overlap = AdaptiveLearningService._semantic_overlap(left, right)
        if overlap >= 0.75:
            return 0.8
        if overlap >= 0.45:
            return 0.55
        if overlap >= 0.2:
            return 0.25
        return 0.0

    @staticmethod
    def _error_signal_from_overlap(overlap: float) -> float:
        if overlap >= 0.75:
            return 1.0
        if overlap >= 0.45:
            return 0.75
        if overlap >= 0.2:
            return 0.45
        return 0.15

    @staticmethod
    def _support_signal(interaction: AdaptiveInteractionEvent) -> float:
        if interaction.hint_used and interaction.retry_used:
            return 0.3
        if interaction.hint_used or interaction.retry_used:
            return 0.7
        return 1.0

    @staticmethod
    def _apply_support_penalty(base_score: float, interaction: AdaptiveInteractionEvent) -> float:
        multiplier = 1.0
        if interaction.hint_used and interaction.retry_used:
            multiplier = 0.7
        elif interaction.hint_used:
            multiplier = 0.8
        elif interaction.retry_used:
            multiplier = 0.95
        return max(0.0, min(1.0, base_score * multiplier))

    @staticmethod
    def _band_verdict(score_0_to_1: float) -> AdaptiveVerdict:
        if score_0_to_1 >= 0.85:
            return "correct"
        if score_0_to_1 >= 0.65:
            return "partial_high"
        if score_0_to_1 >= 0.35:
            return "partial_low"
        return "incorrect"

    @staticmethod
    def _feedback_for_verdict(
        verdict: AdaptiveVerdict,
        dimension: PedagogicalDimension,
        expected: list[str],
    ) -> str:
        core = expected[0] if expected else "la idea central"
        if verdict == "correct":
            return f"Se recuperó correctamente la evidencia principal para {dimension}."
        if verdict == "partial_high":
            return f"La idea central aparece, pero falta precisión: {core}"
        if verdict == "partial_low":
            return f"Hay una base parcial, pero todavía frágil para {dimension}: {core}"
        return f"La respuesta no recupera el componente nuclear esperado: {core}"

    @staticmethod
    def _next_action_from_verdict(verdict: AdaptiveVerdict) -> str:
        if verdict == "correct":
            return "shift_dimension_within_concept_then_reprioritize"
        if verdict == "partial_high":
            return "stay_local_with_light_support"
        if verdict == "partial_low":
            return "decrease_difficulty_and_return_to_basic_dimension"
        return "decrease_difficulty_and_move_to_related_or_prerequisite"

    @staticmethod
    def _corrective_explanation(tutor_context: TutorContextResponse) -> str:
        if tutor_context.claims:
            return tutor_context.claims[0].text
        if tutor_context.source_fragments:
            return tutor_context.source_fragments[0].text
        return "No hay explicación correctiva disponible."

    @staticmethod
    def _transition_explanation(verdict: AdaptiveVerdict, concept_name: str) -> str:
        if verdict == "correct":
            return f"Se cambia la exigencia dentro de {concept_name} para ampliar la evidencia."
        if verdict == "partial_high":
            return f"Se mantiene el foco en {concept_name} porque aún hay valor diagnóstico local."
        if verdict == "partial_low":
            return f"Se baja dificultad en {concept_name} para reconstruir una base más estable."
        return f"Se prioriza reconstruir prerequisitos o volver a reconocimiento en {concept_name}."

    @staticmethod
    def _trend_from_delta(previous_score: float, new_score: float) -> str:
        delta = new_score - previous_score
        if delta > 5:
            return "improving"
        if delta < -5:
            return "declining"
        return "stable"

    @staticmethod
    def _priority_for_state(dimensions: PedagogicalDimensionStates, confidence_0_to_1: float) -> float:
        weakest = dimensions.weakest_dimension()[1].score_0_to_100
        need = 1.0 - (weakest / 100.0)
        confidence_gap = 1.0 - confidence_0_to_1
        return round(max(0.0, min(1.0, (need * 0.7) + (confidence_gap * 0.3))), 2)
