from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.utils import normalize_text


ALLOWED_RELATIONS = {
    "PART_OF",
    "IS_A",
    "RELATED_TO",
    "EXPLAINS",
    "CONTRASTS_WITH",
    "PREREQUISITE_FOR",
    "SUPPORTED_BY",
    "MENTIONED_IN",
    "ALIAS_OF",
}


class AddKnowledgeFragmentRequest(BaseModel):
    text: str = Field(min_length=1)
    source_type: str = "manual_input"
    tags: list[str] = Field(default_factory=list)
    language: str = "es"


class AddKnowledgeFragmentAccepted(BaseModel):
    episode_id: str
    job_id: str
    status: str


class ResetKnowledgeBaseResponse(BaseModel):
    status: Literal["reset"] = "reset"
    scope: Literal["all"] = "all"
    database_reset: bool = True
    queue_cleared_count: int = 0


class UpsertConceptRequest(BaseModel):
    uid: str | None = None
    canonical_name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    domain: str = Field(min_length=1)
    description: str = ""


class CreateConceptRequest(BaseModel):
    canonical_name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    domain: str = Field(min_length=1)
    description: str = ""


class AttachConceptEvidenceRequest(BaseModel):
    concept_ref: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    link_episode_claims: bool = True


class AttachConceptEvidenceResponse(BaseModel):
    status: Literal["attached"]
    concept_uid: str
    episode_id: str
    linked_claim_count: int = 0


class LinkConceptsRequest(BaseModel):
    from_: str = Field(alias="from", min_length=1)
    relation: str
    to: str = Field(min_length=1)
    evidence_episode_id: str | None = None

    @field_validator("relation")
    @classmethod
    def validate_relation(cls, value: str) -> str:
        relation = value.strip().upper()
        if relation not in ALLOWED_RELATIONS:
            raise ValueError(f"unsupported relation: {relation}")
        return relation


class EpisodeDeletionPreviewRequest(BaseModel):
    episode_id: str | None = Field(default=None, min_length=1)
    job_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_single_reference(self) -> "EpisodeDeletionPreviewRequest":
        provided = [name for name, value in (("episode_id", self.episode_id), ("job_id", self.job_id)) if value]
        if len(provided) != 1:
            raise ValueError("exactly one of episode_id or job_id must be provided")
        return self


class EpisodeDeletionExecuteRequest(EpisodeDeletionPreviewRequest):
    confirm: bool = False

    @model_validator(mode="after")
    def validate_confirm(self) -> "EpisodeDeletionExecuteRequest":
        if self.confirm is not True:
            raise ValueError("confirm must be true")
        return self


class RelationDeletionPreviewRequest(BaseModel):
    from_: str = Field(alias="from", min_length=1)
    relation: str
    to: str = Field(min_length=1)
    evidence_episode_id: str | None = Field(default=None, min_length=1)
    delete_all_matching: bool = False

    @field_validator("relation")
    @classmethod
    def validate_relation(cls, value: str) -> str:
        relation = value.strip().upper()
        if relation not in ALLOWED_RELATIONS:
            raise ValueError(f"unsupported relation: {relation}")
        return relation

    @model_validator(mode="after")
    def validate_scope(self) -> "RelationDeletionPreviewRequest":
        if self.evidence_episode_id and self.delete_all_matching:
            raise ValueError("delete_all_matching cannot be combined with evidence_episode_id")
        return self


class RelationDeletionExecuteRequest(RelationDeletionPreviewRequest):
    confirm: bool = False

    @model_validator(mode="after")
    def validate_confirm(self) -> "RelationDeletionExecuteRequest":
        if self.confirm is not True:
            raise ValueError("confirm must be true")
        return self


class EpisodeDeletionResolvedReference(BaseModel):
    input_type: Literal["episode_id", "job_id"]
    input_value: str
    resolved_episode_id: str
    resolved_job_ids: list[str] = Field(default_factory=list)


class EpisodeDeletionImpactCounts(BaseModel):
    episodes: int = 0
    jobs: int = 0
    pedagogical_evidence: int = 0
    relations: int = 0
    concept_mentions: int = 0
    claim_support_links: int = 0
    orphan_claims: int = 0
    preserved_claims: int = 0
    claim_explain_links: int = 0


class EpisodeDeletionImpact(BaseModel):
    counts: EpisodeDeletionImpactCounts
    episode_ids: list[str] = Field(default_factory=list)
    job_ids: list[str] = Field(default_factory=list)
    pedagogical_evidence_ids: list[str] = Field(default_factory=list)
    relation_uids: list[str] = Field(default_factory=list)
    mentioned_concept_uids: list[str] = Field(default_factory=list)
    supported_claim_uids: list[str] = Field(default_factory=list)
    orphan_claim_uids: list[str] = Field(default_factory=list)
    preserved_claim_uids: list[str] = Field(default_factory=list)
    claim_explain_link_uids: list[str] = Field(default_factory=list)


class EpisodeDeletionPreviewResponse(BaseModel):
    resolved_reference: EpisodeDeletionResolvedReference
    can_execute: bool = True
    warnings: list[str] = Field(default_factory=list)
    impact: EpisodeDeletionImpact


class EpisodeDeletionExecuteResponse(BaseModel):
    status: Literal["deleted"] = "deleted"
    resolved_reference: EpisodeDeletionResolvedReference
    warnings: list[str] = Field(default_factory=list)
    impact: EpisodeDeletionImpact


class RelationDeletionResolvedReference(BaseModel):
    input_from: str
    input_to: str
    from_uid: str
    from_name: str
    relation: str
    to_uid: str
    to_name: str
    matched_relation_uids: list[str] = Field(default_factory=list)
    matched_evidence_episode_ids: list[str] = Field(default_factory=list)
    unscoped_match_count: int = 0


class RelationDeletionImpactCounts(BaseModel):
    relations: int = 0


class RelationDeletionImpact(BaseModel):
    counts: RelationDeletionImpactCounts
    relation_uids: list[str] = Field(default_factory=list)


class RelationDeletionPreviewResponse(BaseModel):
    resolved_reference: RelationDeletionResolvedReference
    can_execute: bool = True
    warnings: list[str] = Field(default_factory=list)
    impact: RelationDeletionImpact


class RelationDeletionExecuteResponse(BaseModel):
    status: Literal["deleted"] = "deleted"
    resolved_reference: RelationDeletionResolvedReference
    warnings: list[str] = Field(default_factory=list)
    impact: RelationDeletionImpact


class SearchCandidatesRequest(BaseModel):
    query: str = Field(min_length=1)
    domain_hint: str | None = None
    limit: int = Field(default=10, ge=1, le=50)


class CandidateHit(BaseModel):
    uid: str
    canonical_name: str
    domain: str
    description: str = ""
    score: float
    reason: str
    matched_alias: str | None = None


class SearchCandidatesResponse(BaseModel):
    query: str
    results: list[CandidateHit]


class LearningContextRequest(BaseModel):
    query: str = Field(min_length=1)
    domain_hint: str | None = None
    candidate_limit: int = Field(default=8, ge=1, le=50)
    concept_limit: int = Field(default=3, ge=1, le=10)
    claim_limit: int = Field(default=6, ge=1, le=20)
    episode_limit: int = Field(default=3, ge=1, le=10)
    include_neighborhood: bool = True
    depth: int = Field(default=1, ge=1, le=2)


class LearningContextPrimaryConcept(BaseModel):
    uid: str
    canonical_name: str
    domain: str
    description: str = ""
    retrieval_score: float
    retrieval_reason: str
    quality_flags: list[str] = Field(default_factory=list)


class LearningContextClaim(BaseModel):
    uid: str
    text: str
    confidence: float | None = None


class LearningContextEpisode(BaseModel):
    uid: str
    text: str
    status: str


class LearningContextDebug(BaseModel):
    candidate_count: int
    selected_concept_uids: list[str] = Field(default_factory=list)
    selection_reasons: list[str] = Field(default_factory=list)


class LearningContextResponse(BaseModel):
    query: str
    domain_hint: str | None = None
    status: Literal["ok", "sparse", "no_match"]
    primary_concepts: list[LearningContextPrimaryConcept] = Field(default_factory=list)
    relations: list["NeighborhoodRelation"] = Field(default_factory=list)
    claims: list[LearningContextClaim] = Field(default_factory=list)
    episodes: list[LearningContextEpisode] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    debug: LearningContextDebug


class TutorContextRequest(BaseModel):
    query: str | None = Field(default=None, min_length=1)
    episode_id: str | None = Field(default=None, min_length=1)
    job_id: str | None = Field(default=None, min_length=1)
    depth: int = Field(default=1, ge=1, le=1)
    include_evidence: bool = True

    @model_validator(mode="after")
    def validate_single_reference(self) -> "TutorContextRequest":
        provided = [name for name, value in (("query", self.query), ("episode_id", self.episode_id), ("job_id", self.job_id)) if value]
        if len(provided) != 1:
            raise ValueError("exactly one of query, episode_id or job_id must be provided")
        return self


class TutorContextResolvedReference(BaseModel):
    input_type: Literal["query", "episode_id", "job_id"]
    input_value: str
    resolved_concept_uid: str | None = None
    resolved_concept_name: str | None = None
    resolved_episode_id: str | None = None
    resolved_job_id: str | None = None
    resolution_reason: str | None = None


class TutorContextConcept(BaseModel):
    uid: str
    canonical_name: str
    domain: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)


class TutorContextClaim(BaseModel):
    uid: str
    text: str
    confidence: float | None = None
    evidence_episode_ids: list[str] = Field(default_factory=list)


class TutorContextRelation(BaseModel):
    uid: str
    from_uid: str
    from_name: str
    relation: str
    to_uid: str
    to_name: str
    confidence: float | None = None
    evidence_episode_ids: list[str] = Field(default_factory=list)


class TutorContextSourceFragment(BaseModel):
    episode_id: str
    text: str
    status: str
    source_type: str
    tags: list[str] = Field(default_factory=list)
    language: str


class TutorContextEvidence(BaseModel):
    subject_type: Literal["concept", "claim", "relation"]
    subject_uid: str
    episode_id: str


class PedagogicalEvidenceRecord(BaseModel):
    uid: str
    concept_uid: str
    concept_name: str
    episode_id: str
    source_claim_uid: str
    statement: str
    supporting_quote: str
    kind: Literal["claim", "definition", "relationship"]
    status: Literal["approved", "rejected", "repairable"]
    review_notes_json: str | None = None
    created_at: str
    updated_at: str


class TutorContextPedagogicalEvidence(BaseModel):
    uid: str
    concept_uid: str
    concept_name: str
    episode_id: str
    source_claim_uid: str
    statement: str
    supporting_quote: str
    kind: Literal["claim", "definition", "relationship"]
    status: Literal["approved", "rejected", "repairable"]


class TutorContextBundle(BaseModel):
    concepts: list[TutorContextConcept] = Field(default_factory=list)
    claims: list[TutorContextClaim] = Field(default_factory=list)
    relations: list[TutorContextRelation] = Field(default_factory=list)
    source_fragments: list[TutorContextSourceFragment] = Field(default_factory=list)
    evidence: list[TutorContextEvidence] = Field(default_factory=list)
    pedagogical_evidence: list[TutorContextPedagogicalEvidence] = Field(default_factory=list)


class TutorContextResponse(BaseModel):
    resolved_reference: TutorContextResolvedReference
    status: Literal["ok", "failed"]
    concepts: list[TutorContextConcept] = Field(default_factory=list)
    claims: list[TutorContextClaim] = Field(default_factory=list)
    relations: list[TutorContextRelation] = Field(default_factory=list)
    source_fragments: list[TutorContextSourceFragment] = Field(default_factory=list)
    evidence: list[TutorContextEvidence] = Field(default_factory=list)
    pedagogical_evidence: list[TutorContextPedagogicalEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    failure_reason: str | None = None


PedagogicalDimension = Literal["recognition", "recall", "explanation", "application"]
StudyMode = Literal["hybrid", "backlog", "recovery", "isolated"]
AdaptiveBlockPurpose = Literal["spaced_repetition_review", "new_content"]
AdaptiveQuestionType = Literal[
    "multiple_choice_single",
    "multiple_choice_multi",
    "true_false",
    "cloze",
    "open",
]
AdaptiveDifficulty = Literal["introductory", "intermediate", "advanced"]
AdaptiveBlockGoal = Literal[
    "diagnose_uncertain",
    "reinforce_weak",
    "confirm_improvement",
    "test_depth",
    "close_concept",
]
AdaptiveVerdict = Literal["correct", "partial_high", "partial_low", "incorrect", "unsupported"]
PedagogicalMasteryLabel = Literal["muy bajo", "bajo", "medio", "alto", "muy alto"]
PedagogicalTrendLabel = Literal["improving", "stable", "declining", "insufficient_data"]
PedagogicalStatus = Literal["ok", "sparse", "not_found"]

PEDAGOGICAL_RELATIONS = {"PREREQUISITE_FOR", "PART_OF", "IS_A"}


class PedagogicalDimensionState(BaseModel):
    score_0_to_100: float = Field(default=50.0, ge=0.0, le=100.0)
    last_evaluated_at: str | None = None


class PedagogicalDimensionStates(BaseModel):
    recognition: PedagogicalDimensionState = Field(default_factory=PedagogicalDimensionState)
    recall: PedagogicalDimensionState = Field(default_factory=PedagogicalDimensionState)
    explanation: PedagogicalDimensionState = Field(default_factory=PedagogicalDimensionState)
    application: PedagogicalDimensionState = Field(default_factory=PedagogicalDimensionState)

    def weakest_dimension(self) -> tuple[PedagogicalDimension, PedagogicalDimensionState]:
        pairs: list[tuple[PedagogicalDimension, PedagogicalDimensionState]] = [
            ("recognition", self.recognition),
            ("recall", self.recall),
            ("explanation", self.explanation),
            ("application", self.application),
        ]
        return min(pairs, key=lambda item: item[1].score_0_to_100)

    def average_score(self) -> float:
        values = [
            self.recognition.score_0_to_100,
            self.recall.score_0_to_100,
            self.explanation.score_0_to_100,
            self.application.score_0_to_100,
        ]
        return round(sum(values) / len(values), 2)


class PedagogicalRecentStats(BaseModel):
    recent_average: float = Field(ge=0.0, le=100.0)
    trend: PedagogicalTrendLabel
    deviation: float = Field(ge=0.0)
    last_evaluated_at: str | None = None


class PedagogicalRecalculationTrace(BaseModel):
    kind: str
    message: str
    concept_uid: str | None = None
    related_concept_uid: str | None = None
    domain: str | None = None


class PedagogicalEvaluationEvent(BaseModel):
    user_id: str = Field(min_length=1)
    concept_uid: str = Field(min_length=1)
    concept_name: str = ""
    domain: str = Field(min_length=1)
    score_0_to_100: float = Field(ge=0.0, le=100.0)
    recorded_at: str
    source: Literal["formal_evaluation"] = "formal_evaluation"


class PedagogicalConceptState(BaseModel):
    user_id: str = Field(min_length=1)
    concept_uid: str = Field(min_length=1)
    concept_name: str = ""
    domain: str = Field(min_length=1)
    mastery_score_0_to_100: float = Field(ge=0.0, le=100.0)
    mastery_label: PedagogicalMasteryLabel
    dimensions: PedagogicalDimensionStates = Field(default_factory=PedagogicalDimensionStates)
    confidence_0_to_1: float = Field(default=0.25, ge=0.0, le=1.0)
    trend: PedagogicalTrendLabel = "insufficient_data"
    priority_score: float = Field(default=0.5, ge=0.0, le=1.0)
    last_block_id: str | None = None
    recent_history: list[PedagogicalEvaluationEvent] = Field(default_factory=list, max_length=5)
    recent_stats: PedagogicalRecentStats
    weaknesses: list[str] = Field(default_factory=list)
    detected_gaps: list[str] = Field(default_factory=list)
    suggested_questions: list[str] = Field(default_factory=list)
    effective_depth_used: int = Field(default=3, ge=1, le=10)
    last_evaluated_at: str | None = None
    updated_at: str
    recalculation_traces: list[PedagogicalRecalculationTrace] = Field(default_factory=list)


class PedagogicalDomainState(BaseModel):
    user_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    mastery_score_0_to_100: float = Field(ge=0.0, le=100.0)
    mastery_label: PedagogicalMasteryLabel
    concept_count: int = Field(ge=0)
    weak_concept_uids: list[str] = Field(default_factory=list)
    recent_stats: PedagogicalRecentStats
    updated_at: str
    recalculation_traces: list[PedagogicalRecalculationTrace] = Field(default_factory=list)


class PedagogicalContextSnapshot(BaseModel):
    user_id: str = Field(min_length=1)
    status: PedagogicalStatus
    concepts: list[PedagogicalConceptState] = Field(default_factory=list)
    domains: list[PedagogicalDomainState] = Field(default_factory=list)
    recent_evaluations: list[PedagogicalEvaluationEvent] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class GetPedagogicalContextRequest(BaseModel):
    user_id: str = Field(min_length=1)
    domain: str | None = None
    concept_uids: list[str] = Field(default_factory=list)


class GetPedagogicalContextResponse(PedagogicalContextSnapshot):
    pass


class PedagogicalEvaluationInput(BaseModel):
    concept_uid: str = Field(min_length=1)
    score_0_to_100: float = Field(ge=0.0, le=100.0)
    recorded_at: str | None = None


class UpdatePedagogicalContextRequest(BaseModel):
    user_id: str = Field(min_length=1)
    domain_hint: str | None = None
    evaluations: list[PedagogicalEvaluationInput] = Field(min_length=1)
    session_closed_at: str | None = None


class PedagogicalSessionFocusItem(BaseModel):
    concept_uid: str
    concept_name: str = ""
    domain: str
    mastery_score_0_to_100: float = Field(ge=0.0, le=100.0)
    mastery_label: PedagogicalMasteryLabel
    reason: str


class PedagogicalSessionView(BaseModel):
    user_id: str = Field(min_length=1)
    status: PedagogicalStatus
    summary: str
    weak_concepts: list[PedagogicalSessionFocusItem] = Field(default_factory=list)
    detected_gaps: list[str] = Field(default_factory=list)
    suggested_questions: list[str] = Field(default_factory=list)
    effective_depth_used: int = Field(default=3, ge=1, le=10)
    domain_focus: list[str] = Field(default_factory=list)
    recalculation_traces: list[PedagogicalRecalculationTrace] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PedagogicalSessionViewRequest(BaseModel):
    user_id: str = Field(min_length=1)
    domain_hint: str | None = None
    concept_uids: list[str] = Field(default_factory=list)
    query: str | None = Field(default=None, min_length=1)


class PedagogicalSessionViewResponse(PedagogicalSessionView):
    pass


class UpdatePedagogicalContextResponse(BaseModel):
    user_id: str = Field(min_length=1)
    status: PedagogicalStatus
    context: PedagogicalContextSnapshot
    session_view: PedagogicalSessionView
    warnings: list[str] = Field(default_factory=list)


class AdaptiveSessionConstraints(BaseModel):
    max_items_per_block: int = Field(default=3, ge=2, le=4)
    max_blocks: int = Field(default=4, ge=1, le=12)
    allowed_question_types: list[AdaptiveQuestionType] = Field(
        default_factory=lambda: [
            "multiple_choice_single",
            "multiple_choice_multi",
            "true_false",
            "cloze",
            "open",
        ]
    )
    preferred_max_difficulty: AdaptiveDifficulty | None = None
    allow_scaffolding: bool = True


class AdaptiveScaffoldingPolicy(BaseModel):
    allow_hint_after_error: bool = False
    allow_rephrase_retry: bool = False
    allow_difficulty_drop_next_item: bool = False
    show_corrective_explanation_at_end: bool = True


class AdaptiveSuccessCriteria(BaseModel):
    min_block_score: float = Field(default=0.7, ge=0.0, le=1.0)
    min_dimension_signal: float = Field(default=0.65, ge=0.0, le=1.0)
    max_supported_answers_ratio: float = Field(default=0.5, ge=0.0, le=1.0)


class AdaptiveNextStepPolicy(BaseModel):
    on_success: str
    on_partial_high: str
    on_partial_low: str
    on_failure: str


class SpacedRepetitionState(BaseModel):
    user_id: str = Field(min_length=1)
    concept_uid: str = Field(min_length=1)
    dimension: PedagogicalDimension
    repetitions: int = Field(default=0, ge=0)
    ease_factor: float = Field(default=2.5, ge=1.3)
    interval_days: int = Field(default=0, ge=0)
    last_reviewed_at: str | None = None
    next_review_at: str
    propagation_relief_count: int = Field(default=0, ge=0)
    requires_direct_validation: bool = False
    updated_at: str


class DueSRItem(BaseModel):
    concept_uid: str = Field(min_length=1)
    dimension: PedagogicalDimension
    next_review_at: str
    days_overdue: int = Field(ge=0)
    concept_name: str = ""
    domain: str = ""
    ease_factor: float = Field(default=2.5, ge=1.3)
    interval_days: int = Field(default=0, ge=0)
    mastery_score_0_to_100: float = Field(default=50.0, ge=0.0, le=100.0)
    confidence_0_to_1: float = Field(default=0.25, ge=0.0, le=1.0)
    requires_direct_validation: bool = False


class SRFeedback(BaseModel):
    concept_uid: str = Field(min_length=1)
    dimension: PedagogicalDimension
    calculated_quality_q: int = Field(ge=0, le=5)
    rationale: str


class SRStats(BaseModel):
    total_items: int = Field(default=0, ge=0)
    due_items: int = Field(default=0, ge=0)
    forced_review_items: int = Field(default=0, ge=0)
    average_ease_factor: float = Field(default=0.0, ge=0.0)


class GetSRStateRequest(BaseModel):
    user_id: str = Field(min_length=1)
    concept_uid: str = Field(min_length=1)
    dimension: PedagogicalDimension


class GetSRStateResponse(BaseModel):
    state: SpacedRepetitionState


class GetDueSRItemsResponse(BaseModel):
    user_id: str = Field(min_length=1)
    items: list[DueSRItem] = Field(default_factory=list)


class UpdateSRFromBlockRequest(BaseModel):
    user_id: str = Field(min_length=1)
    concept_uid: str = Field(min_length=1)
    dimension: PedagogicalDimension
    block_verdict: AdaptiveVerdict
    block_difficulty: AdaptiveDifficulty
    hint_used: bool = False
    retry_used: bool = False
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    was_direct_evaluation: bool = True


class UpdateSRFromBlockResponse(BaseModel):
    state: SpacedRepetitionState
    sr_feedback: SRFeedback
    ef_bonus_applied: bool = False


class ApplyPrereqReliefRequest(BaseModel):
    user_id: str = Field(min_length=1)
    source_concept_uid: str = Field(min_length=1)
    source_dimension: PedagogicalDimension
    quality_q: int = Field(ge=0, le=5)


class ApplyPrereqReliefResponse(BaseModel):
    updated_states: list[SpacedRepetitionState] = Field(default_factory=list)


class GetSRStatsResponse(BaseModel):
    user_id: str = Field(min_length=1)
    stats: SRStats


class AdaptiveBlockPlan(BaseModel):
    block_id: str
    block_purpose: AdaptiveBlockPurpose = "new_content"
    block_goal: AdaptiveBlockGoal
    target_concept_uid: str
    target_concept_name: str
    concept_domain: str = ""
    target_dimensions: list[PedagogicalDimension] = Field(min_length=1, max_length=2)
    recommended_question_types: list[AdaptiveQuestionType] = Field(min_length=1, max_length=3)
    difficulty: AdaptiveDifficulty
    due_item: DueSRItem | None = None
    corrective_explanation_seed: str = ""
    scaffolding: AdaptiveScaffoldingPolicy
    success_criteria: AdaptiveSuccessCriteria
    next_step_policy: AdaptiveNextStepPolicy
    planner_explanation: str


class AdaptiveBlockItem(BaseModel):
    item_id: str
    question_type: AdaptiveQuestionType
    concept_uid: str
    target_dimension: PedagogicalDimension
    difficulty: AdaptiveDifficulty
    prompt: str
    choices: list[str] = Field(default_factory=list)
    rubric: dict[str, Any] = Field(default_factory=dict)
    grounding_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdaptiveBlockAnswerKey(BaseModel):
    item_id: str
    grading_mode: str
    expected: list[str] = Field(default_factory=list)
    correct_choice_indexes: list[int] = Field(default_factory=list)
    boolean_answer: bool | None = None
    rationale: str = ""
    evidence_unit_ids: list[str] = Field(default_factory=list)
    grading_rubric: dict[str, Any] = Field(default_factory=dict)
    question_contract_version: str = "pedagogical-v2"
    validator_notes: list[str] = Field(default_factory=list)


class AdaptiveBlockResponse(BaseModel):
    block_id: str
    plan: AdaptiveBlockPlan
    items: list[AdaptiveBlockItem] = Field(default_factory=list)
    answer_keys: list[AdaptiveBlockAnswerKey] = Field(default_factory=list)
    generated_at: str


class AdaptiveInteractionEvent(BaseModel):
    item_id: str = Field(min_length=1)
    hint_used: bool = False
    retry_used: bool = False
    difficulty_drop_applied: bool = False


class AdaptiveItemSubmission(BaseModel):
    item_id: str = Field(min_length=1)
    response_text: str | None = None
    selected_choices: list[int] = Field(default_factory=list)
    boolean_answer: bool | None = None


class AdaptiveItemResult(BaseModel):
    item_id: str
    verdict: AdaptiveVerdict
    score_0_to_1: float = Field(ge=0.0, le=1.0)
    target_dimension: PedagogicalDimension
    used_hint: bool = False
    used_retry: bool = False
    feedback: str
    signals: dict[str, float] = Field(default_factory=dict)


class AdaptiveBlockResult(BaseModel):
    block_id: str
    item_results: list[AdaptiveItemResult] = Field(default_factory=list)
    dimension_summary: dict[PedagogicalDimension, float] = Field(default_factory=dict)
    block_verdict: AdaptiveVerdict
    block_score: float = Field(ge=0.0, le=1.0)
    sr_feedback: SRFeedback | None = None
    recommended_next_action: str
    corrective_explanation: str = ""
    transition_explanation: str = ""


class AdaptiveSessionSummary(BaseModel):
    total_blocks: int = 0
    completed_blocks: int = 0
    review_blocks_target: int = 0
    new_blocks_target: int = 0
    review_blocks_completed: int = 0
    new_blocks_completed: int = 0
    latest_block_verdict: AdaptiveVerdict | None = None
    session_closed: bool = False
    closure_reason: str | None = None


class AdaptiveSessionSnapshot(BaseModel):
    session_id: str
    user_id: str
    study_mode: StudyMode = "hybrid"
    resolved_reference: TutorContextResolvedReference
    domain_hint: str | None = None
    language: str = "es"
    constraints: AdaptiveSessionConstraints = Field(default_factory=AdaptiveSessionConstraints)
    tutor_context: TutorContextResponse
    current_block: AdaptiveBlockResponse | None = None
    block_history: list[AdaptiveBlockResult] = Field(default_factory=list)
    summary: AdaptiveSessionSummary = Field(default_factory=AdaptiveSessionSummary)
    status: Literal["active", "closed"] = "active"
    opened_at: str
    updated_at: str


class AdaptiveSessionStartRequest(BaseModel):
    user_id: str = Field(min_length=1)
    query: str | None = Field(default=None, min_length=1)
    episode_id: str | None = Field(default=None, min_length=1)
    job_id: str | None = Field(default=None, min_length=1)
    study_mode: StudyMode = "hybrid"
    domain_hint: str | None = None
    language: str = "es"
    constraints: AdaptiveSessionConstraints = Field(default_factory=AdaptiveSessionConstraints)

    @model_validator(mode="after")
    def validate_single_reference(self) -> "AdaptiveSessionStartRequest":
        provided = [
            name
            for name, value in (
                ("query", self.query),
                ("episode_id", self.episode_id),
                ("job_id", self.job_id),
            )
            if value
        ]
        if len(provided) != 1:
            raise ValueError("exactly one of query, episode_id or job_id must be provided")
        if self.study_mode == "isolated" and not self.domain_hint:
            raise ValueError("domain_hint is required when study_mode is isolated")
        return self


class AdaptiveSessionStartResponse(BaseModel):
    session: AdaptiveSessionSnapshot
    current_block: AdaptiveBlockResponse
    planner_explanation: str
    grounding_status: Literal["ok"]


class AdaptiveBlockSubmissionRequest(BaseModel):
    block_id: str = Field(min_length=1)
    submissions: list[AdaptiveItemSubmission] = Field(min_length=1)
    interaction_events: list[AdaptiveInteractionEvent] = Field(default_factory=list)


class AdaptiveBlockSubmissionResponse(BaseModel):
    session: AdaptiveSessionSnapshot
    block_result: AdaptiveBlockResult
    updated_context: PedagogicalContextSnapshot
    next_action: str
    next_block: AdaptiveBlockResponse | None = None
    session_closed: bool = False


class AgentToolDebug(BaseModel):
    retrieval_status: Literal["ok", "sparse", "no_match"]
    used_neighborhood: bool = False
    generation_mode: Literal["structured_llm", "stub", "skipped"] = "skipped"
    source_concept_uids: list[str] = Field(default_factory=list)
    source_claim_uids: list[str] = Field(default_factory=list)


class ExplainTopicResponse(BaseModel):
    query: str
    domain_hint: str | None = None
    status: Literal["ok", "sparse", "no_match", "blocked"]
    explanation_markdown: str
    key_points: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    source_concept_uids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    debug: AgentToolDebug
    failure_reason: str | None = None


class QuizQuestion(BaseModel):
    id: str
    type: Literal["multiple_choice", "open"]
    prompt: str
    choices: list[str] = Field(default_factory=list)


class QuizAnswerKeyItem(BaseModel):
    question_id: str
    correct_answer: str
    rationale: str
    evidence_unit_ids: list[str] = Field(default_factory=list)
    grading_rubric: dict[str, Any] = Field(default_factory=dict)
    question_contract_version: str = "pedagogical-v2"
    validator_notes: list[str] = Field(default_factory=list)


class GenerateQuizResponse(BaseModel):
    query: str
    domain_hint: str | None = None
    status: Literal["ok", "sparse", "no_match", "blocked"]
    questions: list[QuizQuestion] = Field(default_factory=list)
    answer_key: list[QuizAnswerKeyItem] = Field(default_factory=list)
    coverage_summary: str
    warnings: list[str] = Field(default_factory=list)
    debug: AgentToolDebug
    failure_reason: str | None = None


class EvaluateAnswerResponse(BaseModel):
    query: str
    status: Literal["ok", "sparse", "no_match", "blocked"]
    verdict: Literal["correct", "partial", "incorrect", "unsupported"]
    score_0_to_1: float = Field(ge=0.0, le=1.0)
    feedback_markdown: str
    matched_points: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    debug: AgentToolDebug
    failure_reason: str | None = None


class EpisodeResponse(BaseModel):
    uid: str
    text: str
    source_type: str
    tags: list[str] = Field(default_factory=list)
    language: str
    status: str
    error_message: str | None = None
    created_at: str


class IngestionSummary(BaseModel):
    episode_id: str
    domain: str
    created_concepts: list[str] = Field(default_factory=list)
    updated_concepts: list[str] = Field(default_factory=list)
    created_claims: int = 0
    relations: list[list[str]] = Field(default_factory=list)
    needs_review: list[str] = Field(default_factory=list)
    partial_errors: list[str] = Field(default_factory=list)


class JobStatusResponse(BaseModel):
    uid: str
    episode_id: str
    status: str
    result: IngestionSummary | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class NeighborhoodNode(BaseModel):
    uid: str
    type: str
    name: str
    domain: str | None = None
    description: str | None = None


class NeighborhoodRelation(BaseModel):
    from_uid: str
    from_name: str
    relation: str
    to_uid: str
    to_name: str
    confidence: float | None = None
    evidence_episode_id: str | None = None


class NeighborhoodResponse(BaseModel):
    concept: NeighborhoodNode
    nodes: list[NeighborhoodNode]
    relations: list[NeighborhoodRelation]
    claims: list[dict[str, Any]]
    episodes: list[dict[str, Any]]


class ExtractedConcept(BaseModel):
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    evidence_quotes: list[str] = Field(default_factory=list)
    confidence: float = 0.8

    @field_validator("canonical_name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return value.strip()

    @field_validator("aliases", "evidence_quotes", mode="before")
    @classmethod
    def coerce_to_list(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value if isinstance(value, list) else []


class ExtractedClaim(BaseModel):
    text: str
    confidence: float = 0.8
    explains: list[str] = Field(default_factory=list)
    supporting_quote: str | None = None

    @field_validator("explains", mode="before")
    @classmethod
    def coerce_explains_to_list(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value if isinstance(value, list) else []


class ExtractedRelation(BaseModel):
    from_name: str
    relation: str
    to_name: str
    confidence: float = 0.8

    @field_validator("relation")
    @classmethod
    def normalize_relation(cls, value: str) -> str:
        relation = value.strip().upper()
        if relation not in ALLOWED_RELATIONS:
            raise ValueError(f"unsupported relation: {relation}")
        return relation


class ExtractionResult(BaseModel):
    domain: str
    topics: list[str] = Field(default_factory=list)
    concepts: list[ExtractedConcept] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)

    @field_validator("topics", mode="before")
    @classmethod
    def coerce_topics_to_list(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value if isinstance(value, list) else []


class PedagogicalEvidenceDecision(BaseModel):
    statement: str
    supporting_quote: str
    kind: Literal["claim", "definition", "relationship"] = "claim"
    status: Literal["approved", "rejected", "repairable"] = "approved"
    review_notes: list[str] = Field(default_factory=list)

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_kind(cls, value: Any) -> str:
        allowed = {"claim", "definition", "relationship"}
        if isinstance(value, str):
            v = value.lower().strip()
            if v in allowed:
                return v
            if "defin" in v:
                return "definition"
            if "relation" in v:
                return "relationship"
            return "claim"
        return value

    @field_validator("review_notes", mode="before")
    @classmethod
    def coerce_review_notes_to_list(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value if isinstance(value, list) else []


class ConceptRecord(BaseModel):
    uid: str
    canonical_name: str
    normalized_name: str
    domain: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)
    created_at: str
    updated_at: str


class EpisodeRecord(BaseModel):
    uid: str
    text: str
    source_type: str
    tags: list[str] = Field(default_factory=list)
    language: str
    status: str
    created_at: str
    error_message: str | None = None


class ClaimRecord(BaseModel):
    uid: str
    text: str
    normalized_text: str
    confidence: float
    status: str
    created_at: str
    embedding: list[float] = Field(default_factory=list)
    supporting_quote: str | None = None


class JobRecord(BaseModel):
    uid: str
    episode_id: str
    status: str
    result: IngestionSummary | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class ConceptResolution(BaseModel):
    strategy: Literal["created", "updated", "matched", "ambiguous", "rejected"]
    concept: ConceptRecord | None = None
    needs_review_reason: str | None = None


def concept_ref_to_normalized(value: str) -> str:
    return normalize_text(value)
