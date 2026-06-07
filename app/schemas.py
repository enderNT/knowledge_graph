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


class UpsertConceptRequest(BaseModel):
    canonical_name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    domain: str = Field(min_length=1)
    description: str = ""


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


class TutorContextBundle(BaseModel):
    concepts: list[TutorContextConcept] = Field(default_factory=list)
    claims: list[TutorContextClaim] = Field(default_factory=list)
    relations: list[TutorContextRelation] = Field(default_factory=list)
    source_fragments: list[TutorContextSourceFragment] = Field(default_factory=list)
    evidence: list[TutorContextEvidence] = Field(default_factory=list)


class TutorContextResponse(BaseModel):
    resolved_reference: TutorContextResolvedReference
    status: Literal["ok", "failed"]
    concepts: list[TutorContextConcept] = Field(default_factory=list)
    claims: list[TutorContextClaim] = Field(default_factory=list)
    relations: list[TutorContextRelation] = Field(default_factory=list)
    source_fragments: list[TutorContextSourceFragment] = Field(default_factory=list)
    evidence: list[TutorContextEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    failure_reason: str | None = None


PedagogicalMasteryLabel = Literal["muy bajo", "bajo", "medio", "alto", "muy alto"]
PedagogicalTrendLabel = Literal["improving", "stable", "declining", "insufficient_data"]
PedagogicalStatus = Literal["ok", "sparse", "not_found"]

PEDAGOGICAL_RELATIONS = {"PREREQUISITE_FOR", "PART_OF", "IS_A"}


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


class AgentToolDebug(BaseModel):
    retrieval_status: Literal["ok", "sparse", "no_match"]
    used_neighborhood: bool = False
    generation_mode: Literal["structured_llm", "stub", "skipped"] = "skipped"
    source_concept_uids: list[str] = Field(default_factory=list)
    source_claim_uids: list[str] = Field(default_factory=list)


class ExplainTopicResponse(BaseModel):
    query: str
    domain_hint: str | None = None
    status: Literal["ok", "sparse", "no_match"]
    explanation_markdown: str
    key_points: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    source_concept_uids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    debug: AgentToolDebug


class QuizQuestion(BaseModel):
    id: str
    type: Literal["multiple_choice", "open"]
    prompt: str
    choices: list[str] = Field(default_factory=list)


class QuizAnswerKeyItem(BaseModel):
    question_id: str
    correct_answer: str
    rationale: str


class GenerateQuizResponse(BaseModel):
    query: str
    domain_hint: str | None = None
    status: Literal["ok", "sparse", "no_match"]
    questions: list[QuizQuestion] = Field(default_factory=list)
    answer_key: list[QuizAnswerKeyItem] = Field(default_factory=list)
    coverage_summary: str
    warnings: list[str] = Field(default_factory=list)
    debug: AgentToolDebug


class EvaluateAnswerResponse(BaseModel):
    query: str
    status: Literal["ok", "sparse", "no_match"]
    verdict: Literal["correct", "partial", "incorrect", "unsupported"]
    score_0_to_1: float = Field(ge=0.0, le=1.0)
    feedback_markdown: str
    matched_points: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    debug: AgentToolDebug


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
    confidence: float = 0.8

    @field_validator("canonical_name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return value.strip()


class ExtractedClaim(BaseModel):
    text: str
    confidence: float = 0.8
    explains: list[str] = Field(default_factory=list)


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


class JobRecord(BaseModel):
    uid: str
    episode_id: str
    status: str
    result: IngestionSummary | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class ConceptResolution(BaseModel):
    strategy: Literal["created", "updated", "matched", "ambiguous"]
    concept: ConceptRecord | None = None
    needs_review_reason: str | None = None


def concept_ref_to_normalized(value: str) -> str:
    return normalize_text(value)
