from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

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
