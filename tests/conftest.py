from __future__ import annotations

from typing import Callable

import pytest
from fastapi.testclient import TestClient

from app.ai_provider import AIProvider, StubAIProvider
from app.config import Settings
from app.main import create_app
from app.schemas import (
    ExtractedClaim,
    ExtractionResult,
    ExtractionVettingResult,
    GeneratedAdaptiveBlock,
    LearnerResponseGradingDecision,
    PedagogicalEvidenceDecision,
)
from app.store import InMemoryKnowledgeStore


class FakeExtractionProvider(AIProvider):
    """Test double for AIProvider: stubs embed/vet/grade/generate via StubAIProvider,
    implements extract() via an optional callable (defaults to empty extraction),
    and implements vet_extraction() as a pass-through keep-all."""

    def __init__(
        self,
        settings: Settings,
        extraction_fn: Callable[[str, str, list[str]], ExtractionResult] | None = None,
    ) -> None:
        self._stub = StubAIProvider(settings)
        self._extraction_fn = extraction_fn

    async def embed(self, text: str) -> list[float]:
        return await self._stub.embed(text)

    async def extract(self, text: str, language: str, tags: list[str]) -> ExtractionResult:
        if self._extraction_fn is not None:
            return self._extraction_fn(text, language, tags)
        domain = tags[0].strip().title() if tags else "General"
        # Produce one claim so tests that call /v1/concepts/evidence get linked_claim_count >= 1
        first_sentence = text.split(".")[0].strip()
        claims = []
        if len(first_sentence) >= 6:
            quote = first_sentence[:60]
            claims = [ExtractedClaim(
                text=first_sentence + ".",
                confidence=0.75,
                explains=[],
                supporting_quote=quote,
            )]
        return ExtractionResult(domain=domain, topics=[domain], claims=claims)

    async def vet_extraction(
        self,
        *,
        extraction: ExtractionResult,
        text: str,
        language: str,
    ) -> ExtractionVettingResult:
        return ExtractionVettingResult(
            concepts=extraction.concepts,
            claims=extraction.claims,
            relations=extraction.relations,
        )

    async def vet_pedagogical_evidence(
        self,
        *,
        concept_name: str,
        claim_text: str,
        supporting_quote: str,
        language: str,
    ) -> PedagogicalEvidenceDecision:
        return await self._stub.vet_pedagogical_evidence(
            concept_name=concept_name,
            claim_text=claim_text,
            supporting_quote=supporting_quote,
            language=language,
        )

    async def grade_learner_response(
        self,
        *,
        learner_response: str,
        expected_statement: str,
        pedagogical_evidence: list[str],
        concept_name: str,
        question_type: str,
        language: str,
    ) -> LearnerResponseGradingDecision:
        return await self._stub.grade_learner_response(
            learner_response=learner_response,
            expected_statement=expected_statement,
            pedagogical_evidence=pedagogical_evidence,
            concept_name=concept_name,
            question_type=question_type,
            language=language,
        )

    async def generate_adaptive_block(
        self,
        *,
        evidence_units: list[dict[str, str]],
        target_dimension: str,
        difficulty: str,
        question_types: list[str],
        served_evidence_uids: list[str],
        item_count: int,
        concept_name: str,
        language: str,
    ) -> GeneratedAdaptiveBlock:
        return await self._stub.generate_adaptive_block(
            evidence_units=evidence_units,
            target_dimension=target_dimension,
            difficulty=difficulty,
            question_types=question_types,
            served_evidence_uids=served_evidence_uids,
            item_count=item_count,
            concept_name=concept_name,
            language=language,
        )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        embedding_dimensions=16,
    )


@pytest.fixture
def store(settings: Settings) -> InMemoryKnowledgeStore:
    return InMemoryKnowledgeStore(settings)


@pytest.fixture
def extraction_fn() -> Callable[[str, str, list[str]], ExtractionResult] | None:
    return None


@pytest.fixture
def client(settings: Settings, store: InMemoryKnowledgeStore, extraction_fn) -> TestClient:
    app = create_app(
        settings=settings,
        store=store,
        ai_provider=FakeExtractionProvider(settings, extraction_fn=extraction_fn),
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(settings: Settings) -> dict[str, str]:
    return {"X-API-Key": settings.api_key}
