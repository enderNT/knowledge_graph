from __future__ import annotations

import asyncio
import time

import pytest

from app.ai_provider import StubAIProvider
from app.ingestion import IngestionService
from app.schemas import (
    AddKnowledgeFragmentRequest,
    CandidateHit,
    ExtractedConcept,
    ExtractedClaim,
    ExtractedRelation,
    ExtractionResult,
    ExtractionVettingResult,
    UpsertConceptRequest,
)


@pytest.fixture
def extraction_fn():
    """Module-level override: returns appropriate fake extractions for integration tests."""
    def _extract(text: str, language: str, tags: list[str]) -> ExtractionResult:
        domain = tags[0].strip().title() if tags else "General"
        if "Condicionamiento" in text and "contrasta" in text.lower():
            return ExtractionResult(
                domain="Psicología",
                topics=["Psicología"],
                concepts=[
                    ExtractedConcept(
                        canonical_name="Condicionamiento Clásico",
                        confidence=0.9,
                        evidence_quotes=["Condicionamiento Clásico contrasta con"],
                    ),
                    ExtractedConcept(
                        canonical_name="Condicionamiento Operante",
                        confidence=0.9,
                        evidence_quotes=["contrasta con Condicionamiento Operante"],
                    ),
                ],
                claims=[
                    ExtractedClaim(
                        text="El condicionamiento clásico contrasta con el condicionamiento operante.",
                        confidence=0.85,
                        explains=["Condicionamiento Clásico", "Condicionamiento Operante"],
                        supporting_quote="Condicionamiento Clásico contrasta con Condicionamiento Operante",
                    )
                ],
                relations=[
                    ExtractedRelation(
                        from_name="Condicionamiento Clásico",
                        relation="CONTRASTS_WITH",
                        to_name="Condicionamiento Operante",
                        confidence=0.88,
                    )
                ],
            )
        if "Condicionamiento" in text:
            return ExtractionResult(
                domain="Psicología",
                topics=["Psicología"],
                concepts=[
                    ExtractedConcept(
                        canonical_name="Condicionamiento Clásico",
                        confidence=0.9,
                        evidence_quotes=["condicionamiento clásico"],
                    ),
                ],
                claims=[
                    ExtractedClaim(
                        text=text.split(".")[0].strip() + ".",
                        confidence=0.8,
                        explains=["Condicionamiento Clásico"],
                        supporting_quote="condicionamiento clásico",
                    )
                ],
                relations=[],
            )
        if "Hardware" in text or "hardware" in text:
            return ExtractionResult(
                domain="Tecnología",
                topics=["Tecnología"],
                concepts=[
                    ExtractedConcept(
                        canonical_name="Hardware",
                        confidence=0.9,
                        evidence_quotes=["Hardware y software son componentes"],
                    ),
                    ExtractedConcept(
                        canonical_name="Software",
                        confidence=0.9,
                        evidence_quotes=["Hardware y software son componentes"],
                    ),
                    ExtractedConcept(
                        canonical_name="Conectividad",
                        confidence=0.88,
                        evidence_quotes=["conectividad permiten comunicar"],
                    ),
                ],
                claims=[
                    ExtractedClaim(
                        text="Hardware y software son componentes fundamentales.",
                        confidence=0.85,
                        explains=["Hardware", "Software"],
                        supporting_quote="Hardware y software son componentes fundamentales",
                    )
                ],
                relations=[],
            )
        return ExtractionResult(domain=domain, topics=[domain])
    return _extract


def _wait_for_job_completion(client, headers, job_id: str, timeout: float = 2.0):
    started = time.time()
    while time.time() - started < timeout:
        response = client.get(f"/v1/jobs/{job_id}", headers=headers)
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("job did not complete in time")


def test_ingestion_pipeline_end_to_end(client, auth_headers):
    response = client.post(
        "/v1/knowledge/fragments",
        headers=auth_headers,
        json={
            "text": "Condicionamiento Clásico contrasta con Condicionamiento Operante.",
            "source_type": "manual_input",
            "tags": ["Psicología"],
            "language": "es",
        },
    )
    assert response.status_code == 202
    accepted = response.json()

    job = _wait_for_job_completion(client, auth_headers, accepted["job_id"])
    assert job["status"] == "completed"
    assert job["result"]["episode_id"] == accepted["episode_id"]
    assert job["result"]["relations"]

    episode = client.get(f"/v1/episodes/{accepted['episode_id']}", headers=auth_headers)
    assert episode.status_code == 200
    assert episode.json()["status"] == "processed"

    tutor_by_episode = client.post(
        "/v1/search/tutor-context",
        headers=auth_headers,
        json={"episode_id": accepted["episode_id"]},
    )
    tutor_by_job = client.post(
        "/v1/search/tutor-context",
        headers=auth_headers,
        json={"job_id": accepted["job_id"]},
    )
    assert tutor_by_episode.status_code == 200
    assert tutor_by_job.status_code == 200
    assert tutor_by_episode.json()["status"] == "ok"
    assert tutor_by_job.json()["status"] == "ok"
    assert tutor_by_job.json()["resolved_reference"]["resolved_episode_id"] == accepted["episode_id"]
    assert tutor_by_job.json()["concepts"] == tutor_by_episode.json()["concepts"]
    assert tutor_by_job.json()["claims"] == tutor_by_episode.json()["claims"]


def test_ingestion_pipeline_persists_canonical_trace(client, auth_headers, store):
    response = client.post(
        "/v1/knowledge/fragments",
        headers=auth_headers,
        json={
            "text": "Condicionamiento Clásico contrasta con Condicionamiento Operante.",
            "source_type": "manual_input",
            "tags": ["Psicología"],
            "language": "es",
        },
    )
    accepted = response.json()
    job = _wait_for_job_completion(client, auth_headers, accepted["job_id"])

    traces = asyncio.run(store.list_canonical_traces(execution_id=accepted["job_id"]))
    assert job["status"] == "completed"
    assert len(traces) == 1
    trace = asyncio.run(store.get_canonical_trace(traces[0].trace_id))
    assert trace is not None
    assert trace.summary.execution_id == accepted["job_id"]
    assert trace.summary.episode_id == accepted["episode_id"]
    assert trace.summary.status in {"succeeded", "needs_review"}
    assert [event.type for event in trace.events] == [
        "fragment_received",
        "knowledge_extracted",
        "extraction_vetted",
        "ingestion_finalized",
    ]
    assert [event.sequence for event in trace.events] == [1, 2, 3, 4]
    assert trace.events[0].title == "Fragmento recibido"
    assert trace.summary.total_steps == 4


@pytest.mark.asyncio
async def test_failed_ingestion_persists_failed_canonical_trace(settings, store):
    class FailingProvider(StubAIProvider):
        async def extract(self, text: str, language: str, tags: list[str]) -> ExtractionResult:
            raise ValueError("boom")

        async def vet_extraction(
            self,
            *,
            extraction: ExtractionResult,
            text: str,
            language: str,
        ) -> ExtractionVettingResult:
            return ExtractionVettingResult()

    queue: asyncio.Queue[str] = asyncio.Queue()
    service = IngestionService(settings=settings, store=store, ai_provider=FailingProvider(settings), queue=queue)
    accepted = await service.submit_fragment(
        AddKnowledgeFragmentRequest(text="Texto que falla.", tags=["General"], language="es")
    )

    with pytest.raises(ValueError, match="boom"):
        await service.process_job(accepted.job_id)

    traces = await store.list_canonical_traces(execution_id=accepted.job_id)
    assert len(traces) == 1
    trace = await store.get_canonical_trace(traces[0].trace_id)
    assert trace is not None
    assert trace.summary.status == "failed"
    assert trace.events[-1].type == "ingestion_failed"
    assert trace.events[-1].status == "failed"


def test_neighborhood_exposes_claims_and_episode_evidence(client, auth_headers):
    first = client.put(
        "/v1/concepts/upsert",
        headers=auth_headers,
        json={
            "canonical_name": "Condicionamiento clásico",
            "aliases": [],
            "domain": "Psicología",
            "description": "Aprendizaje asociativo.",
        },
    )
    second = client.put(
        "/v1/concepts/upsert",
        headers=auth_headers,
        json={
            "canonical_name": "Condicionamiento operante",
            "aliases": [],
            "domain": "Psicología",
            "description": "Aprendizaje por consecuencias.",
        },
    )
    first_uid = first.json()["concept"]["uid"]
    second_uid = second.json()["concept"]["uid"]

    fragment = client.post(
        "/v1/knowledge/fragments",
        headers=auth_headers,
        json={
            "text": "Condicionamiento clásico explica un aprendizaje por asociación.",
            "source_type": "manual_input",
            "tags": ["Psicología"],
            "language": "es",
        },
    ).json()
    _wait_for_job_completion(client, auth_headers, fragment["job_id"])

    link = client.post(
        "/v1/concepts/link",
        headers=auth_headers,
        json={
            "from": first_uid,
            "relation": "CONTRASTS_WITH",
            "to": second_uid,
            "evidence_episode_id": fragment["episode_id"],
        },
    )
    assert link.status_code == 200

    neighborhood = client.get(
        f"/v1/concepts/{first_uid}/neighborhood?depth=2",
        headers=auth_headers,
    )
    assert neighborhood.status_code == 200
    body = neighborhood.json()
    assert any(item["relation"] == "CONTRASTS_WITH" for item in body["relations"])
    assert body["episodes"]


def test_neighborhood_depth_two_includes_second_hop(client, auth_headers):
    root = client.put(
        "/v1/concepts/upsert",
        headers=auth_headers,
        json={
            "canonical_name": "Hardware",
            "aliases": [],
            "domain": "Tecnología",
            "description": "Componentes físicos.",
        },
    ).json()["concept"]["uid"]
    middle = client.put(
        "/v1/concepts/upsert",
        headers=auth_headers,
        json={
            "canonical_name": "Software",
            "aliases": [],
            "domain": "Tecnología",
            "description": "Componentes lógicos.",
        },
    ).json()["concept"]["uid"]
    leaf = client.put(
        "/v1/concepts/upsert",
        headers=auth_headers,
        json={
            "canonical_name": "Conectividad",
            "aliases": [],
            "domain": "Tecnología",
            "description": "Comunicación entre dispositivos.",
        },
    ).json()["concept"]["uid"]

    assert client.post(
        "/v1/concepts/link",
        headers=auth_headers,
        json={"from": root, "relation": "RELATED_TO", "to": middle},
    ).status_code == 200
    assert client.post(
        "/v1/concepts/link",
        headers=auth_headers,
        json={"from": middle, "relation": "RELATED_TO", "to": leaf},
    ).status_code == 200

    depth_one = client.get(f"/v1/concepts/{root}/neighborhood?depth=1", headers=auth_headers)
    depth_two = client.get(f"/v1/concepts/{root}/neighborhood?depth=2", headers=auth_headers)

    assert depth_one.status_code == 200
    assert depth_two.status_code == 200
    assert {item["name"] for item in depth_one.json()["nodes"]} == {"Software"}
    assert {"Software", "Conectividad"}.issubset({item["name"] for item in depth_two.json()["nodes"]})


def test_ingestion_search_returns_canonical_technology_concepts(client, auth_headers):
    accepted = client.post(
        "/v1/knowledge/fragments",
        headers=auth_headers,
        json={
            "text": (
                "Hardware y software son componentes fundamentales. "
                "Redes y conectividad permiten comunicar dispositivos."
            ),
            "source_type": "manual_input",
            "tags": ["Tecnología"],
            "language": "es",
        },
    ).json()
    job = _wait_for_job_completion(client, auth_headers, accepted["job_id"])

    assert job["status"] == "completed"

    for query, expected in (
        ("hardware", "Hardware"),
        ("software", "Software"),
        ("conectividad", "Conectividad"),
    ):
        response = client.post(
            "/v1/search/candidates",
            headers=auth_headers,
            json={"query": query, "domain_hint": "Tecnología", "limit": 5},
        )
        assert response.status_code == 200
        assert response.json()["results"][0]["canonical_name"] == expected


def test_ambiguous_resolution_marks_needs_review(settings, store):
    provider = StubAIProvider(settings)
    service = IngestionService(
        settings=settings,
        store=store,
        ai_provider=provider,
        queue=asyncio.Queue(),
    )

    candidates = [
        CandidateHit(
            uid="cn_1",
            canonical_name="Extinción A",
            domain="Psicología",
            description="",
            score=0.95,
            reason="vector",
        ),
        CandidateHit(
            uid="cn_2",
            canonical_name="Extinción B",
            domain="Psicología",
            description="",
            score=0.94,
            reason="vector",
        ),
    ]

    async def run_assertion():
        resolution = await service._resolve_concept(
            canonical_name="Extinción",
            domain="Psicología",
            description="Proceso de reducción.",
            aliases=[],
            confidence=0.9,
            embedding=await provider.embed("Extinción"),
            evidence_quotes=["La extinción reduce una respuesta aprendida."],
            claim_support_count=1,
            candidates=candidates,
        )
        assert resolution.strategy == "ambiguous"
        assert resolution.concept is None

    asyncio.run(run_assertion())


def test_new_concept_without_evidence_or_claim_support_is_rejected(settings, store):
    provider = StubAIProvider(settings)
    service = IngestionService(
        settings=settings,
        store=store,
        ai_provider=provider,
        queue=asyncio.Queue(),
    )

    async def run_assertion():
        resolution = await service._resolve_concept(
            canonical_name="Punto Flotante de Precision Simple",
            domain="Programacion Cobol",
            description="",
            aliases=[],
            confidence=0.7,
            embedding=await provider.embed("Punto Flotante de Precision Simple"),
            evidence_quotes=[],
            claim_support_count=0,
            candidates=[],
        )
        assert resolution.strategy == "rejected"
        assert resolution.concept is None
        assert "missing claim support or traceable evidence" in (resolution.needs_review_reason or "")

    asyncio.run(run_assertion())


def test_conflicting_exact_alias_reuses_existing_concept_identity(settings, store):
    provider = StubAIProvider(settings)
    service = IngestionService(
        settings=settings,
        store=store,
        ai_provider=provider,
        queue=asyncio.Queue(),
    )

    async def run_assertion():
        seeded, _ = await store.upsert_concept(
            payload=UpsertConceptRequest(
                canonical_name="Representación Zonificada",
                aliases=["formato decimal zonificado", "decimal zonificado"],
                domain="Programacion Cobol",
                description="Nombre alterno previo.",
            ),
            embedding=await provider.embed("Representación Zonificada"),
            source_confidence=0.9,
        )
        resolution = await service._resolve_concept(
            canonical_name="Formato Decimal Zonificado",
            domain="Programacion Cobol",
            description="Representacion numerica con un digito por byte.",
            aliases=["decimal zonificado"],
            confidence=0.8,
            embedding=await provider.embed("Formato Decimal Zonificado"),
            evidence_quotes=["decimal zonificado (punto fijo)"],
            claim_support_count=1,
            candidates=[],
        )
        assert resolution.strategy == "updated"
        assert resolution.concept is not None
        assert resolution.concept.uid == seeded.uid

    asyncio.run(run_assertion())
