from __future__ import annotations

import asyncio

from app.ai_provider import StubAIProvider
from app.schemas import UpsertConceptRequest


def _seed_concept(store, provider, *, canonical_name: str, domain: str, description: str, aliases: list[str]):
    async def run():
        embedding = await provider.embed(f"{canonical_name}\n{description}")
        concept, _ = await store.upsert_concept(
            UpsertConceptRequest(
                canonical_name=canonical_name,
                aliases=aliases,
                domain=domain,
                description=description,
            ),
            embedding=embedding,
            source_confidence=1.0,
        )
        return concept

    return asyncio.run(run())


def _seed_claim_and_episode(store, provider, *, claim_text: str, concept_uids: list[str], episode_text: str):
    async def run():
        episode = await store.create_episode(
            uid="ep_learning_context",
            text=episode_text,
            source_type="manual_input",
            tags=["Psicología"],
            language="es",
        )
        await store.update_episode(episode.uid, status="processed")
        claim = await store.create_claim(
            text=claim_text,
            confidence=0.92,
            status="approved",
            embedding=await provider.embed(claim_text),
            supporting_quote=claim_text,
        )
        for concept_uid in concept_uids:
            await store.link_claim_to_concept(claim.uid, concept_uid, confidence=0.92)
            await store.link_concept_to_episode(concept_uid, episode.uid, confidence=0.92)
            concept = await store.get_concept(concept_uid)
            if concept is not None:
                await store.create_pedagogical_evidence(
                    concept_uid=concept_uid,
                    concept_name=concept.canonical_name,
                    episode_id=episode.uid,
                    source_claim_uid=claim.uid,
                    statement=claim_text,
                    supporting_quote=claim_text,
                    kind="claim",
                    status="approved",
                )
        await store.link_claim_to_episode(claim.uid, episode.uid, confidence=0.92)
        return episode, claim

    return asyncio.run(run())


def _link_concepts(store, *, from_ref: str, relation: str, to_ref: str, evidence_episode_id: str | None = None):
    async def run():
        return await store.create_relation(
            from_ref=from_ref,
            relation=relation,
            to_ref=to_ref,
            evidence_episode_id=evidence_episode_id,
            confidence=0.95,
        )

    return asyncio.run(run())


def test_learning_context_returns_ok_with_structured_context(client, auth_headers, settings, store):
    provider = StubAIProvider(settings)
    primary = _seed_concept(
        store,
        provider,
        canonical_name="Memoria episódica",
        aliases=["recuerdo autobiográfico"],
        domain="Psicología",
        description="Sistema de memoria que conserva eventos personales situados en tiempo y contexto.",
    )
    related = _seed_concept(
        store,
        provider,
        canonical_name="Memoria semántica",
        aliases=[],
        domain="Psicología",
        description="Sistema de memoria que organiza hechos y conceptos desvinculados de una vivencia puntual.",
    )
    episode, claim = _seed_claim_and_episode(
        store,
        provider,
        claim_text="La memoria episódica permite recuperar experiencias personales con contexto temporal.",
        concept_uids=[primary.uid],
        episode_text="La memoria episódica organiza recuerdos autobiográficos y se distingue de la memoria semántica.",
    )
    assert _link_concepts(
        store,
        from_ref=primary.uid,
        relation="RELATED_TO",
        to_ref=related.uid,
        evidence_episode_id=episode.uid,
    )

    response = client.post(
        "/v1/search/learning-context",
        headers=auth_headers,
        json={"query": "recuerdo autobiográfico", "domain_hint": "Psicología"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["warnings"] == []
    assert body["primary_concepts"][0]["uid"] == primary.uid
    assert body["primary_concepts"][0]["retrieval_reason"] == "alias"
    assert body["claims"] == [
        {
            "uid": claim.uid,
            "text": "La memoria episódica permite recuperar experiencias personales con contexto temporal.",
            "confidence": 0.92,
        }
    ]
    assert body["episodes"] == [
        {
            "uid": episode.uid,
            "text": "La memoria episódica organiza recuerdos autobiográficos y se distingue de la memoria semántica.",
            "status": "processed",
        }
    ]
    assert any(item["relation"] == "RELATED_TO" for item in body["relations"])
    assert body["debug"]["selected_concept_uids"] == [primary.uid]


def test_learning_context_returns_sparse_for_weak_context(client, auth_headers, settings, store):
    provider = StubAIProvider(settings)
    concept = _seed_concept(
        store,
        provider,
        canonical_name="Memoria",
        aliases=[],
        domain="Psicología",
        description="",
    )

    response = client.post(
        "/v1/search/learning-context",
        headers=auth_headers,
        json={"query": "Memoria", "domain_hint": "Psicología"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "sparse"
    assert body["primary_concepts"][0]["uid"] == concept.uid
    assert set(body["primary_concepts"][0]["quality_flags"]) == {"generic_name", "weak_description"}
    assert "generic_primary_concepts" in body["warnings"]
    assert "weak_concept_descriptions" in body["warnings"]
    assert "no_neighbor_relations" in body["warnings"]
    assert "no_supporting_claims" in body["warnings"]
    assert "no_source_episodes" in body["warnings"]


def test_learning_context_returns_no_match_without_error(client, auth_headers):
    response = client.post(
        "/v1/search/learning-context",
        headers=auth_headers,
        json={"query": "tema inexistente", "domain_hint": "Psicología"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "tema inexistente",
        "domain_hint": "Psicología",
        "status": "no_match",
        "primary_concepts": [],
        "relations": [],
        "claims": [],
        "episodes": [],
        "warnings": ["no_candidates_found"],
        "debug": {
            "candidate_count": 0,
            "selected_concept_uids": [],
            "selection_reasons": [],
        },
    }


def test_learning_context_deduplicates_claims_and_episodes(client, auth_headers, settings, store):
    provider = StubAIProvider(settings)
    first = _seed_concept(
        store,
        provider,
        canonical_name="Memoria episódica",
        aliases=[],
        domain="Psicología",
        description="Sistema de memoria de experiencias vividas con contexto temporal y espacial.",
    )
    second = _seed_concept(
        store,
        provider,
        canonical_name="Memoria semántica",
        aliases=[],
        domain="Psicología",
        description="Sistema de memoria de hechos y conceptos generales estabilizados por aprendizaje.",
    )
    episode, claim = _seed_claim_and_episode(
        store,
        provider,
        claim_text="Ambos sistemas participan en la recuperación y organización del conocimiento declarativo.",
        concept_uids=[first.uid, second.uid],
        episode_text="La memoria episódica y la memoria semántica forman parte del conocimiento declarativo.",
    )
    assert _link_concepts(
        store,
        from_ref=first.uid,
        relation="RELATED_TO",
        to_ref=second.uid,
        evidence_episode_id=episode.uid,
    )

    response = client.post(
        "/v1/search/learning-context",
        headers=auth_headers,
        json={"query": "memoria", "domain_hint": "Psicología", "concept_limit": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "sparse"
    assert len(body["primary_concepts"]) == 2
    assert len(body["claims"]) == 1
    assert body["claims"][0]["uid"] == claim.uid
    assert len(body["episodes"]) == 1
    assert body["episodes"][0]["uid"] == episode.uid


def test_tutor_context_request_rejects_ambiguous_input(client, auth_headers):
    response = client.post(
        "/v1/search/tutor-context",
        headers=auth_headers,
        json={"query": "memoria", "episode_id": "ep_1"},
    )

    assert response.status_code == 422
    assert "exactly one of query, episode_id or job_id must be provided" in response.text


def test_tutor_context_by_episode_returns_strict_traceable_payload(client, auth_headers, settings, store):
    provider = StubAIProvider(settings)
    primary = _seed_concept(
        store,
        provider,
        canonical_name="Memoria episódica",
        aliases=["recuerdo autobiográfico"],
        domain="Psicología",
        description="Sistema de memoria que conserva eventos personales situados en tiempo y contexto.",
    )
    related = _seed_concept(
        store,
        provider,
        canonical_name="Memoria semántica",
        aliases=[],
        domain="Psicología",
        description="Sistema de memoria que organiza hechos y conceptos desvinculados de una vivencia puntual.",
    )
    episode, claim = _seed_claim_and_episode(
        store,
        provider,
        claim_text="La memoria episódica permite recuperar experiencias personales con contexto temporal.",
        concept_uids=[primary.uid],
        episode_text="La memoria episódica organiza recuerdos autobiográficos y se distingue de la memoria semántica.",
    )
    assert _link_concepts(
        store,
        from_ref=primary.uid,
        relation="RELATED_TO",
        to_ref=related.uid,
        evidence_episode_id=episode.uid,
    )

    response = client.post(
        "/v1/search/tutor-context",
        headers=auth_headers,
        json={"episode_id": episode.uid},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["resolved_reference"]["resolved_episode_id"] == episode.uid
    assert [item["uid"] for item in body["concepts"]] == [primary.uid, related.uid]
    assert body["claims"][0]["uid"] == claim.uid
    assert body["claims"][0]["evidence_episode_ids"] == [episode.uid]
    assert body["relations"][0]["evidence_episode_ids"] == [episode.uid]
    assert body["source_fragments"][0]["episode_id"] == episode.uid
    assert any(item["subject_type"] == "claim" for item in body["evidence"])


def test_tutor_context_by_query_returns_failure_without_traceable_evidence(client, auth_headers, settings, store):
    provider = StubAIProvider(settings)
    _seed_concept(
        store,
        provider,
        canonical_name="Memoria",
        aliases=[],
        domain="Psicología",
        description="Sistema cognitivo general.",
    )

    response = client.post(
        "/v1/search/tutor-context",
        headers=auth_headers,
        json={"query": "Memoria"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "resolved_reference": {
            "input_type": "query",
            "input_value": "Memoria",
            "resolved_concept_uid": next(iter(store.concepts.values())).uid,
            "resolved_concept_name": "Memoria",
            "resolved_episode_id": None,
            "resolved_job_id": None,
            "resolution_reason": "normalized_name",
        },
        "status": "failed",
        "concepts": [],
        "claims": [],
        "relations": [],
        "source_fragments": [],
        "evidence": [],
        "pedagogical_evidence": [],
        "warnings": ["no_source_fragments", "no_pedagogical_evidence", "no_evidence_links"],
        "failure_reason": "insufficient_traceable_evidence",
    }
