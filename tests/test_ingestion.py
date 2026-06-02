from __future__ import annotations

import asyncio
import time

from app.ai_provider import StubAIProvider
from app.ingestion import IngestionService
from app.schemas import CandidateHit


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
            candidates=candidates,
        )
        assert resolution.strategy == "ambiguous"
        assert resolution.concept is None

    asyncio.run(run_assertion())
