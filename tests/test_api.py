from __future__ import annotations

import time


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


def test_health_live_is_public(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "live"}


def test_business_endpoints_require_api_key(client):
    response = client.post("/v1/search/candidates", json={"query": "test"})
    assert response.status_code == 401
    pedagogical = client.post("/v1/pedagogical/context", json={"user_id": "test-user"})
    assert pedagogical.status_code == 401


def test_concept_upsert_and_alias_search(client, auth_headers):
    response = client.put(
        "/v1/concepts/upsert",
        headers=auth_headers,
        json={
            "canonical_name": "Condicionamiento clásico",
            "aliases": ["condicionamiento pavloviano"],
            "domain": "Psicología",
            "description": "Aprendizaje asociativo.",
        },
    )
    assert response.status_code == 200
    concept_uid = response.json()["concept"]["uid"]

    search = client.post(
        "/v1/search/candidates",
        headers=auth_headers,
        json={
            "query": "condicionamiento pavloviano",
            "domain_hint": "Psicología",
            "limit": 10,
        },
    )
    assert search.status_code == 200
    body = search.json()
    assert body["results"][0]["uid"] == concept_uid
    assert body["results"][0]["reason"] == "alias"


def test_create_concept_is_strict_on_alias_conflicts(client, auth_headers):
    seed = client.post(
        "/v1/concepts",
        headers=auth_headers,
        json={
            "canonical_name": "Formato",
            "aliases": ["decimal empaquetado"],
            "domain": "Programacion Cobol",
            "description": "Concepto generico.",
        },
    )
    assert seed.status_code == 200

    conflict = client.post(
        "/v1/concepts",
        headers=auth_headers,
        json={
            "canonical_name": "Formato Decimal Empaquetado",
            "aliases": ["decimal empaquetado"],
            "domain": "Programacion Cobol",
            "description": "Representacion compacta.",
        },
    )
    assert conflict.status_code == 409
    assert "decimal empaquetado" in conflict.json()["detail"]


def test_concept_upsert_by_uid_updates_normalized_name(client, auth_headers):
    created = client.post(
        "/v1/concepts",
        headers=auth_headers,
        json={
            "canonical_name": "Sistema Operativo",
            "aliases": [],
            "domain": "Tecnología",
            "description": "Software base.",
        },
    )
    assert created.status_code == 200
    concept_uid = created.json()["concept"]["uid"]

    updated = client.put(
        "/v1/concepts/upsert",
        headers=auth_headers,
        json={
            "uid": concept_uid,
            "canonical_name": "Sistema Binario",
            "aliases": ["sistema binario"],
            "domain": "Programacion Cobol",
            "description": "Base 2.",
        },
    )
    assert updated.status_code == 200
    body = updated.json()
    assert body["created"] is False
    assert body["concept"]["canonical_name"] == "Sistema Binario"
    assert body["concept"]["normalized_name"] == "sistema binario"

    search = client.post(
        "/v1/search/candidates",
        headers=auth_headers,
        json={
            "query": "Sistema Binario",
            "domain_hint": "Programacion Cobol",
            "limit": 5,
        },
    )
    assert search.status_code == 200
    assert search.json()["results"][0]["uid"] == concept_uid
    assert search.json()["results"][0]["reason"] == "normalized_name"


def test_concept_upsert_returns_404_for_unknown_uid(client, auth_headers):
    response = client.put(
        "/v1/concepts/upsert",
        headers=auth_headers,
        json={
            "uid": "cn_missing",
            "canonical_name": "Sistema Binario",
            "aliases": [],
            "domain": "Programacion Cobol",
            "description": "Base 2.",
        },
    )
    assert response.status_code == 404


def test_attach_concept_evidence_enables_tutor_context_for_curated_concept(client, auth_headers):
    concept_response = client.post(
        "/v1/concepts",
        headers=auth_headers,
        json={
            "canonical_name": "Punto Flotante de Precision Simple",
            "aliases": ["precision simple"],
            "domain": "Programacion Cobol",
            "description": "Representacion de numeros reales con precision simple.",
        },
    )
    assert concept_response.status_code == 200
    concept_uid = concept_response.json()["concept"]["uid"]

    accepted = client.post(
        "/v1/knowledge/fragments",
        headers=auth_headers,
        json={
            "text": "Punto flotante de precision simple en COBOL. El punto flotante de precision simple representa numeros reales con precision simple.",
            "source_type": "manual_input",
            "tags": ["Programacion COBOL Curado", "Punto Flotante de Precision Simple"],
            "language": "es",
        },
    ).json()
    _wait_for_job_completion(client, auth_headers, accepted["job_id"])

    attach = client.post(
        "/v1/concepts/evidence",
        headers=auth_headers,
        json={
            "concept_ref": concept_uid,
            "episode_id": accepted["episode_id"],
            "link_episode_claims": True,
        },
    )
    assert attach.status_code == 200
    assert attach.json()["linked_claim_count"] >= 1

    tutor = client.post(
        "/v1/search/tutor-context",
        headers=auth_headers,
        json={"query": "precision simple"},
    )
    assert tutor.status_code == 200
    body = tutor.json()
    assert body["status"] == "ok"
    assert body["resolved_reference"]["resolved_concept_uid"] == concept_uid
    assert body["concepts"][0]["canonical_name"] == "Punto Flotante de Precision Simple"
    assert accepted["episode_id"] in [fragment["episode_id"] for fragment in body["source_fragments"]]


def test_reset_knowledge_base_clears_all_persisted_state(client, auth_headers):
    created = client.post(
        "/v1/concepts",
        headers=auth_headers,
        json={
            "canonical_name": "Memoria de trabajo",
            "aliases": ["working memory"],
            "domain": "Psicología",
            "description": "Sistema temporal de mantenimiento.",
        },
    )
    assert created.status_code == 200

    before = client.post(
        "/v1/search/candidates",
        headers=auth_headers,
        json={"query": "working memory", "domain_hint": "Psicología"},
    )
    assert before.status_code == 200
    assert before.json()["results"]

    reset = client.post("/v1/knowledge/reset", headers=auth_headers)
    assert reset.status_code == 200
    assert reset.json()["status"] == "reset"

    after = client.post(
        "/v1/search/candidates",
        headers=auth_headers,
        json={"query": "working memory", "domain_hint": "Psicología"},
    )
    assert after.status_code == 200
    assert after.json()["results"] == []
