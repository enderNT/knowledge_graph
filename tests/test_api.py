from __future__ import annotations


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
