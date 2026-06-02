from __future__ import annotations


def test_health_live_is_public(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "live"}


def test_business_endpoints_require_api_key(client):
    response = client.post("/v1/search/candidates", json={"query": "test"})
    assert response.status_code == 401


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
