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


def test_episode_deletion_preview_and_execute_remove_derived_content(client, auth_headers):
    primary = client.post(
        "/v1/concepts",
        headers=auth_headers,
        json={
            "canonical_name": "Memoria episódica",
            "aliases": ["recuerdo autobiográfico"],
            "domain": "Psicología",
            "description": "Recupera experiencias personales.",
        },
    ).json()["concept"]["uid"]
    related = client.post(
        "/v1/concepts",
        headers=auth_headers,
        json={
            "canonical_name": "Memoria semántica",
            "aliases": [],
            "domain": "Psicología",
            "description": "Recupera conocimiento general.",
        },
    ).json()["concept"]["uid"]

    accepted = client.post(
        "/v1/knowledge/fragments",
        headers=auth_headers,
        json={
            "text": "La memoria episódica recupera experiencias personales con contexto temporal.",
            "source_type": "manual_input",
            "tags": ["Psicología"],
            "language": "es",
        },
    ).json()
    _wait_for_job_completion(client, auth_headers, accepted["job_id"])

    attach = client.post(
        "/v1/concepts/evidence",
        headers=auth_headers,
        json={
            "concept_ref": primary,
            "episode_id": accepted["episode_id"],
            "link_episode_claims": True,
        },
    )
    assert attach.status_code == 200

    link = client.post(
        "/v1/concepts/link",
        headers=auth_headers,
        json={
            "from": primary,
            "relation": "RELATED_TO",
            "to": related,
            "evidence_episode_id": accepted["episode_id"],
        },
    )
    assert link.status_code == 200

    preview = client.post(
        "/v1/deletions/episode/preview",
        headers=auth_headers,
        json={"episode_id": accepted["episode_id"]},
    )
    assert preview.status_code == 200
    body = preview.json()
    assert body["can_execute"] is True
    assert body["resolved_reference"]["resolved_job_ids"] == [accepted["job_id"]]
    assert body["impact"]["counts"]["episodes"] == 1
    assert body["impact"]["counts"]["jobs"] == 1
    assert body["impact"]["counts"]["pedagogical_evidence"] >= 1
    assert body["impact"]["counts"]["relations"] == 1
    assert body["impact"]["counts"]["claim_support_links"] >= 1

    execute = client.post(
        "/v1/deletions/episode/execute",
        headers=auth_headers,
        json={"episode_id": accepted["episode_id"], "confirm": True},
    )
    assert execute.status_code == 200
    assert execute.json()["status"] == "deleted"

    episode = client.get(f"/v1/episodes/{accepted['episode_id']}", headers=auth_headers)
    job = client.get(f"/v1/jobs/{accepted['job_id']}", headers=auth_headers)
    assert episode.status_code == 404
    assert job.status_code == 404

    neighborhood = client.get(f"/v1/concepts/{primary}/neighborhood?depth=1", headers=auth_headers)
    assert neighborhood.status_code == 200
    assert neighborhood.json()["relations"] == []
    assert neighborhood.json()["episodes"] == []


def test_relation_deletion_handles_ambiguous_and_scoped_matches(client, auth_headers):
    left = client.post(
        "/v1/concepts",
        headers=auth_headers,
        json={
            "canonical_name": "Condicionamiento clásico",
            "aliases": [],
            "domain": "Psicología",
            "description": "Aprendizaje asociativo.",
        },
    ).json()["concept"]["uid"]
    right = client.post(
        "/v1/concepts",
        headers=auth_headers,
        json={
            "canonical_name": "Condicionamiento operante",
            "aliases": [],
            "domain": "Psicología",
            "description": "Aprendizaje por consecuencias.",
        },
    ).json()["concept"]["uid"]

    first = client.post(
        "/v1/knowledge/fragments",
        headers=auth_headers,
        json={
            "text": "Primer episodio sobre condicionamiento.",
            "source_type": "manual_input",
            "tags": ["Psicología"],
            "language": "es",
        },
    ).json()
    second = client.post(
        "/v1/knowledge/fragments",
        headers=auth_headers,
        json={
            "text": "Segundo episodio sobre condicionamiento.",
            "source_type": "manual_input",
            "tags": ["Psicología"],
            "language": "es",
        },
    ).json()
    _wait_for_job_completion(client, auth_headers, first["job_id"])
    _wait_for_job_completion(client, auth_headers, second["job_id"])

    assert client.post(
        "/v1/concepts/link",
        headers=auth_headers,
        json={"from": left, "relation": "CONTRASTS_WITH", "to": right, "evidence_episode_id": first["episode_id"]},
    ).status_code == 200
    assert client.post(
        "/v1/concepts/link",
        headers=auth_headers,
        json={"from": left, "relation": "CONTRASTS_WITH", "to": right, "evidence_episode_id": second["episode_id"]},
    ).status_code == 200

    preview = client.post(
        "/v1/deletions/relation/preview",
        headers=auth_headers,
        json={"from": left, "relation": "CONTRASTS_WITH", "to": right},
    )
    assert preview.status_code == 200
    body = preview.json()
    assert body["can_execute"] is False
    assert sorted(body["resolved_reference"]["matched_evidence_episode_ids"]) == sorted(
        [first["episode_id"], second["episode_id"]]
    )

    ambiguous_execute = client.post(
        "/v1/deletions/relation/execute",
        headers=auth_headers,
        json={"from": left, "relation": "CONTRASTS_WITH", "to": right, "confirm": True},
    )
    assert ambiguous_execute.status_code == 409

    scoped_execute = client.post(
        "/v1/deletions/relation/execute",
        headers=auth_headers,
        json={
            "from": left,
            "relation": "CONTRASTS_WITH",
            "to": right,
            "evidence_episode_id": first["episode_id"],
            "confirm": True,
        },
    )
    assert scoped_execute.status_code == 200
    assert scoped_execute.json()["impact"]["counts"]["relations"] == 1

    all_execute = client.post(
        "/v1/deletions/relation/execute",
        headers=auth_headers,
        json={
            "from": left,
            "relation": "CONTRASTS_WITH",
            "to": right,
            "delete_all_matching": True,
            "confirm": True,
        },
    )
    assert all_execute.status_code == 200
    assert all_execute.json()["status"] == "deleted"
