from __future__ import annotations

import asyncio

from app.schemas import SpacedRepetitionState


def test_sr_endpoints_require_api_key(client):
    response = client.get("/v1/sr/state", params={"user_id": "user-1", "concept_uid": "cn_1", "dimension": "recall"})
    assert response.status_code == 401


def test_sr_endpoints_expose_state_due_update_relief_and_stats(client, auth_headers, store):
    state = client.get(
        "/v1/sr/state",
        headers=auth_headers,
        params={"user_id": "user-1", "concept_uid": "cn_1", "dimension": "recall"},
    )
    assert state.status_code == 200
    assert state.json()["state"]["ease_factor"] == 2.5

    update = client.post(
        "/v1/sr/update",
        headers=auth_headers,
        json={
            "user_id": "user-1",
            "concept_uid": "cn_1",
            "dimension": "recall",
            "block_verdict": "correct",
            "block_difficulty": "intermediate",
            "coverage": 1.0,
            "precision": 1.0,
            "was_direct_evaluation": True,
        },
    )
    assert update.status_code == 200
    assert update.json()["sr_feedback"]["calculated_quality_q"] == 5

    async def seed_due():
        await store.upsert_spaced_repetition_state(
            SpacedRepetitionState(
                user_id="user-1",
                concept_uid="cn_due",
                dimension="recognition",
                repetitions=1,
                ease_factor=2.5,
                interval_days=1,
                last_reviewed_at=None,
                next_review_at="2026-06-01T10:00:00+00:00",
                propagation_relief_count=0,
                requires_direct_validation=False,
                updated_at="2026-06-06T10:00:00+00:00",
            )
        )

    asyncio.run(seed_due())

    due = client.get("/v1/sr/due", headers=auth_headers, params={"user_id": "user-1"})
    assert due.status_code == 200
    assert due.json()["user_id"] == "user-1"
    assert len(due.json()["items"]) >= 1

    relief = client.post(
        "/v1/sr/relief",
        headers=auth_headers,
        json={
            "user_id": "user-1",
            "source_concept_uid": "cn_source",
            "source_dimension": "recognition",
            "quality_q": 2,
        },
    )
    assert relief.status_code == 200
    assert relief.json()["updated_states"] == []

    stats = client.get("/v1/sr/stats", headers=auth_headers, params={"user_id": "user-1"})
    assert stats.status_code == 200
    assert stats.json()["stats"]["total_items"] >= 2
