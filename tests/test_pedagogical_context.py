from __future__ import annotations

import asyncio

from app.ai_provider import StubAIProvider
from app.pedagogical_context import PedagogicalContextBuilder
from app.schemas import UpdatePedagogicalContextRequest, UpsertConceptRequest


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


def _link_concepts(store, *, from_ref: str, relation: str, to_ref: str):
    async def run():
        return await store.create_relation(
            from_ref=from_ref,
            relation=relation,
            to_ref=to_ref,
            evidence_episode_id=None,
            confidence=0.9,
        )

    return asyncio.run(run())


def test_pedagogical_context_returns_not_found_for_new_user(client, auth_headers):
    response = client.post(
        "/v1/pedagogical/context",
        headers=auth_headers,
        json={"user_id": "user-new"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": "user-new",
        "status": "not_found",
        "concepts": [],
        "domains": [],
        "recent_evaluations": [],
        "warnings": ["empty_user_context"],
    }


def test_pedagogical_update_persists_context_and_session_view(client, auth_headers, settings, store):
    provider = StubAIProvider(settings)
    concept = _seed_concept(
        store,
        provider,
        canonical_name="Condicionamiento clasico",
        aliases=[],
        domain="Psicologia",
        description="Aprendizaje asociativo entre estimulos.",
    )

    update = client.post(
        "/v1/pedagogical/update-from-evaluation",
        headers=auth_headers,
        json={
            "user_id": "user-1",
            "domain_hint": "Psicologia",
            "evaluations": [{"concept_uid": concept.uid, "score_0_to_100": 82, "recorded_at": "2026-06-06T10:00:00+00:00"}],
        },
    )

    assert update.status_code == 200
    body = update.json()
    assert body["status"] == "ok"
    assert body["context"]["concepts"][0]["concept_uid"] == concept.uid
    assert body["context"]["concepts"][0]["mastery_label"] == "muy alto"
    assert body["context"]["domains"][0]["domain"] == "Psicologia"
    assert body["session_view"]["status"] == "ok"
    assert body["session_view"]["effective_depth_used"] >= 4

    context = client.post(
        "/v1/pedagogical/context",
        headers=auth_headers,
        json={"user_id": "user-1", "domain": "Psicologia"},
    )
    assert context.status_code == 200
    assert context.json()["concepts"][0]["recent_history"][0]["score_0_to_100"] == 82.0


def test_pedagogical_update_amortizes_history_and_isolates_users(client, auth_headers, settings, store):
    provider = StubAIProvider(settings)
    concept = _seed_concept(
        store,
        provider,
        canonical_name="Memoria episodica",
        aliases=[],
        domain="Psicologia",
        description="Recuperacion de experiencias personales.",
    )

    first = client.post(
        "/v1/pedagogical/update-from-evaluation",
        headers=auth_headers,
        json={
            "user_id": "user-1",
            "evaluations": [{"concept_uid": concept.uid, "score_0_to_100": 90, "recorded_at": "2026-06-01T10:00:00+00:00"}],
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/pedagogical/update-from-evaluation",
        headers=auth_headers,
        json={
            "user_id": "user-1",
            "evaluations": [{"concept_uid": concept.uid, "score_0_to_100": 20, "recorded_at": "2026-06-02T10:00:00+00:00"}],
        },
    )
    assert second.status_code == 200
    second_score = second.json()["context"]["concepts"][0]["mastery_score_0_to_100"]
    assert 20.0 < second_score < 90.0

    other_user = client.post(
        "/v1/pedagogical/update-from-evaluation",
        headers=auth_headers,
        json={
            "user_id": "user-2",
            "evaluations": [{"concept_uid": concept.uid, "score_0_to_100": 55, "recorded_at": "2026-06-02T10:00:00+00:00"}],
        },
    )
    assert other_user.status_code == 200
    assert other_user.json()["context"]["concepts"][0]["mastery_score_0_to_100"] != second_score


def test_pedagogical_builder_applies_decay_and_propagation(settings, store):
    provider = StubAIProvider(settings)
    prereq = _seed_concept(
        store,
        provider,
        canonical_name="Neuronas",
        aliases=[],
        domain="Biologia",
        description="Unidad funcional del sistema nervioso.",
    )
    dependent = _seed_concept(
        store,
        provider,
        canonical_name="Sinapsis",
        aliases=[],
        domain="Biologia",
        description="Conexion funcional entre neuronas.",
    )
    assert _link_concepts(store, from_ref=prereq.uid, relation="PREREQUISITE_FOR", to_ref=dependent.uid)

    async def run():
        builder = PedagogicalContextBuilder(settings=settings, store=store, ai_provider=provider)
        response = await builder.apply_evaluation_results(
            UpdatePedagogicalContextRequest(
                user_id="user-3",
                domain_hint="Biologia",
                evaluations=[
                    {"concept_uid": prereq.uid, "score_0_to_100": 80, "recorded_at": "2026-05-20T10:00:00+00:00"}
                ],
            )
        )
        return response

    response = asyncio.run(run())
    concept_states = {item.concept_uid: item for item in response.context.concepts}
    assert concept_states[prereq.uid].mastery_score_0_to_100 < 80.0
    assert dependent.uid in concept_states
    assert concept_states[dependent.uid].recalculation_traces[-1].kind == "propagation"
