from __future__ import annotations

import asyncio

from app.adaptive_learning import AdaptiveLearningService
from app.ai_provider import StubAIProvider
from app.schemas import (
    AdaptiveBlockAnswerKey,
    AdaptiveBlockItem,
    AdaptiveInteractionEvent,
    AdaptiveItemSubmission,
    AdaptiveSessionStartRequest,
    PedagogicalConceptState,
    PedagogicalDimensionState,
    PedagogicalDimensionStates,
    PedagogicalRecentStats,
    UpsertConceptRequest,
)


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


def _seed_claim_and_episode(store, provider, *, claim_text: str, concept_uids: list[str], episode_text: str, episode_id: str):
    async def run():
        episode = await store.create_episode(
            uid=episode_id,
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
        )
        for concept_uid in concept_uids:
            await store.link_claim_to_concept(claim.uid, concept_uid, confidence=0.92)
            await store.link_concept_to_episode(concept_uid, episode.uid, confidence=0.92)
        await store.link_claim_to_episode(claim.uid, episode.uid, confidence=0.92)
        return episode, claim

    return asyncio.run(run())


def _make_service(settings, store) -> AdaptiveLearningService:
    return AdaptiveLearningService(settings=settings, store=store, ai_provider=StubAIProvider(settings))


def test_adaptive_endpoints_require_api_key(client):
    response = client.post("/v1/adaptive/sessions/start", json={"user_id": "user-1", "query": "memoria"})
    assert response.status_code == 401


def test_adaptive_session_start_supports_query_episode_and_job(client, auth_headers, settings, store):
    provider = StubAIProvider(settings)
    concept = _seed_concept(
        store,
        provider,
        canonical_name="Condicionamiento clásico",
        aliases=["condicionamiento pavloviano"],
        domain="Psicología",
        description="Aprendizaje asociativo con soporte experimental.",
    )
    episode, _ = _seed_claim_and_episode(
        store,
        provider,
        claim_text="El condicionamiento clásico asocia un estímulo condicionado con uno incondicionado.",
        concept_uids=[concept.uid],
        episode_text="Pavlov observó que un estímulo neutro podía anticipar alimento tras repeticiones.",
        episode_id="ep_adaptive_query",
    )

    async def create_job():
        await store.create_job(uid="job_adaptive_1", episode_id=episode.uid, status="completed")

    asyncio.run(create_job())

    query_response = client.post(
        "/v1/adaptive/sessions/start",
        headers=auth_headers,
        json={"user_id": "user-1", "query": "condicionamiento pavloviano"},
    )
    assert query_response.status_code == 200
    assert query_response.json()["grounding_status"] == "ok"

    episode_response = client.post(
        "/v1/adaptive/sessions/start",
        headers=auth_headers,
        json={"user_id": "user-1", "episode_id": episode.uid},
    )
    assert episode_response.status_code == 200
    assert episode_response.json()["session"]["resolved_reference"]["resolved_episode_id"] == episode.uid

    job_response = client.post(
        "/v1/adaptive/sessions/start",
        headers=auth_headers,
        json={"user_id": "user-1", "job_id": "job_adaptive_1"},
    )
    assert job_response.status_code == 200
    assert job_response.json()["session"]["resolved_reference"]["resolved_job_id"] == "job_adaptive_1"


def test_adaptive_session_rejects_missing_traceable_evidence(client, auth_headers, settings, store):
    provider = StubAIProvider(settings)
    concept = _seed_concept(
        store,
        provider,
        canonical_name="Tema incompleto",
        aliases=[],
        domain="Psicología",
        description="Concepto sin evidencia suficiente.",
    )

    async def run():
        episode = await store.create_episode(
            uid="ep_sin_soporte",
            text="Fragmento sin claims.",
            source_type="manual_input",
            tags=["Psicología"],
            language="es",
        )
        await store.update_episode(episode.uid, status="processed")
        await store.link_concept_to_episode(concept.uid, episode.uid, confidence=0.8)

    asyncio.run(run())

    response = client.post(
        "/v1/adaptive/sessions/start",
        headers=auth_headers,
        json={"user_id": "user-1", "episode_id": "ep_sin_soporte"},
    )
    assert response.status_code == 422


def test_adaptive_submit_returns_next_block_or_closes_session(client, auth_headers, settings, store):
    provider = StubAIProvider(settings)
    concept = _seed_concept(
        store,
        provider,
        canonical_name="Memoria episódica",
        aliases=["recuerdo autobiográfico"],
        domain="Psicología",
        description="Sistema de memoria autobiográfica.",
    )
    _seed_claim_and_episode(
        store,
        provider,
        claim_text="La memoria episódica recupera experiencias personales con contexto temporal.",
        concept_uids=[concept.uid],
        episode_text="La memoria episódica conserva experiencias situadas en tiempo y contexto.",
        episode_id="ep_adaptive_flow",
    )

    start = client.post(
        "/v1/adaptive/sessions/start",
        headers=auth_headers,
        json={"user_id": "user-1", "query": "recuerdo autobiográfico"},
    )
    assert start.status_code == 200
    body = start.json()
    block = body["current_block"]
    submissions = []
    for answer_key in block["answer_keys"]:
        if answer_key["correct_choice_indexes"]:
            submissions.append({"item_id": answer_key["item_id"], "selected_choices": answer_key["correct_choice_indexes"]})
        elif answer_key["boolean_answer"] is not None:
            submissions.append({"item_id": answer_key["item_id"], "boolean_answer": answer_key["boolean_answer"]})
        else:
            submissions.append({"item_id": answer_key["item_id"], "response_text": answer_key["expected"][0]})

    submit = client.post(
        f"/v1/adaptive/sessions/{body['session']['session_id']}/submit",
        headers=auth_headers,
        json={"block_id": block["block_id"], "submissions": submissions},
    )
    assert submit.status_code == 200
    assert submit.json()["next_block"] is not None
    assert submit.json()["session_closed"] is False

    start_closing = client.post(
        "/v1/adaptive/sessions/start",
        headers=auth_headers,
        json={
            "user_id": "user-2",
            "query": "recuerdo autobiográfico",
            "constraints": {"max_blocks": 1},
        },
    )
    assert start_closing.status_code == 200
    closing = start_closing.json()
    closing_submissions = []
    for answer_key in closing["current_block"]["answer_keys"]:
        if answer_key["correct_choice_indexes"]:
            closing_submissions.append({"item_id": answer_key["item_id"], "selected_choices": answer_key["correct_choice_indexes"]})
        elif answer_key["boolean_answer"] is not None:
            closing_submissions.append({"item_id": answer_key["item_id"], "boolean_answer": answer_key["boolean_answer"]})
        else:
            closing_submissions.append({"item_id": answer_key["item_id"], "response_text": answer_key["expected"][0]})
    closing_submit = client.post(
        f"/v1/adaptive/sessions/{closing['session']['session_id']}/submit",
        headers=auth_headers,
        json={"block_id": closing["current_block"]["block_id"], "submissions": closing_submissions},
    )
    assert closing_submit.status_code == 200
    assert closing_submit.json()["session_closed"] is True


def test_adaptive_service_scores_multi_cloze_and_open(settings, store):
    service = _make_service(settings, store)
    concept_state = PedagogicalConceptState(
        user_id="user-1",
        concept_uid="cn_1",
        concept_name="Memoria episódica",
        domain="Psicología",
        mastery_score_0_to_100=50.0,
        mastery_label="medio",
        dimensions=PedagogicalDimensionStates(
            recognition=PedagogicalDimensionState(score_0_to_100=40.0),
            recall=PedagogicalDimensionState(score_0_to_100=35.0),
            explanation=PedagogicalDimensionState(score_0_to_100=30.0),
            application=PedagogicalDimensionState(score_0_to_100=25.0),
        ),
        confidence_0_to_1=0.4,
        trend="stable",
        priority_score=0.8,
        last_block_id=None,
        recent_history=[],
        recent_stats=PedagogicalRecentStats(recent_average=0.0, trend="insufficient_data", deviation=0.0, last_evaluated_at=None),
        weaknesses=[],
        detected_gaps=[],
        suggested_questions=[],
        effective_depth_used=3,
        last_evaluated_at=None,
        updated_at="2026-06-06T10:00:00+00:00",
        recalculation_traces=[],
    )
    multi_item = AdaptiveBlockItem(
        item_id="item_multi",
        question_type="multiple_choice_multi",
        concept_uid="cn_1",
        target_dimension="application",
        difficulty="intermediate",
        prompt="Selecciona todas las correctas.",
        choices=["a", "b", "c", "d"],
        rubric={},
        grounding_refs=["cl_1"],
    )
    multi_key = AdaptiveBlockAnswerKey(
        item_id="item_multi",
        grading_mode="partial_multi_choice",
        expected=["a", "b"],
        correct_choice_indexes=[0, 1],
        rationale="",
    )
    multi_result = service._evaluate_item(
        item=multi_item,
        answer_key=multi_key,
        submission=AdaptiveItemSubmission(item_id="item_multi", selected_choices=[0, 2]),
        interaction=AdaptiveInteractionEvent(item_id="item_multi"),
    )
    assert multi_result.verdict == "partial_low"

    cloze_item = AdaptiveBlockItem(
        item_id="item_cloze",
        question_type="cloze",
        concept_uid="cn_1",
        target_dimension="recall",
        difficulty="intermediate",
        prompt="Completa.",
        choices=[],
        rubric={},
        grounding_refs=["cl_1"],
    )
    cloze_key = AdaptiveBlockAnswerKey(
        item_id="item_cloze",
        grading_mode="rules_plus_semantic",
        expected=["experiencias personales con contexto temporal"],
        rationale="",
    )
    cloze_result = service._evaluate_item(
        item=cloze_item,
        answer_key=cloze_key,
        submission=AdaptiveItemSubmission(item_id="item_cloze", response_text="experiencias personales con contexto"),
        interaction=AdaptiveInteractionEvent(item_id="item_cloze", hint_used=True),
    )
    assert cloze_result.verdict in {"partial_high", "partial_low"}
    assert cloze_result.score_0_to_1 < 1.0

    open_item = AdaptiveBlockItem(
        item_id="item_open",
        question_type="open",
        concept_uid="cn_1",
        target_dimension="explanation",
        difficulty="intermediate",
        prompt="Explica.",
        choices=[],
        rubric={},
        grounding_refs=["cl_1"],
    )
    open_key = AdaptiveBlockAnswerKey(
        item_id="item_open",
        grading_mode="rubric_structured",
        expected=["La memoria episódica recupera experiencias personales con contexto temporal."],
        rationale="",
    )
    open_result = service._evaluate_item(
        item=open_item,
        answer_key=open_key,
        submission=AdaptiveItemSubmission(item_id="item_open", response_text="Recupera experiencias personales con contexto temporal."),
        interaction=AdaptiveInteractionEvent(item_id="item_open", retry_used=True),
    )
    assert open_result.verdict in {"partial_high", "correct"}
    assert open_result.score_0_to_1 < 1.0
