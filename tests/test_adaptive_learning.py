from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

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
    SpacedRepetitionState,
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


def _make_service(settings, store) -> AdaptiveLearningService:
    return AdaptiveLearningService(settings=settings, store=store, ai_provider=StubAIProvider(settings))


def _build_block_submissions(block: dict) -> list[dict]:
    submissions = []
    for answer_key in block["answer_keys"]:
        if answer_key["correct_choice_indexes"]:
            submissions.append({"item_id": answer_key["item_id"], "selected_choices": answer_key["correct_choice_indexes"]})
        elif answer_key["boolean_answer"] is not None:
            submissions.append({"item_id": answer_key["item_id"], "boolean_answer": answer_key["boolean_answer"]})
        else:
            submissions.append({"item_id": answer_key["item_id"], "response_text": answer_key["expected"][0]})
    return submissions


def test_adaptive_endpoints_require_api_key(client):
    response = client.post("/v1/adaptive/sessions/start", json={"user_id": "user-1", "query": "memoria"})
    assert response.status_code == 401


def test_adaptive_session_start_request_defaults_to_hybrid_and_requires_domain_for_isolated():
    request = AdaptiveSessionStartRequest(user_id="user-1", query="memoria")
    assert request.study_mode == "hybrid"

    with pytest.raises(ValidationError):
        AdaptiveSessionStartRequest(user_id="user-1", query="memoria", study_mode="isolated")


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
    multi_result = asyncio.run(service._evaluate_item(
        item=multi_item,
        answer_key=multi_key,
        submission=AdaptiveItemSubmission(item_id="item_multi", selected_choices=[0, 2]),
        interaction=AdaptiveInteractionEvent(item_id="item_multi"),
    ))
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
    cloze_result = asyncio.run(service._evaluate_item(
        item=cloze_item,
        answer_key=cloze_key,
        submission=AdaptiveItemSubmission(item_id="item_cloze", response_text="experiencias personales con contexto"),
        interaction=AdaptiveInteractionEvent(item_id="item_cloze", hint_used=True),
    ))
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
    open_result = asyncio.run(service._evaluate_item(
        item=open_item,
        answer_key=open_key,
        submission=AdaptiveItemSubmission(item_id="item_open", response_text="Recupera experiencias personales con contexto temporal."),
        interaction=AdaptiveInteractionEvent(item_id="item_open", retry_used=True),
    ))
    assert open_result.verdict in {"partial_high", "correct"}
    assert open_result.score_0_to_1 < 1.0


def test_adaptive_session_uses_review_quota_when_due_queue_is_saturated(client, auth_headers, settings, store):
    provider = StubAIProvider(settings)
    concepts = []
    for index in range(4):
        concept = _seed_concept(
            store,
            provider,
            canonical_name=f"Concepto SR {index}",
            aliases=[f"concepto-sr-{index}"],
            domain="Psicología",
            description=f"Descripcion {index}.",
        )
        _seed_claim_and_episode(
            store,
            provider,
            claim_text=f"Claim de soporte {index} para repeticion espaciada.",
            concept_uids=[concept.uid],
            episode_text=f"Episodio de soporte {index}.",
            episode_id=f"ep_sr_{index}",
        )
        concepts.append(concept)

    async def seed_sr():
        for concept in concepts:
            await store.upsert_spaced_repetition_state(
                SpacedRepetitionState(
                    user_id="user-review",
                    concept_uid=concept.uid,
                    dimension="recall",
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

    asyncio.run(seed_sr())

    start = client.post(
        "/v1/adaptive/sessions/start",
        headers=auth_headers,
        json={"user_id": "user-review", "query": "concepto-sr-0"},
    )
    assert start.status_code == 200
    body = start.json()
    assert body["session"]["summary"]["review_blocks_target"] == 3
    assert body["session"]["summary"]["new_blocks_target"] == 1
    assert body["current_block"]["plan"]["block_purpose"] == "spaced_repetition_review"
    assert body["current_block"]["plan"]["due_item"]["concept_uid"] in {concept.uid for concept in concepts}

    submit = client.post(
        f"/v1/adaptive/sessions/{body['session']['session_id']}/submit",
        headers=auth_headers,
        json={"block_id": body["current_block"]["block_id"], "submissions": _build_block_submissions(body["current_block"])},
    )
    assert submit.status_code == 200
    assert submit.json()["block_result"]["sr_feedback"] is not None


def test_adaptive_session_prioritizes_requires_direct_validation_before_regular_due(client, auth_headers, settings, store):
    provider = StubAIProvider(settings)
    forced = _seed_concept(
        store,
        provider,
        canonical_name="Concepto Forzado",
        aliases=["concepto forzado"],
        domain="Psicología",
        description="Debe validarse directo.",
    )
    due = _seed_concept(
        store,
        provider,
        canonical_name="Concepto Due",
        aliases=["concepto due"],
        domain="Psicología",
        description="Esta vencido.",
    )
    _seed_claim_and_episode(
        store,
        provider,
        claim_text="Soporte del concepto forzado.",
        concept_uids=[forced.uid],
        episode_text="Episodio forzado.",
        episode_id="ep_forzado",
    )
    _seed_claim_and_episode(
        store,
        provider,
        claim_text="Soporte del concepto due.",
        concept_uids=[due.uid],
        episode_text="Episodio due.",
        episode_id="ep_due",
    )

    async def seed_sr():
        await store.upsert_spaced_repetition_state(
            SpacedRepetitionState(
                user_id="user-forced",
                concept_uid=forced.uid,
                dimension="application",
                repetitions=2,
                ease_factor=2.5,
                interval_days=6,
                last_reviewed_at="2026-06-01T10:00:00+00:00",
                next_review_at="2026-06-20T10:00:00+00:00",
                propagation_relief_count=2,
                requires_direct_validation=True,
                updated_at="2026-06-06T10:00:00+00:00",
            )
        )
        await store.upsert_spaced_repetition_state(
            SpacedRepetitionState(
                user_id="user-forced",
                concept_uid=due.uid,
                dimension="recall",
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

    asyncio.run(seed_sr())

    start = client.post(
        "/v1/adaptive/sessions/start",
        headers=auth_headers,
        json={"user_id": "user-forced", "query": "concepto due"},
    )
    assert start.status_code == 200
    body = start.json()
    assert body["current_block"]["plan"]["block_purpose"] == "spaced_repetition_review"
    assert body["current_block"]["plan"]["due_item"]["concept_uid"] == forced.uid
    assert body["current_block"]["plan"]["due_item"]["requires_direct_validation"] is True


def test_adaptive_backlog_mode_requires_due_candidates(client, auth_headers, settings, store):
    provider = StubAIProvider(settings)
    concept = _seed_concept(
        store,
        provider,
        canonical_name="Memoria semántica",
        aliases=["memoria semantica"],
        domain="Psicología",
        description="Conocimiento factual estable.",
    )
    _seed_claim_and_episode(
        store,
        provider,
        claim_text="La memoria semántica almacena conocimiento general.",
        concept_uids=[concept.uid],
        episode_text="La memoria semántica permite recordar hechos y conceptos.",
        episode_id="ep_backlog_no_due",
    )

    response = client.post(
        "/v1/adaptive/sessions/start",
        headers=auth_headers,
        json={"user_id": "user-no-review", "query": "memoria semantica", "study_mode": "backlog"},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "no_review_candidates_for_mode"


def test_adaptive_recovery_mode_filters_to_urgent_reviews(client, auth_headers, settings, store):
    provider = StubAIProvider(settings)
    urgent = _seed_concept(
        store,
        provider,
        canonical_name="Recuerdo Fragil",
        aliases=["recuerdo fragil"],
        domain="Psicología",
        description="Concepto urgente.",
    )
    non_urgent = _seed_concept(
        store,
        provider,
        canonical_name="Recuerdo Estable",
        aliases=["recuerdo estable"],
        domain="Psicología",
        description="Concepto no urgente.",
    )
    _seed_claim_and_episode(
        store,
        provider,
        claim_text="Soporte de recuerdo fragil.",
        concept_uids=[urgent.uid],
        episode_text="Episodio urgente.",
        episode_id="ep_recovery_urgent",
    )
    _seed_claim_and_episode(
        store,
        provider,
        claim_text="Soporte de recuerdo estable.",
        concept_uids=[non_urgent.uid],
        episode_text="Episodio estable.",
        episode_id="ep_recovery_stable",
    )

    async def seed_sr():
        await store.upsert_spaced_repetition_state(
            SpacedRepetitionState(
                user_id="user-recovery",
                concept_uid=urgent.uid,
                dimension="recall",
                repetitions=3,
                ease_factor=1.6,
                interval_days=6,
                last_reviewed_at="2026-06-01T10:00:00+00:00",
                next_review_at="2026-06-02T10:00:00+00:00",
                propagation_relief_count=0,
                requires_direct_validation=False,
                updated_at="2026-06-06T10:00:00+00:00",
            )
        )
        await store.upsert_spaced_repetition_state(
            SpacedRepetitionState(
                user_id="user-recovery",
                concept_uid=non_urgent.uid,
                dimension="recall",
                repetitions=3,
                ease_factor=2.5,
                interval_days=7,
                last_reviewed_at="2026-06-01T10:00:00+00:00",
                next_review_at="2026-06-02T10:00:00+00:00",
                propagation_relief_count=0,
                requires_direct_validation=False,
                updated_at="2026-06-06T10:00:00+00:00",
            )
        )

    asyncio.run(seed_sr())

    response = client.post(
        "/v1/adaptive/sessions/start",
        headers=auth_headers,
        json={"user_id": "user-recovery", "query": "recuerdo fragil", "study_mode": "recovery"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["session"]["study_mode"] == "recovery"
    assert body["session"]["summary"]["review_blocks_target"] == body["session"]["summary"]["total_blocks"]
    assert body["session"]["summary"]["new_blocks_target"] == 0
    assert body["current_block"]["plan"]["block_purpose"] == "spaced_repetition_review"
    assert body["current_block"]["plan"]["due_item"]["concept_uid"] == urgent.uid


def test_adaptive_isolated_mode_restricts_review_and_new_content_to_domain(client, auth_headers, settings, store):
    provider = StubAIProvider(settings)
    js_concept = _seed_concept(
        store,
        provider,
        canonical_name="Promesas",
        aliases=["promesas js"],
        domain="JavaScript",
        description="Control de asincronía en JS.",
    )
    psycho_concept = _seed_concept(
        store,
        provider,
        canonical_name="Memoria de trabajo",
        aliases=["working memory"],
        domain="Psicología",
        description="Sistema temporal de procesamiento cognitivo.",
    )
    _seed_claim_and_episode(
        store,
        provider,
        claim_text="Las promesas representan resultados futuros en JavaScript.",
        concept_uids=[js_concept.uid],
        episode_text="Las promesas encapsulan valores asincrónicos.",
        episode_id="ep_iso_js",
    )
    _seed_claim_and_episode(
        store,
        provider,
        claim_text="La memoria de trabajo mantiene información activa temporalmente.",
        concept_uids=[psycho_concept.uid],
        episode_text="La memoria de trabajo ayuda a manipular información actual.",
        episode_id="ep_iso_psy",
    )

    async def seed_sr():
        await store.upsert_spaced_repetition_state(
            SpacedRepetitionState(
                user_id="user-isolated",
                concept_uid=psycho_concept.uid,
                dimension="recall",
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

    asyncio.run(seed_sr())

    response = client.post(
        "/v1/adaptive/sessions/start",
        headers=auth_headers,
        json={
            "user_id": "user-isolated",
            "query": "promesas js",
            "study_mode": "isolated",
            "domain_hint": "JavaScript",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["session"]["study_mode"] == "isolated"
    assert body["current_block"]["plan"]["block_purpose"] == "new_content"
    assert body["current_block"]["plan"]["concept_domain"] == "JavaScript"
