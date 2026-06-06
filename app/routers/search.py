from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import require_private_api_access
from app.learning_context import LearningContextBuilder, TutorContextBuilder
from app.schemas import (
    LearningContextRequest,
    LearningContextResponse,
    SearchCandidatesRequest,
    SearchCandidatesResponse,
    TutorContextRequest,
    TutorContextResponse,
)


router = APIRouter(prefix="/v1/search", tags=["search"])


@router.post("/candidates", response_model=SearchCandidatesResponse)
async def search_candidates(
    payload: SearchCandidatesRequest,
    services=Depends(require_private_api_access),
) -> SearchCandidatesResponse:
    query_embedding = await services.ai_provider.embed(payload.query)
    results = await services.store.search_candidates(
        query=payload.query,
        domain_hint=payload.domain_hint,
        query_embedding=query_embedding,
        limit=payload.limit,
    )
    return SearchCandidatesResponse(query=payload.query, results=results)


@router.post("/learning-context", response_model=LearningContextResponse)
async def get_learning_context(
    payload: LearningContextRequest,
    services=Depends(require_private_api_access),
) -> LearningContextResponse:
    builder = LearningContextBuilder(
        settings=services.settings,
        store=services.store,
        ai_provider=services.ai_provider,
    )
    return await builder.build(payload)


@router.post("/tutor-context", response_model=TutorContextResponse)
async def get_tutor_context(
    payload: TutorContextRequest,
    services=Depends(require_private_api_access),
) -> TutorContextResponse:
    builder = TutorContextBuilder(
        settings=services.settings,
        store=services.store,
        ai_provider=services.ai_provider,
    )
    return await builder.build(payload)
