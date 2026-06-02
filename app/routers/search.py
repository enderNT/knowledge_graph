from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import require_private_api_access
from app.schemas import SearchCandidatesRequest, SearchCandidatesResponse


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
