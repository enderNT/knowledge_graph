from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.dependencies import require_private_api_access
from app.schemas import LinkConceptsRequest, NeighborhoodResponse, UpsertConceptRequest


router = APIRouter(prefix="/v1/concepts", tags=["concepts"])


@router.put("/upsert")
async def upsert_concept(payload: UpsertConceptRequest, services=Depends(require_private_api_access)):
    embedding = await services.ai_provider.embed(f"{payload.canonical_name}\n{payload.description}")
    concept, created = await services.store.upsert_concept(
        payload,
        embedding=embedding,
        source_confidence=1.0,
    )
    return {"concept": concept, "created": created}


@router.post("/link")
async def link_concepts(payload: LinkConceptsRequest, services=Depends(require_private_api_access)):
    created = await services.store.create_relation(
        from_ref=payload.from_,
        relation=payload.relation,
        to_ref=payload.to,
        evidence_episode_id=payload.evidence_episode_id,
        confidence=1.0,
    )
    if not created:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="source or target concept not found",
        )
    return {"status": "linked"}


@router.get("/{concept_ref}/neighborhood", response_model=NeighborhoodResponse)
async def get_neighborhood(
    concept_ref: str,
    depth: int = Query(default=1, ge=1, le=2),
    services=Depends(require_private_api_access),
) -> NeighborhoodResponse:
    neighborhood = await services.store.get_neighborhood(concept_ref, depth)
    if not neighborhood:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="concept not found")
    return neighborhood
