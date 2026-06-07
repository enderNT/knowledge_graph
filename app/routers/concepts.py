from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.dependencies import require_private_api_access
from app.schemas import (
    AttachConceptEvidenceRequest,
    AttachConceptEvidenceResponse,
    CreateConceptRequest,
    LinkConceptsRequest,
    NeighborhoodResponse,
    UpsertConceptRequest,
)
from app.store import ConceptConflictError, ConceptUpsertTargetNotFoundError


router = APIRouter(prefix="/v1/concepts", tags=["concepts"])


@router.post("")
async def create_concept(payload: CreateConceptRequest, services=Depends(require_private_api_access)):
    embedding = await services.ai_provider.embed(f"{payload.canonical_name}\n{payload.description}")
    try:
        concept = await services.store.create_concept(
            payload,
            embedding=embedding,
            source_confidence=1.0,
        )
    except ConceptConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return {"concept": concept, "created": True}


@router.put("/upsert")
async def upsert_concept(payload: UpsertConceptRequest, services=Depends(require_private_api_access)):
    embedding = await services.ai_provider.embed(f"{payload.canonical_name}\n{payload.description}")
    try:
        concept, created = await services.store.upsert_concept(
            payload,
            embedding=embedding,
            source_confidence=1.0,
        )
    except ConceptConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ConceptUpsertTargetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
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


@router.post("/evidence", response_model=AttachConceptEvidenceResponse)
async def attach_concept_evidence(
    payload: AttachConceptEvidenceRequest,
    services=Depends(require_private_api_access),
) -> AttachConceptEvidenceResponse:
    try:
        return await services.store.attach_concept_evidence(
            concept_ref=payload.concept_ref,
            episode_id=payload.episode_id,
            link_episode_claims=payload.link_episode_claims,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


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
