from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import require_private_api_access
from app.schemas import (
    EpisodeDeletionExecuteRequest,
    EpisodeDeletionExecuteResponse,
    EpisodeDeletionPreviewRequest,
    EpisodeDeletionPreviewResponse,
    RelationDeletionExecuteRequest,
    RelationDeletionExecuteResponse,
    RelationDeletionPreviewRequest,
    RelationDeletionPreviewResponse,
)
from app.store import DeletionTargetNotFoundError, RelationDeletionConflictError

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/v1/deletions", tags=["deletions"])


@router.post("/episode/preview", response_model=EpisodeDeletionPreviewResponse)
async def preview_delete_episode_content(
    payload: EpisodeDeletionPreviewRequest,
    services=Depends(require_private_api_access),
) -> EpisodeDeletionPreviewResponse:
    try:
        return await services.store.preview_delete_episode_content(
            episode_id=payload.episode_id,
            job_id=payload.job_id,
        )
    except DeletionTargetNotFoundError as exc:
        logger.warning("delete preview target not found", extra={"episode_id": payload.episode_id, "job_id": payload.job_id, "detail": str(exc)})
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/episode/execute", response_model=EpisodeDeletionExecuteResponse)
async def delete_episode_content(
    payload: EpisodeDeletionExecuteRequest,
    services=Depends(require_private_api_access),
) -> EpisodeDeletionExecuteResponse:
    try:
        return await services.store.delete_episode_content(
            episode_id=payload.episode_id,
            job_id=payload.job_id,
        )
    except DeletionTargetNotFoundError as exc:
        logger.warning("delete episode target not found", extra={"episode_id": payload.episode_id, "job_id": payload.job_id, "detail": str(exc)})
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/relation/preview", response_model=RelationDeletionPreviewResponse)
async def preview_delete_relation(
    payload: RelationDeletionPreviewRequest,
    services=Depends(require_private_api_access),
) -> RelationDeletionPreviewResponse:
    try:
        return await services.store.preview_delete_relation(
            from_ref=payload.from_,
            relation=payload.relation,
            to_ref=payload.to,
            evidence_episode_id=payload.evidence_episode_id,
            delete_all_matching=payload.delete_all_matching,
        )
    except DeletionTargetNotFoundError as exc:
        logger.warning("relation delete preview target not found", extra={"from_ref": payload.from_, "relation": payload.relation, "to_ref": payload.to, "detail": str(exc)})
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/relation/execute", response_model=RelationDeletionExecuteResponse)
async def delete_relation(
    payload: RelationDeletionExecuteRequest,
    services=Depends(require_private_api_access),
) -> RelationDeletionExecuteResponse:
    try:
        return await services.store.delete_relation(
            from_ref=payload.from_,
            relation=payload.relation,
            to_ref=payload.to,
            evidence_episode_id=payload.evidence_episode_id,
            delete_all_matching=payload.delete_all_matching,
        )
    except DeletionTargetNotFoundError as exc:
        logger.warning("relation delete target not found", extra={"from_ref": payload.from_, "relation": payload.relation, "to_ref": payload.to, "detail": str(exc)})
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RelationDeletionConflictError as exc:
        logger.warning("relation delete conflict", extra={"from_ref": payload.from_, "relation": payload.relation, "to_ref": payload.to, "detail": str(exc)})
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
