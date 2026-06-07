from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import require_private_api_access
from app.schemas import (
    ApplyPrereqReliefRequest,
    ApplyPrereqReliefResponse,
    GetDueSRItemsResponse,
    GetSRStateRequest,
    GetSRStateResponse,
    GetSRStatsResponse,
    UpdateSRFromBlockRequest,
    UpdateSRFromBlockResponse,
)
from app.spaced_repetition import SpacedRepetitionService


router = APIRouter(prefix="/v1/sr", tags=["spaced_repetition"])


@router.get("/state", response_model=GetSRStateResponse)
async def get_sr_state(
    user_id: str,
    concept_uid: str,
    dimension: str,
    services=Depends(require_private_api_access),
) -> GetSRStateResponse:
    service = SpacedRepetitionService(settings=services.settings, store=services.store)
    return await service.get_state(
        GetSRStateRequest(
            user_id=user_id,
            concept_uid=concept_uid,
            dimension=dimension,
        )
    )


@router.get("/due", response_model=GetDueSRItemsResponse)
async def get_due_sr_items(
    user_id: str,
    services=Depends(require_private_api_access),
) -> GetDueSRItemsResponse:
    service = SpacedRepetitionService(settings=services.settings, store=services.store)
    return await service.get_due_items(user_id=user_id)


@router.post("/update", response_model=UpdateSRFromBlockResponse)
async def update_sr_from_block(
    payload: UpdateSRFromBlockRequest,
    services=Depends(require_private_api_access),
) -> UpdateSRFromBlockResponse:
    service = SpacedRepetitionService(settings=services.settings, store=services.store)
    return await service.update_from_block_result(payload)


@router.post("/relief", response_model=ApplyPrereqReliefResponse)
async def apply_prereq_relief(
    payload: ApplyPrereqReliefRequest,
    services=Depends(require_private_api_access),
) -> ApplyPrereqReliefResponse:
    service = SpacedRepetitionService(settings=services.settings, store=services.store)
    return await service.apply_prereq_relief(payload)


@router.get("/stats", response_model=GetSRStatsResponse)
async def get_sr_stats(
    user_id: str,
    services=Depends(require_private_api_access),
) -> GetSRStatsResponse:
    service = SpacedRepetitionService(settings=services.settings, store=services.store)
    return await service.get_stats(user_id=user_id)
