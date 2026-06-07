from __future__ import annotations

from fastapi import APIRouter, Depends

from app.adaptive_learning import AdaptiveLearningService
from app.dependencies import require_private_api_access
from app.schemas import (
    AdaptiveBlockSubmissionRequest,
    AdaptiveBlockSubmissionResponse,
    AdaptiveSessionSnapshot,
    AdaptiveSessionStartRequest,
    AdaptiveSessionStartResponse,
)


router = APIRouter(prefix="/v1/adaptive", tags=["adaptive"])


@router.post("/sessions/start", response_model=AdaptiveSessionStartResponse)
async def start_adaptive_session(
    payload: AdaptiveSessionStartRequest,
    services=Depends(require_private_api_access),
) -> AdaptiveSessionStartResponse:
    service = AdaptiveLearningService(
        settings=services.settings,
        store=services.store,
        ai_provider=services.ai_provider,
    )
    return await service.start_session(payload)


@router.post("/sessions/{session_id}/submit", response_model=AdaptiveBlockSubmissionResponse)
async def submit_adaptive_block(
    session_id: str,
    payload: AdaptiveBlockSubmissionRequest,
    services=Depends(require_private_api_access),
) -> AdaptiveBlockSubmissionResponse:
    service = AdaptiveLearningService(
        settings=services.settings,
        store=services.store,
        ai_provider=services.ai_provider,
    )
    return await service.submit_block(session_id=session_id, payload=payload)


@router.get("/sessions/{session_id}", response_model=AdaptiveSessionSnapshot)
async def get_adaptive_session(
    session_id: str,
    services=Depends(require_private_api_access),
) -> AdaptiveSessionSnapshot:
    service = AdaptiveLearningService(
        settings=services.settings,
        store=services.store,
        ai_provider=services.ai_provider,
    )
    return await service.get_session(session_id)
