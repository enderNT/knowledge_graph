from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, status

from app.dependencies import require_private_api_access
from app.schemas import AddKnowledgeFragmentAccepted, AddKnowledgeFragmentRequest, ResetKnowledgeBaseResponse


router = APIRouter(prefix="/v1/knowledge", tags=["knowledge"])


@router.post("/fragments", response_model=AddKnowledgeFragmentAccepted, status_code=status.HTTP_202_ACCEPTED)
async def add_knowledge_fragment(
    payload: AddKnowledgeFragmentRequest,
    services=Depends(require_private_api_access),
) -> AddKnowledgeFragmentAccepted:
    return await services.ingestion_service.submit_fragment(payload)


@router.post("/reset", response_model=ResetKnowledgeBaseResponse)
async def reset_knowledge_base(
    services=Depends(require_private_api_access),
) -> ResetKnowledgeBaseResponse:
    cleared = 0
    while True:
        try:
            services.queue.get_nowait()
            services.queue.task_done()
            cleared += 1
        except asyncio.QueueEmpty:
            break
    await services.store.reset_all_data()
    return ResetKnowledgeBaseResponse(queue_cleared_count=cleared)
