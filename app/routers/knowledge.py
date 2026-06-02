from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.dependencies import require_private_api_access
from app.schemas import AddKnowledgeFragmentAccepted, AddKnowledgeFragmentRequest


router = APIRouter(prefix="/v1/knowledge", tags=["knowledge"])


@router.post("/fragments", response_model=AddKnowledgeFragmentAccepted, status_code=status.HTTP_202_ACCEPTED)
async def add_knowledge_fragment(
    payload: AddKnowledgeFragmentRequest,
    services=Depends(require_private_api_access),
) -> AddKnowledgeFragmentAccepted:
    return await services.ingestion_service.submit_fragment(payload)
