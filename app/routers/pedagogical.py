from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import require_private_api_access
from app.pedagogical_context import PedagogicalContextBuilder
from app.schemas import (
    GetPedagogicalContextRequest,
    GetPedagogicalContextResponse,
    PedagogicalSessionViewRequest,
    PedagogicalSessionViewResponse,
    UpdatePedagogicalContextRequest,
    UpdatePedagogicalContextResponse,
)


router = APIRouter(prefix="/v1/pedagogical", tags=["pedagogical"])


@router.post("/context", response_model=GetPedagogicalContextResponse)
async def get_pedagogical_context(
    payload: GetPedagogicalContextRequest,
    services=Depends(require_private_api_access),
) -> GetPedagogicalContextResponse:
    builder = PedagogicalContextBuilder(
        settings=services.settings,
        store=services.store,
        ai_provider=services.ai_provider,
    )
    return await builder.load_context(payload)


@router.post("/update-from-evaluation", response_model=UpdatePedagogicalContextResponse)
async def update_pedagogical_context(
    payload: UpdatePedagogicalContextRequest,
    services=Depends(require_private_api_access),
) -> UpdatePedagogicalContextResponse:
    builder = PedagogicalContextBuilder(
        settings=services.settings,
        store=services.store,
        ai_provider=services.ai_provider,
    )
    return await builder.apply_evaluation_results(payload)


@router.post("/session-view", response_model=PedagogicalSessionViewResponse)
async def get_session_view(
    payload: PedagogicalSessionViewRequest,
    services=Depends(require_private_api_access),
) -> PedagogicalSessionViewResponse:
    builder = PedagogicalContextBuilder(
        settings=services.settings,
        store=services.store,
        ai_provider=services.ai_provider,
    )
    return await builder.build_session_view(payload)
