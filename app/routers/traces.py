from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from app.dependencies import require_private_api_access
from app.trace_export import render_trace_text
from app.trace_models import CanonicalTrace, TraceSummary


router = APIRouter(prefix="/v1/traces", tags=["traces"])


@router.get("", response_model=list[TraceSummary])
async def list_traces(
    sort: Annotated[str, Query(pattern="^(asc|desc)$")] = "desc",
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    skip: Annotated[int, Query(ge=0)] = 0,
    execution_id: str | None = None,
    episode_id: str | None = None,
    trace_status: Annotated[str | None, Query(alias="status")] = None,
    domain: str | None = None,
    services=Depends(require_private_api_access),
) -> list[TraceSummary]:
    return await services.store.list_canonical_traces(
        sort=sort,
        limit=limit,
        skip=skip,
        execution_id=execution_id,
        episode_id=episode_id,
        status=trace_status,
        domain=domain,
    )


@router.get("/{trace_id}", response_model=CanonicalTrace)
async def get_trace(trace_id: str, services=Depends(require_private_api_access)) -> CanonicalTrace:
    trace = await services.store.get_canonical_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="trace not found")
    return trace


@router.get("/{trace_id}/export")
async def export_trace(trace_id: str, services=Depends(require_private_api_access)) -> Response:
    trace = await services.store.get_canonical_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="trace not found")
    return Response(content=render_trace_text(trace), media_type="text/plain; charset=utf-8")
