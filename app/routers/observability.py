from __future__ import annotations

import asyncio
import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.dependencies import require_private_api_access

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/observability", tags=["observability"])


def _obs(services) -> "ObservabilityService":  # type: ignore[name-defined]
    from app.observability import ObservabilityService

    svc = getattr(services, "obs_service", None)
    if svc is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="observability not available")
    return svc


@router.get("/runs")
async def list_runs(
    sort: Annotated[str, Query(pattern="^(asc|desc)$")] = "desc",
    limit: Annotated[int, Query(ge=1, le=10)] = 10,
    skip: Annotated[int, Query(ge=0)] = 0,
    job_id: str | None = None,
    session_id: str | None = None,
    episode_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    services=Depends(require_private_api_access),
):
    obs = _obs(services)
    runs = await obs.list_runs(
        sort=sort,
        limit=limit,
        skip=skip,
        job_id=job_id,
        session_id=session_id,
        episode_id=episode_id,
        since=since,
        until=until,
    )
    return {"runs": runs, "limit": limit, "skip": skip}


@router.get("/runs/{run_id}")
async def get_run_detail(run_id: str, services=Depends(require_private_api_access)):
    obs = _obs(services)
    run = await obs.get_run(run_id)
    events = await obs.get_run_events(run_id)
    return {"run": run, "events": events}


@router.get("/stream")
async def stream_logs(
    run_id: str | None = None,
    job_id: str | None = None,
    session_id: str | None = None,
    services=Depends(require_private_api_access),
):
    obs = _obs(services)

    async def _generate():
        q = obs.subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=25.0)
                    if run_id and event.get("run_id") != run_id:
                        continue
                    if job_id and event.get("job_id") != job_id:
                        continue
                    if session_id and event.get("session_id") != session_id:
                        continue
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            obs.unsubscribe(q)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
