from __future__ import annotations

import math
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.dependencies import require_private_api_access
from app.schemas import (
    EpisodeConceptSummary,
    EpisodeListItem,
    EpisodeListResponse,
    EpisodeResponse,
)


router = APIRouter(prefix="/v1/episodes", tags=["episodes"])


@router.get("", response_model=EpisodeListResponse)
async def list_episodes(
    sort_by: Annotated[str, Query(pattern="^(alphabetical|date)$")] = "alphabetical",
    sort_order: Annotated[str, Query(pattern="^(asc|desc)$")] = "asc",
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    page: Annotated[int, Query(ge=1)] = 1,
    concept_sort_by: Annotated[str | None, Query(pattern="^(alphabetical|date)$")] = None,
    concept_sort_order: Annotated[str, Query(pattern="^(asc|desc)$")] = "asc",
    services=Depends(require_private_api_access),
) -> EpisodeListResponse:
    rows, total = await services.store.list_episodes(
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        page=page,
        concept_sort_by=concept_sort_by,
        concept_sort_order=concept_sort_order,
    )

    warnings: list[str] = []
    if total > 0 and limit > total:
        warnings.append(f"limit ({limit}) exceeds total available episodes ({total}); returning all {total}")

    total_pages = max(1, math.ceil(total / limit)) if total > 0 else 1

    episodes = [
        EpisodeListItem(
            uid=row["uid"],
            source_type=row["source_type"],
            tags=row.get("tags") or [],
            language=row["language"],
            status=row["status"],
            error_message=row.get("error_message"),
            created_at=row["created_at"],
            concepts=[EpisodeConceptSummary(**c) for c in row.get("concepts", [])],
        )
        for row in rows
    ]

    return EpisodeListResponse(
        episodes=episodes,
        total=total,
        page=page,
        limit=limit,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1,
        warnings=warnings,
    )


@router.get("/{episode_id}", response_model=EpisodeResponse)
async def get_episode(episode_id: str, services=Depends(require_private_api_access)) -> EpisodeResponse:
    episode = await services.store.get_episode(episode_id)
    if not episode:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="episode not found")
    return EpisodeResponse.model_validate(episode.model_dump())
