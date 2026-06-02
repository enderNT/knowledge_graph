from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import require_private_api_access
from app.schemas import EpisodeResponse


router = APIRouter(prefix="/v1/episodes", tags=["episodes"])


@router.get("/{episode_id}", response_model=EpisodeResponse)
async def get_episode(episode_id: str, services=Depends(require_private_api_access)) -> EpisodeResponse:
    episode = await services.store.get_episode(episode_id)
    if not episode:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="episode not found")
    return EpisodeResponse.model_validate(episode.model_dump())
