from fastapi import Header, HTTPException, status

from app.config import Settings


def require_api_key(settings: Settings, api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid api key",
        )
