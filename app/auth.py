import logging

from fastapi import Header, HTTPException, status

from app.config import Settings

logger = logging.getLogger(__name__)


def require_api_key(settings: Settings, api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if api_key != settings.api_key:
        logger.warning("rejected request with invalid api key", extra={"api_key_present": api_key is not None})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid api key",
        )
