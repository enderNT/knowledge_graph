from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import Header, HTTPException, Request, status

logger = logging.getLogger(__name__)


def get_services(request: Request) -> Any:
    return request.app.state.services  # type: ignore[no-any-return]


def require_private_api_access(
    request: Request,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> Any:
    services = get_services(request)
    if api_key != services.settings.api_key:
        logger.warning(
            "rejected request with invalid api key",
            extra={"path": request.url.path, "api_key_present": api_key is not None},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid api key",
        )
    if not services.bootstrap_complete:
        logger.warning(
            "rejected request because bootstrap is not complete",
            extra={"path": request.url.path, "bootstrap_error": services.bootstrap_error},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=services.bootstrap_error or "bootstrap in progress",
        )
    return services
