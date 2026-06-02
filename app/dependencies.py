from __future__ import annotations

from typing import Annotated, Any

from fastapi import Header, HTTPException, Request, status

def get_services(request: Request) -> Any:
    return request.app.state.services  # type: ignore[no-any-return]


def require_private_api_access(
    request: Request,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> Any:
    services = get_services(request)
    if api_key != services.settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid api key",
        )
    if not services.bootstrap_complete:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=services.bootstrap_error or "bootstrap in progress",
        )
    return services
