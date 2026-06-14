from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


def _raise_for_status(
    response: httpx.Response,
    *,
    operation: str,
    command: str | None = None,
    ignore_if: str | None = None,
) -> None:
    """raise_for_status, but log the ArcadeDB error body first so DB-level
    failures (constraint violations, syntax errors) are not opaque.

    If `ignore_if` is set and that substring appears in the error body,
    the error is silently swallowed (used for idempotent schema bootstrap).
    """
    if response.status_code < 400:
        return
    body = (response.text or "").strip()
    if ignore_if and ignore_if in body:
        response.raise_for_status()
        return
    logger.error(
        "arcadedb error",
        extra={
            "operation": operation,
            "status_code": response.status_code,
            "command": (command or "")[:300],
            "body": body[:500],
        },
    )
    response.raise_for_status()


class ArcadeDBClient:
    def __init__(self, settings: Settings, *, database: str | None = None) -> None:
        self.settings = settings
        self._database = database or settings.arcadedb_database
        self.client = httpx.AsyncClient(
            base_url=settings.arcadedb_url.rstrip("/") + "/",
            auth=(settings.arcadedb_root_username, settings.arcadedb_root_password),
            timeout=60.0,
        )

    async def ready(self) -> bool:
        response = await self.client.get("api/v1/ready")
        return response.status_code in {200, 204}

    async def server_command(self, command: str) -> Any:
        response = await self.client.post(
            "api/v1/server",
            json={"command": command},
            headers={"Content-Type": "application/json"},
        )
        _raise_for_status(response, operation="server_command", command=command)
        return response.json().get("result")

    async def database_exists(self, database: str | None = None) -> bool:
        name = database or self._database
        response = await self.client.get(f"api/v1/exists/{name}")
        _raise_for_status(response, operation="database_exists", command=name)
        return bool(response.json()["result"])

    async def query(self, command: str, params: dict[str, Any] | None = None, language: str = "sql") -> list[dict[str, Any]]:
        response = await self.client.post(
            f"api/v1/query/{self._database}",
            json={"language": language, "command": command, "params": params or {}},
            headers={"Content-Type": "application/json"},
        )
        _raise_for_status(response, operation="query", command=command)
        return response.json().get("result", [])

    async def command(self, command: str, params: dict[str, Any] | None = None, language: str = "sql") -> list[dict[str, Any]]:
        response = await self.client.post(
            f"api/v1/command/{self._database}",
            json={"language": language, "command": command, "params": params or {}},
            headers={"Content-Type": "application/json"},
        )
        _raise_for_status(response, operation="command", command=command)
        return response.json().get("result", [])

    async def command_idempotent(self, command: str, params: dict[str, Any] | None = None, language: str = "sql") -> list[dict[str, Any]]:
        """Like command(), but silences 'already exists' errors without logging them."""
        response = await self.client.post(
            f"api/v1/command/{self._database}",
            json={"language": language, "command": command, "params": params or {}},
            headers={"Content-Type": "application/json"},
        )
        _raise_for_status(response, operation="command", command=command, ignore_if="already exists")
        try:
            return response.json().get("result", [])
        except Exception:
            return []

    async def close(self) -> None:
        await self.client.aclose()
