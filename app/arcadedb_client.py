from __future__ import annotations

from typing import Any

import httpx

from app.config import Settings


class ArcadeDBClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
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
        response.raise_for_status()
        return response.json().get("result")

    async def database_exists(self, database: str | None = None) -> bool:
        name = database or self.settings.arcadedb_database
        response = await self.client.get(f"api/v1/exists/{name}")
        response.raise_for_status()
        return bool(response.json()["result"])

    async def query(self, command: str, params: dict[str, Any] | None = None, language: str = "sql") -> list[dict[str, Any]]:
        response = await self.client.post(
            f"api/v1/query/{self.settings.arcadedb_database}",
            json={"language": language, "command": command, "params": params or {}},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json().get("result", [])

    async def command(self, command: str, params: dict[str, Any] | None = None, language: str = "sql") -> list[dict[str, Any]]:
        response = await self.client.post(
            f"api/v1/command/{self.settings.arcadedb_database}",
            json={"language": language, "command": command, "params": params or {}},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json().get("result", [])

    async def close(self) -> None:
        await self.client.aclose()
