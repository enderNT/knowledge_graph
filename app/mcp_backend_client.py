from __future__ import annotations

import time
from typing import Any

import httpx

from app.config import Settings


class MCPBackendError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class MCPBackendAuthError(MCPBackendError):
    pass


class MCPBackendNotFoundError(MCPBackendError):
    pass


class MCPBackendClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.kg_api_base_url.rstrip("/"),
            headers={"X-API-Key": settings.kg_api_key},
            timeout=timeout,
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def check_ready(self) -> tuple[bool, dict[str, Any]]:
        try:
            response = await self._client.get("/health/ready")
        except httpx.HTTPError:
            return False, {"status": "degraded", "detail": "backend request failed"}

        try:
            payload = response.json()
        except ValueError:
            payload = {"status": "degraded", "detail": response.text.strip() or "invalid backend response"}

        if response.status_code < 400:
            return True, payload
        if response.status_code == 503 and isinstance(payload, dict):
            return False, payload
        return False, {"status": "degraded", "detail": self._extract_error_detail(response) or "backend not ready"}

    async def add_knowledge_fragment(
        self,
        *,
        text: str,
        source_type: str = "manual_input",
        tags: list[str] | None = None,
        language: str = "es",
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/knowledge/fragments",
            json={
                "text": text,
                "source_type": source_type,
                "tags": tags or [],
                "language": language,
            },
        )

    async def get_job(self, job_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/jobs/{job_id}")

    async def search_candidates(
        self,
        *,
        query: str,
        domain_hint: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/search/candidates",
            json={"query": query, "domain_hint": domain_hint, "limit": limit},
        )

    async def upsert_concept(
        self,
        *,
        canonical_name: str,
        aliases: list[str] | None = None,
        domain: str,
        description: str = "",
    ) -> dict[str, Any]:
        return await self._request(
            "PUT",
            "/v1/concepts/upsert",
            json={
                "canonical_name": canonical_name,
                "aliases": aliases or [],
                "domain": domain,
                "description": description,
            },
        )

    async def link_concepts(
        self,
        *,
        from_ref: str,
        relation: str,
        to_ref: str,
        evidence_episode_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/concepts/link",
            json={
                "from": from_ref,
                "relation": relation,
                "to": to_ref,
                "evidence_episode_id": evidence_episode_id,
            },
        )

    async def get_neighborhood(self, *, concept: str, depth: int = 1) -> dict[str, Any]:
        return await self._request(
            "GET",
            f"/v1/concepts/{concept}/neighborhood",
            params={"depth": depth},
        )

    async def ingest_fragment_and_wait(
        self,
        *,
        text: str,
        source_type: str = "manual_input",
        tags: list[str] | None = None,
        language: str = "es",
    ) -> dict[str, Any]:
        accepted = await self.add_knowledge_fragment(
            text=text,
            source_type=source_type,
            tags=tags,
            language=language,
        )
        episode_id = accepted["episode_id"]
        job_id = accepted["job_id"]
        deadline = time.monotonic() + self._settings.mcp_ingestion_timeout_seconds

        while True:
            job = await self.get_job(job_id)
            status = job["status"]
            if status == "completed":
                return {
                    "status": status,
                    "episode_id": episode_id,
                    "job_id": job_id,
                    "result": job.get("result"),
                }
            if status == "failed":
                return {
                    "status": status,
                    "episode_id": episode_id,
                    "job_id": job_id,
                    "error": job.get("error") or "ingestion failed",
                }
            if time.monotonic() >= deadline:
                return {
                    "status": "processing",
                    "episode_id": episode_id,
                    "job_id": job_id,
                }
            await self._sleep(self._settings.mcp_poll_interval_seconds)

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise MCPBackendError("backend request failed") from exc

        if response.status_code < 400:
            return response.json()

        detail = self._extract_error_detail(response)
        if response.status_code in {401, 403}:
            raise MCPBackendAuthError(detail or "backend authentication failed", status_code=response.status_code)
        if response.status_code == 404:
            raise MCPBackendNotFoundError(detail or "backend resource not found", status_code=response.status_code)
        if response.status_code >= 500:
            raise MCPBackendError(detail or "backend server error", status_code=response.status_code)
        raise MCPBackendError(detail or "backend request failed", status_code=response.status_code)

    @staticmethod
    def _extract_error_detail(response: httpx.Response) -> str | None:
        try:
            payload = response.json()
        except ValueError:
            return response.text.strip() or None

        if isinstance(payload, dict):
            detail = payload.get("detail")
            if isinstance(detail, str):
                return detail
            error = payload.get("error")
            if isinstance(error, str):
                return error
        return None

    async def _sleep(self, seconds: float) -> None:
        import asyncio

        await asyncio.sleep(seconds)