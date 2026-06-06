from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.config import Settings


class AgentMCPUpstreamError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class AgentMCPUpstreamClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 20.0,
    ) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.knowledge_mcp_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.knowledge_mcp_bearer_token}"},
            timeout=timeout,
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def check_ready(self) -> tuple[bool, dict[str, Any]]:
        try:
            async with self._session() as session:
                tools = await session.list_tools()
        except AgentMCPUpstreamError as exc:
            return False, {"status": "degraded", "detail": str(exc)}

        names = sorted(tool.name for tool in tools.tools)
        expected = {
            "add_knowledge_fragment",
            "search_candidates",
            "get_learning_context",
            "get_tutor_context",
            "upsert_concept",
            "link_concepts",
            "get_neighborhood",
        }
        missing = sorted(expected.difference(names))
        if missing:
            return False, {"status": "degraded", "detail": f"missing upstream tools: {', '.join(missing)}"}
        return True, {"status": "ready", "upstream_tool_count": len(names)}

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            async with self._session() as session:
                result = await session.call_tool(name, arguments or {})
        except AgentMCPUpstreamError:
            raise
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise AgentMCPUpstreamError("upstream MCP request failed") from exc

        if result.isError:
            raise AgentMCPUpstreamError(self._extract_error_text(result) or f"upstream tool failed: {name}")

        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict):
            return structured
        if isinstance(structured, list):
            return {"items": structured}

        text = self._extract_text(result)
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AgentMCPUpstreamError(f"invalid upstream response for tool: {name}") from exc
        if not isinstance(payload, dict):
            return {"value": payload}
        return payload

    async def add_knowledge_fragment(
        self,
        *,
        text: str,
        source_type: str = "manual_input",
        tags: list[str] | None = None,
        language: str = "es",
    ) -> dict[str, Any]:
        return await self.call_tool(
            "add_knowledge_fragment",
            {
                "text": text,
                "source_type": source_type,
                "tags": tags or [],
                "language": language,
            },
        )

    async def search_candidates(
        self,
        *,
        query: str,
        domain_hint: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "search_candidates",
            {"query": query, "domain_hint": domain_hint, "limit": limit},
        )

    async def get_learning_context(
        self,
        *,
        query: str,
        domain_hint: str | None = None,
        candidate_limit: int = 8,
        concept_limit: int = 3,
        claim_limit: int = 6,
        episode_limit: int = 3,
        include_neighborhood: bool = True,
        depth: int = 1,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "get_learning_context",
            {
                "query": query,
                "domain_hint": domain_hint,
                "candidate_limit": candidate_limit,
                "concept_limit": concept_limit,
                "claim_limit": claim_limit,
                "episode_limit": episode_limit,
                "include_neighborhood": include_neighborhood,
                "depth": depth,
            },
        )

    async def get_tutor_context(
        self,
        *,
        query: str | None = None,
        episode_id: str | None = None,
        job_id: str | None = None,
        depth: int = 1,
        include_evidence: bool = True,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "get_tutor_context",
            {
                "query": query,
                "episode_id": episode_id,
                "job_id": job_id,
                "depth": depth,
                "include_evidence": include_evidence,
            },
        )

    async def upsert_concept(
        self,
        *,
        canonical_name: str,
        aliases: list[str] | None = None,
        domain: str,
        description: str = "",
    ) -> dict[str, Any]:
        return await self.call_tool(
            "upsert_concept",
            {
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
        return await self.call_tool(
            "link_concepts",
            {
                "from": from_ref,
                "relation": relation,
                "to": to_ref,
                "evidence_episode_id": evidence_episode_id,
            },
        )

    async def get_neighborhood(self, *, concept: str, depth: int = 1) -> dict[str, Any]:
        return await self.call_tool(
            "get_neighborhood",
            {"concept": concept, "depth": depth},
        )

    @asynccontextmanager
    async def _session(self):
        try:
            async with streamable_http_client(
                self._settings.knowledge_mcp_base_url.rstrip("/") + "/mcp",
                http_client=self._client,
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session
        except httpx.HTTPStatusError as exc:
            message = f"upstream MCP http error: {exc.response.status_code}"
            raise AgentMCPUpstreamError(message, status_code=exc.response.status_code) from exc
        except httpx.HTTPError as exc:
            raise AgentMCPUpstreamError("upstream MCP transport failed") from exc
        except Exception as exc:  # pragma: no cover - SDK-level failure
            raise AgentMCPUpstreamError("upstream MCP session failed") from exc

    @staticmethod
    def _extract_text(result: Any) -> str:
        for item in getattr(result, "content", []) or []:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                return text
        return ""

    @classmethod
    def _extract_error_text(cls, result: Any) -> str | None:
        text = cls._extract_text(result)
        return text.strip() or None
