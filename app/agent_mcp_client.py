from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.config import Settings

logger = logging.getLogger(__name__)


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
        timeout: float | None = None,
    ) -> None:
        self._settings = settings
        effective_timeout = timeout
        if effective_timeout is None:
            # Keep the upstream MCP session alive long enough for ingestion tools
            # that intentionally wait on the backend job lifecycle.
            effective_timeout = max(20.0, settings.mcp_ingestion_timeout_seconds + 10.0)
        self._client = httpx.AsyncClient(
            base_url=settings.knowledge_mcp_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.knowledge_mcp_bearer_token}"},
            timeout=effective_timeout,
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
            "patch_episode_temporality",
            "list_episodes",
            "reset_knowledge_base",
            "attach_concept_evidence",
            "create_concept",
            "search_candidates",
            "get_learning_context",
            "get_tutor_context",
            "upsert_concept",
            "link_concepts",
            "preview_delete_episode_content",
            "delete_episode_content",
            "preview_delete_relation",
            "delete_relation",
            "get_neighborhood",
            "get_pedagogical_context",
            "update_pedagogical_context",
            "get_pedagogical_session_view",
            "start_adaptive_session",
            "submit_adaptive_block",
            "get_adaptive_session",
        }
        missing = sorted(expected.difference(names))
        if missing:
            return False, {"status": "degraded", "detail": f"missing upstream tools: {', '.join(missing)}"}
        return True, {"status": "ready", "upstream_tool_count": len(names)}

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        t0_tool = time.monotonic()
        logger.info("mcp tool start", extra={"tool_name": name, "input_shape": {"arg_keys": list((arguments or {}).keys())}})
        try:
            async with self._session() as session:
                result = await session.call_tool(name, arguments or {})
        except AgentMCPUpstreamError:
            raise
        except Exception as exc:  # pragma: no cover - defensive wrapper
            logger.error("upstream MCP request failed", extra={"tool": name, "error": str(exc) or exc.__class__.__name__})
            raise AgentMCPUpstreamError(f"upstream MCP request failed: {exc or exc.__class__.__name__}") from exc

        if result.isError:
            error_text = self._extract_error_text(result)
            logger.warning("upstream tool returned error", extra={"tool": name, "error": error_text})
            raise AgentMCPUpstreamError(error_text or f"upstream tool failed: {name}")

        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict):
            return structured
        if isinstance(structured, list):
            return {"items": structured}

        text = self._extract_text(result)
        if not text:
            logger.warning("upstream tool returned empty result", extra={"tool": name})
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AgentMCPUpstreamError(f"invalid upstream response for tool: {name}") from exc
        if not isinstance(payload, dict):
            return {"value": payload}
        duration_ms = int((time.monotonic() - t0_tool) * 1000)
        logger.info("mcp tool success", extra={"tool_name": name, "duration_ms": duration_ms, "output_shape": {"result_keys": list(payload.keys())}})
        return payload

    async def add_knowledge_fragment(
        self,
        *,
        text: str,
        source_type: str = "manual_input",
        tags: list[str] | None = None,
        language: str = "es",
        temporal: bool = False,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {
            "text": text,
            "source_type": source_type,
            "tags": tags or [],
            "language": language,
        }
        if temporal:
            args["temporal"] = temporal
        if expires_at is not None:
            args["expires_at"] = expires_at
        return await self.call_tool("add_knowledge_fragment", args)

    async def patch_episode_temporality(
        self,
        episode_id: str,
        *,
        temporal: bool,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"episode_id": episode_id, "temporal": temporal}
        if expires_at is not None:
            args["expires_at"] = expires_at
        return await self.call_tool("patch_episode_temporality", args)

    async def reset_knowledge_base(self) -> dict[str, Any]:
        return await self.call_tool("reset_knowledge_base", {})

    async def list_episodes(
        self,
        *,
        sort_by: str = "alphabetical",
        sort_order: str = "asc",
        limit: int = 10,
        page: int = 1,
        concept_sort_by: str | None = None,
        concept_sort_order: str = "asc",
    ) -> dict[str, Any]:
        return await self.call_tool(
            "list_episodes",
            {
                "sort_by": sort_by,
                "sort_order": sort_order,
                "limit": limit,
                "page": page,
                "concept_sort_by": concept_sort_by,
                "concept_sort_order": concept_sort_order,
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

    async def create_concept(
        self,
        *,
        canonical_name: str,
        aliases: list[str] | None = None,
        domain: str,
        description: str = "",
    ) -> dict[str, Any]:
        return await self.call_tool(
            "create_concept",
            {
                "canonical_name": canonical_name,
                "aliases": aliases or [],
                "domain": domain,
                "description": description,
            },
        )

    async def attach_concept_evidence(
        self,
        *,
        concept_ref: str,
        episode_id: str,
        link_episode_claims: bool = True,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "attach_concept_evidence",
            {
                "concept_ref": concept_ref,
                "episode_id": episode_id,
                "link_episode_claims": link_episode_claims,
            },
        )

    async def upsert_concept(
        self,
        *,
        uid: str | None = None,
        canonical_name: str,
        aliases: list[str] | None = None,
        domain: str,
        description: str = "",
    ) -> dict[str, Any]:
        return await self.call_tool(
            "upsert_concept",
            {
                "uid": uid,
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

    async def preview_delete_episode_content(
        self,
        *,
        episode_id: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "preview_delete_episode_content",
            {"episode_id": episode_id, "job_id": job_id},
        )

    async def delete_episode_content(
        self,
        *,
        episode_id: str | None = None,
        job_id: str | None = None,
        confirm: bool = True,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "delete_episode_content",
            {"episode_id": episode_id, "job_id": job_id, "confirm": confirm},
        )

    async def preview_delete_relation(
        self,
        *,
        from_ref: str,
        relation: str,
        to_ref: str,
        evidence_episode_id: str | None = None,
        delete_all_matching: bool = False,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "preview_delete_relation",
            {
                "from": from_ref,
                "relation": relation,
                "to": to_ref,
                "evidence_episode_id": evidence_episode_id,
                "delete_all_matching": delete_all_matching,
            },
        )

    async def delete_relation(
        self,
        *,
        from_ref: str,
        relation: str,
        to_ref: str,
        evidence_episode_id: str | None = None,
        delete_all_matching: bool = False,
        confirm: bool = True,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "delete_relation",
            {
                "from": from_ref,
                "relation": relation,
                "to": to_ref,
                "evidence_episode_id": evidence_episode_id,
                "delete_all_matching": delete_all_matching,
                "confirm": confirm,
            },
        )

    async def get_neighborhood(self, *, concept: str, depth: int = 1) -> dict[str, Any]:
        return await self.call_tool(
            "get_neighborhood",
            {"concept": concept, "depth": depth},
        )

    async def get_pedagogical_context(
        self,
        *,
        user_id: str,
        domain: str | None = None,
        concept_uids: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "get_pedagogical_context",
            {"user_id": user_id, "domain": domain, "concept_uids": concept_uids or []},
        )

    async def update_pedagogical_context(
        self,
        *,
        user_id: str,
        evaluations: list[dict[str, Any]],
        domain_hint: str | None = None,
        session_closed_at: str | None = None,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "update_pedagogical_context",
            {
                "user_id": user_id,
                "evaluations": evaluations,
                "domain_hint": domain_hint,
                "session_closed_at": session_closed_at,
            },
        )

    async def get_pedagogical_session_view(
        self,
        *,
        user_id: str,
        domain_hint: str | None = None,
        concept_uids: list[str] | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "get_pedagogical_session_view",
            {
                "user_id": user_id,
                "domain_hint": domain_hint,
                "concept_uids": concept_uids or [],
                "query": query,
            },
        )

    async def get_sr_state(
        self,
        *,
        user_id: str,
        concept_uid: str,
        dimension: str,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "get_sr_state",
            {"user_id": user_id, "concept_uid": concept_uid, "dimension": dimension},
        )

    async def get_due_sr_items(
        self,
        *,
        user_id: str,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "get_due_sr_items",
            {"user_id": user_id},
        )

    async def update_sr_from_block_result(
        self,
        *,
        user_id: str,
        concept_uid: str,
        dimension: str,
        block_verdict: str,
        block_difficulty: str,
        hint_used: bool = False,
        retry_used: bool = False,
        coverage: float = 0.0,
        precision: float = 0.0,
        was_direct_evaluation: bool = True,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "update_sr_from_block_result",
            {
                "user_id": user_id,
                "concept_uid": concept_uid,
                "dimension": dimension,
                "block_verdict": block_verdict,
                "block_difficulty": block_difficulty,
                "hint_used": hint_used,
                "retry_used": retry_used,
                "coverage": coverage,
                "precision": precision,
                "was_direct_evaluation": was_direct_evaluation,
            },
        )

    async def apply_prereq_relief(
        self,
        *,
        user_id: str,
        source_concept_uid: str,
        source_dimension: str,
        quality_q: int,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "apply_prereq_relief",
            {
                "user_id": user_id,
                "source_concept_uid": source_concept_uid,
                "source_dimension": source_dimension,
                "quality_q": quality_q,
            },
        )

    async def get_sr_stats(
        self,
        *,
        user_id: str,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "get_sr_stats",
            {"user_id": user_id},
        )

    async def start_adaptive_session(
        self,
        *,
        user_id: str,
        query: str | None = None,
        episode_id: str | None = None,
        job_id: str | None = None,
        study_mode: str = "hybrid",
        domain_hint: str | None = None,
        language: str = "es",
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "start_adaptive_session",
            {
                "user_id": user_id,
                "query": query,
                "episode_id": episode_id,
                "job_id": job_id,
                "study_mode": study_mode,
                "domain_hint": domain_hint,
                "language": language,
                "constraints": constraints or {},
            },
        )

    async def submit_adaptive_block(
        self,
        *,
        session_id: str,
        block_id: str,
        submissions: list[dict[str, Any]],
        interaction_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "submit_adaptive_block",
            {
                "session_id": session_id,
                "block_id": block_id,
                "submissions": submissions,
                "interaction_events": interaction_events or [],
            },
        )

    async def get_adaptive_session(self, *, session_id: str) -> dict[str, Any]:
        return await self.call_tool("get_adaptive_session", {"session_id": session_id})

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
            body = (exc.response.text or "").strip()
            logger.warning(
                "upstream MCP http error",
                extra={"status_code": exc.response.status_code, "body": body[:500]},
            )
            message = f"upstream MCP http error: {exc.response.status_code}"
            if body:
                message = f"{message}: {body[:500]}"
            raise AgentMCPUpstreamError(message, status_code=exc.response.status_code) from exc
        except httpx.HTTPError as exc:
            logger.error("upstream MCP transport failed", extra={"error": str(exc) or exc.__class__.__name__})
            raise AgentMCPUpstreamError(f"upstream MCP transport failed: {exc or exc.__class__.__name__}") from exc
        except Exception as exc:  # pragma: no cover - SDK-level failure
            logger.error("upstream MCP session failed", extra={"error": str(exc) or exc.__class__.__name__})
            raise AgentMCPUpstreamError(f"upstream MCP session failed: {exc or exc.__class__.__name__}") from exc

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
