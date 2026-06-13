from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


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
        except httpx.HTTPError as exc:
            logger.warning("backend readiness check transport error", extra={"error": str(exc) or exc.__class__.__name__})
            return False, {"status": "degraded", "detail": f"backend transport error: {exc or exc.__class__.__name__}"}

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
        temporal: bool = False,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/knowledge/fragments",
            json={
                "text": text,
                "source_type": source_type,
                "tags": tags or [],
                "language": language,
                "temporal": temporal,
                "expires_at": expires_at,
            },
        )

    async def patch_episode_temporality(
        self,
        episode_id: str,
        *,
        temporal: bool,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "PATCH",
            f"/v1/episodes/{episode_id}/temporality",
            json={"temporal": temporal, "expires_at": expires_at},
        )

    async def reset_knowledge_base(self) -> dict[str, Any]:
        return await self._request("POST", "/v1/knowledge/reset")

    async def create_concept(
        self,
        *,
        canonical_name: str,
        aliases: list[str] | None = None,
        domain: str,
        description: str = "",
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/concepts",
            json={
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
        return await self._request(
            "POST",
            "/v1/concepts/evidence",
            json={
                "concept_ref": concept_ref,
                "episode_id": episode_id,
                "link_episode_claims": link_episode_claims,
            },
        )

    async def get_job(self, job_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/jobs/{job_id}")

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
        params: dict[str, Any] = {
            "sort_by": sort_by,
            "sort_order": sort_order,
            "limit": limit,
            "page": page,
            "concept_sort_order": concept_sort_order,
        }
        if concept_sort_by is not None:
            params["concept_sort_by"] = concept_sort_by
        return await self._request("GET", "/v1/episodes", params=params)

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
        return await self._request(
            "POST",
            "/v1/search/learning-context",
            json={
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
        return await self._request(
            "POST",
            "/v1/search/tutor-context",
            json={
                "query": query,
                "episode_id": episode_id,
                "job_id": job_id,
                "depth": depth,
                "include_evidence": include_evidence,
            },
        )

    async def get_pedagogical_context(
        self,
        *,
        user_id: str,
        domain: str | None = None,
        concept_uids: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/pedagogical/context",
            json={"user_id": user_id, "domain": domain, "concept_uids": concept_uids or []},
        )

    async def update_pedagogical_context(
        self,
        *,
        user_id: str,
        domain_hint: str | None = None,
        evaluations: list[dict[str, Any]],
        session_closed_at: str | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/pedagogical/update-from-evaluation",
            json={
                "user_id": user_id,
                "domain_hint": domain_hint,
                "evaluations": evaluations,
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
        return await self._request(
            "POST",
            "/v1/pedagogical/session-view",
            json={
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
        return await self._request(
            "GET",
            "/v1/sr/state",
            params={"user_id": user_id, "concept_uid": concept_uid, "dimension": dimension},
        )

    async def get_due_sr_items(
        self,
        *,
        user_id: str,
    ) -> dict[str, Any]:
        return await self._request(
            "GET",
            "/v1/sr/due",
            params={"user_id": user_id},
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
        return await self._request(
            "POST",
            "/v1/sr/update",
            json={
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
        return await self._request(
            "POST",
            "/v1/sr/relief",
            json={
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
        return await self._request(
            "GET",
            "/v1/sr/stats",
            params={"user_id": user_id},
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
        return await self._request(
            "POST",
            "/v1/adaptive/sessions/start",
            json={
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
        return await self._request(
            "POST",
            f"/v1/adaptive/sessions/{session_id}/submit",
            json={
                "block_id": block_id,
                "submissions": submissions,
                "interaction_events": interaction_events or [],
            },
        )

    async def get_adaptive_session(self, *, session_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/adaptive/sessions/{session_id}")

    async def upsert_concept(
        self,
        *,
        uid: str | None = None,
        canonical_name: str,
        aliases: list[str] | None = None,
        domain: str,
        description: str = "",
    ) -> dict[str, Any]:
        return await self._request(
            "PUT",
            "/v1/concepts/upsert",
            json={
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

    async def preview_delete_episode_content(
        self,
        *,
        episode_id: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/deletions/episode/preview",
            json={"episode_id": episode_id, "job_id": job_id},
        )

    async def delete_episode_content(
        self,
        *,
        episode_id: str | None = None,
        job_id: str | None = None,
        confirm: bool = True,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/deletions/episode/execute",
            json={"episode_id": episode_id, "job_id": job_id, "confirm": confirm},
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
        return await self._request(
            "POST",
            "/v1/deletions/relation/preview",
            json={
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
        return await self._request(
            "POST",
            "/v1/deletions/relation/execute",
            json={
                "from": from_ref,
                "relation": relation,
                "to": to_ref,
                "evidence_episode_id": evidence_episode_id,
                "delete_all_matching": delete_all_matching,
                "confirm": confirm,
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
        temporal: bool = False,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        accepted = await self.add_knowledge_fragment(
            text=text,
            source_type=source_type,
            tags=tags,
            language=language,
            temporal=temporal,
            expires_at=expires_at,
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
            logger.error(
                "backend transport error",
                extra={"method": method, "path": path, "error": str(exc) or exc.__class__.__name__},
            )
            raise MCPBackendError(f"backend transport error: {exc or exc.__class__.__name__}") from exc

        if response.status_code < 400:
            return response.json()

        detail = self._extract_error_detail(response)
        logger.warning(
            "backend request failed",
            extra={"method": method, "path": path, "status_code": response.status_code, "detail": detail},
        )
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
            if isinstance(detail, list):
                # FastAPI validation errors: [{"loc": [...], "msg": "...", ...}, ...]
                summary = MCPBackendClient._summarize_validation_errors(detail)
                if summary:
                    return summary
            if isinstance(detail, dict):
                return json.dumps(detail, ensure_ascii=False)
            error = payload.get("error")
            if isinstance(error, str):
                return error
            if detail is not None:
                return json.dumps(detail, ensure_ascii=False, default=str)
        return response.text.strip() or None

    @staticmethod
    def _summarize_validation_errors(errors: list[Any]) -> str | None:
        parts: list[str] = []
        for item in errors:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            loc = item.get("loc")
            msg = item.get("msg") or item.get("type") or "invalid"
            if isinstance(loc, (list, tuple)):
                location = ".".join(str(segment) for segment in loc)
            else:
                location = str(loc) if loc is not None else ""
            parts.append(f"{location}: {msg}" if location else str(msg))
        return "; ".join(parts) or None

    async def _sleep(self, seconds: float) -> None:
        import asyncio

        await asyncio.sleep(seconds)
