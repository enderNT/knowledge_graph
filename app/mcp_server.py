from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Any, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import Field

from app.config import Settings, get_settings
from app.mcp_backend_client import MCPBackendClient, MCPBackendError


class MCPBackendProtocol(Protocol):
    async def add_knowledge_fragment(
        self,
        *,
        text: str,
        source_type: str = "manual_input",
        tags: list[str] | None = None,
        language: str = "es",
    ) -> dict[str, Any]: ...

    async def ingest_fragment_and_wait(
        self,
        *,
        text: str,
        source_type: str = "manual_input",
        tags: list[str] | None = None,
        language: str = "es",
    ) -> dict[str, Any]: ...

    async def reset_knowledge_base(self) -> dict[str, Any]: ...

    async def list_episodes(
        self,
        *,
        sort_by: str = "alphabetical",
        sort_order: str = "asc",
        limit: int = 10,
        page: int = 1,
        concept_sort_by: str | None = None,
        concept_sort_order: str = "asc",
    ) -> dict[str, Any]: ...

    async def search_candidates(
        self,
        *,
        query: str,
        domain_hint: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]: ...

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
    ) -> dict[str, Any]: ...

    async def get_tutor_context(
        self,
        *,
        query: str | None = None,
        episode_id: str | None = None,
        job_id: str | None = None,
        depth: int = 1,
        include_evidence: bool = True,
    ) -> dict[str, Any]: ...

    async def create_concept(
        self,
        *,
        canonical_name: str,
        aliases: list[str] | None = None,
        domain: str,
        description: str = "",
    ) -> dict[str, Any]: ...

    async def attach_concept_evidence(
        self,
        *,
        concept_ref: str,
        episode_id: str,
        link_episode_claims: bool = True,
    ) -> dict[str, Any]: ...

    async def upsert_concept(
        self,
        *,
        uid: str | None = None,
        canonical_name: str,
        aliases: list[str] | None = None,
        domain: str,
        description: str = "",
    ) -> dict[str, Any]: ...

    async def link_concepts(
        self,
        *,
        from_ref: str,
        relation: str,
        to_ref: str,
        evidence_episode_id: str | None = None,
    ) -> dict[str, Any]: ...

    async def preview_delete_episode_content(
        self,
        *,
        episode_id: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]: ...

    async def delete_episode_content(
        self,
        *,
        episode_id: str | None = None,
        job_id: str | None = None,
        confirm: bool = True,
    ) -> dict[str, Any]: ...

    async def preview_delete_relation(
        self,
        *,
        from_ref: str,
        relation: str,
        to_ref: str,
        evidence_episode_id: str | None = None,
        delete_all_matching: bool = False,
    ) -> dict[str, Any]: ...

    async def delete_relation(
        self,
        *,
        from_ref: str,
        relation: str,
        to_ref: str,
        evidence_episode_id: str | None = None,
        delete_all_matching: bool = False,
        confirm: bool = True,
    ) -> dict[str, Any]: ...

    async def get_neighborhood(self, *, concept: str, depth: int = 1) -> dict[str, Any]: ...

    async def get_pedagogical_context(
        self,
        *,
        user_id: str,
        domain: str | None = None,
        concept_uids: list[str] | None = None,
    ) -> dict[str, Any]: ...

    async def update_pedagogical_context(
        self,
        *,
        user_id: str,
        domain_hint: str | None = None,
        evaluations: list[dict[str, Any]],
        session_closed_at: str | None = None,
    ) -> dict[str, Any]: ...

    async def get_pedagogical_session_view(
        self,
        *,
        user_id: str,
        domain_hint: str | None = None,
        concept_uids: list[str] | None = None,
        query: str | None = None,
    ) -> dict[str, Any]: ...

    async def get_sr_state(
        self,
        *,
        user_id: str,
        concept_uid: str,
        dimension: str,
    ) -> dict[str, Any]: ...

    async def get_due_sr_items(
        self,
        *,
        user_id: str,
    ) -> dict[str, Any]: ...

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
    ) -> dict[str, Any]: ...

    async def apply_prereq_relief(
        self,
        *,
        user_id: str,
        source_concept_uid: str,
        source_dimension: str,
        quality_q: int,
    ) -> dict[str, Any]: ...

    async def get_sr_stats(
        self,
        *,
        user_id: str,
    ) -> dict[str, Any]: ...

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
    ) -> dict[str, Any]: ...

    async def submit_adaptive_block(
        self,
        *,
        session_id: str,
        block_id: str,
        submissions: list[dict[str, Any]],
        interaction_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]: ...

    async def get_adaptive_session(self, *, session_id: str) -> dict[str, Any]: ...

    async def check_ready(self) -> tuple[bool, dict[str, Any]]: ...

    async def close(self) -> None: ...


def create_mcp_server(
    *,
    settings: Settings | None = None,
    backend_client: MCPBackendProtocol | None = None,
) -> FastMCP:
    settings = settings or get_settings()
    backend = backend_client or MCPBackendClient(settings)

    mcp = FastMCP(
        "knowledge-graph-mcp",
        instructions="Knowledge graph semantico con tools de ingesta, busqueda, contexto recuperado, tutor, vecindad y contexto pedagogico persistente del usuario.",
        host="0.0.0.0",
        stateless_http=True,
        json_response=True,
        streamable_http_path="/mcp",
    )

    def translate_backend_error(exc: MCPBackendError) -> ToolError:
        return ToolError(str(exc))

    @mcp.tool(name="add_knowledge_fragment")
    async def add_knowledge_fragment(
        text: Annotated[str, Field(min_length=1)],
        source_type: str = "manual_input",
        tags: list[str] | None = None,
        language: str = "es",
    ) -> dict[str, Any]:
        """Ingest a text fragment and wait for a semantic job summary when possible."""
        try:
            return await backend.ingest_fragment_and_wait(
                text=text,
                source_type=source_type,
                tags=tags,
                language=language,
            )
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="reset_knowledge_base")
    async def reset_knowledge_base() -> dict[str, Any]:
        """Completely reset the stored graph, pedagogical state, sessions and queued ingestion payloads."""
        try:
            return await backend.reset_knowledge_base()
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="list_episodes")
    async def list_episodes(
        sort_by: Annotated[str, Field(pattern="^(alphabetical|date)$")] = "alphabetical",
        sort_order: Annotated[str, Field(pattern="^(asc|desc)$")] = "asc",
        limit: Annotated[int, Field(ge=1, le=100)] = 10,
        page: Annotated[int, Field(ge=1)] = 1,
        concept_sort_by: Annotated[str | None, Field(pattern="^(alphabetical|date)$")] = None,
        concept_sort_order: Annotated[str, Field(pattern="^(asc|desc)$")] = "asc",
    ) -> dict[str, Any]:
        """List all ingested episodes with pagination, sorting, and concept summaries per episode."""
        try:
            return await backend.list_episodes(
                sort_by=sort_by,
                sort_order=sort_order,
                limit=limit,
                page=page,
                concept_sort_by=concept_sort_by,
                concept_sort_order=concept_sort_order,
            )
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="search_candidates")
    async def search_candidates(
        query: Annotated[str, Field(min_length=1)],
        domain_hint: str | None = None,
        limit: Annotated[int, Field(ge=1, le=50)] = 10,
    ) -> dict[str, Any]:
        """Search candidate concepts related to a query."""
        try:
            return await backend.search_candidates(query=query, domain_hint=domain_hint, limit=limit)
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="get_learning_context")
    async def get_learning_context(
        query: Annotated[str, Field(min_length=1)],
        domain_hint: str | None = None,
        candidate_limit: Annotated[int, Field(ge=1, le=50)] = 8,
        concept_limit: Annotated[int, Field(ge=1, le=10)] = 3,
        claim_limit: Annotated[int, Field(ge=1, le=20)] = 6,
        episode_limit: Annotated[int, Field(ge=1, le=10)] = 3,
        include_neighborhood: bool = True,
        depth: Annotated[int, Field(ge=1, le=2)] = 1,
    ) -> dict[str, Any]:
        """Fetch deterministic learning-ready context built from search candidates and graph neighborhood."""
        try:
            return await backend.get_learning_context(
                query=query,
                domain_hint=domain_hint,
                candidate_limit=candidate_limit,
                concept_limit=concept_limit,
                claim_limit=claim_limit,
                episode_limit=episode_limit,
                include_neighborhood=include_neighborhood,
                depth=depth,
            )
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="get_tutor_context")
    async def get_tutor_context(
        query: str | None = None,
        episode_id: str | None = None,
        job_id: str | None = None,
        depth: Annotated[int, Field(ge=1, le=1)] = 1,
        include_evidence: bool = True,
    ) -> dict[str, Any]:
        """Fetch strict tutor-ready context from exactly one input reference."""
        try:
            return await backend.get_tutor_context(
                query=query,
                episode_id=episode_id,
                job_id=job_id,
                depth=depth,
                include_evidence=include_evidence,
            )
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="create_concept")
    async def create_concept(
        canonical_name: Annotated[str, Field(min_length=1)],
        domain: Annotated[str, Field(min_length=1)],
        aliases: list[str] | None = None,
        description: str = "",
    ) -> dict[str, Any]:
        """Create a concept strictly and fail on name or alias collisions."""
        try:
            return await backend.create_concept(
                canonical_name=canonical_name,
                aliases=aliases,
                domain=domain,
                description=description,
            )
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="attach_concept_evidence")
    async def attach_concept_evidence(
        concept_ref: Annotated[str, Field(min_length=1)],
        episode_id: Annotated[str, Field(min_length=1)],
        link_episode_claims: bool = True,
    ) -> dict[str, Any]:
        """Attach an episode as explicit evidence for a curated concept and optionally link episode claims."""
        try:
            return await backend.attach_concept_evidence(
                concept_ref=concept_ref,
                episode_id=episode_id,
                link_episode_claims=link_episode_claims,
            )
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="upsert_concept")
    async def upsert_concept(
        canonical_name: Annotated[str, Field(min_length=1)],
        domain: Annotated[str, Field(min_length=1)],
        aliases: list[str] | None = None,
        description: str = "",
        uid: str | None = None,
    ) -> dict[str, Any]:
        """Create or update a concept by exact uid or normalized canonical name."""
        try:
            return await backend.upsert_concept(
                uid=uid,
                canonical_name=canonical_name,
                aliases=aliases,
                domain=domain,
                description=description,
            )
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="link_concepts")
    async def link_concepts(
        from_: Annotated[str, Field(min_length=1)],
        relation: Annotated[str, Field(min_length=1)],
        to: Annotated[str, Field(min_length=1)],
        evidence_episode_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a semantic relation between two concepts."""
        try:
            return await backend.link_concepts(
                from_ref=from_,
                relation=relation,
                to_ref=to,
                evidence_episode_id=evidence_episode_id,
            )
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="preview_delete_episode_content")
    async def preview_delete_episode_content(
        episode_id: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Preview the exact impact of deleting one ingested episode or its ingestion job."""
        try:
            return await backend.preview_delete_episode_content(episode_id=episode_id, job_id=job_id)
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="delete_episode_content")
    async def delete_episode_content(
        episode_id: str | None = None,
        job_id: str | None = None,
        confirm: bool = True,
    ) -> dict[str, Any]:
        """Delete one ingested episode or job after an explicit confirmation step in the client."""
        try:
            return await backend.delete_episode_content(episode_id=episode_id, job_id=job_id, confirm=confirm)
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="preview_delete_relation")
    async def preview_delete_relation(
        from_: Annotated[str, Field(min_length=1)],
        relation: Annotated[str, Field(min_length=1)],
        to: Annotated[str, Field(min_length=1)],
        evidence_episode_id: str | None = None,
        delete_all_matching: bool = False,
    ) -> dict[str, Any]:
        """Preview the exact impact of deleting one concept-to-concept relation instance or all matches."""
        try:
            return await backend.preview_delete_relation(
                from_ref=from_,
                relation=relation,
                to_ref=to,
                evidence_episode_id=evidence_episode_id,
                delete_all_matching=delete_all_matching,
            )
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="delete_relation")
    async def delete_relation(
        from_: Annotated[str, Field(min_length=1)],
        relation: Annotated[str, Field(min_length=1)],
        to: Annotated[str, Field(min_length=1)],
        evidence_episode_id: str | None = None,
        delete_all_matching: bool = False,
        confirm: bool = True,
    ) -> dict[str, Any]:
        """Delete one concept-to-concept relation instance or all matching instances after confirmation."""
        try:
            return await backend.delete_relation(
                from_ref=from_,
                relation=relation,
                to_ref=to,
                evidence_episode_id=evidence_episode_id,
                delete_all_matching=delete_all_matching,
                confirm=confirm,
            )
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="get_neighborhood")
    async def get_neighborhood(
        concept: Annotated[str, Field(min_length=1)],
        depth: Annotated[int, Field(ge=1, le=2)] = 1,
    ) -> dict[str, Any]:
        """Fetch the graph neighborhood for a concept."""
        try:
            return await backend.get_neighborhood(concept=concept, depth=depth)
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="get_pedagogical_context")
    async def get_pedagogical_context(
        user_id: Annotated[str, Field(min_length=1)],
        domain: str | None = None,
        concept_uids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch persisted pedagogical context for a user."""
        try:
            return await backend.get_pedagogical_context(
                user_id=user_id,
                domain=domain,
                concept_uids=concept_uids,
            )
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="update_pedagogical_context")
    async def update_pedagogical_context(
        user_id: Annotated[str, Field(min_length=1)],
        evaluations: list[dict[str, Any]],
        domain_hint: str | None = None,
        session_closed_at: str | None = None,
    ) -> dict[str, Any]:
        """Apply formal evaluation results to the persisted pedagogical context."""
        try:
            return await backend.update_pedagogical_context(
                user_id=user_id,
                domain_hint=domain_hint,
                evaluations=evaluations,
                session_closed_at=session_closed_at,
            )
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="get_pedagogical_session_view")
    async def get_pedagogical_session_view(
        user_id: Annotated[str, Field(min_length=1)],
        domain_hint: str | None = None,
        concept_uids: list[str] | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Build the operational pedagogical session view for a user."""
        try:
            return await backend.get_pedagogical_session_view(
                user_id=user_id,
                domain_hint=domain_hint,
                concept_uids=concept_uids,
                query=query,
            )
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="get_sr_state")
    async def get_sr_state(
        user_id: Annotated[str, Field(min_length=1)],
        concept_uid: Annotated[str, Field(min_length=1)],
        dimension: Annotated[str, Field(pattern="^(recognition|recall|explanation|application)$")],
    ) -> dict[str, Any]:
        """Fetch spaced repetition state for one concept-dimension."""
        try:
            return await backend.get_sr_state(
                user_id=user_id,
                concept_uid=concept_uid,
                dimension=dimension,
            )
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="get_due_sr_items")
    async def get_due_sr_items(
        user_id: Annotated[str, Field(min_length=1)],
    ) -> dict[str, Any]:
        """Fetch sorted due spaced repetition items for a user."""
        try:
            return await backend.get_due_sr_items(user_id=user_id)
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="update_sr_from_block_result")
    async def update_sr_from_block_result(
        user_id: Annotated[str, Field(min_length=1)],
        concept_uid: Annotated[str, Field(min_length=1)],
        dimension: Annotated[str, Field(pattern="^(recognition|recall|explanation|application)$")],
        block_verdict: Annotated[str, Field(pattern="^(correct|partial_high|partial_low|incorrect|unsupported)$")],
        block_difficulty: Annotated[str, Field(pattern="^(introductory|intermediate|advanced)$")],
        hint_used: bool = False,
        retry_used: bool = False,
        coverage: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0,
        precision: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0,
        was_direct_evaluation: bool = True,
    ) -> dict[str, Any]:
        """Apply deterministic q mapping and persist updated SR state."""
        try:
            return await backend.update_sr_from_block_result(
                user_id=user_id,
                concept_uid=concept_uid,
                dimension=dimension,
                block_verdict=block_verdict,
                block_difficulty=block_difficulty,
                hint_used=hint_used,
                retry_used=retry_used,
                coverage=coverage,
                precision=precision,
                was_direct_evaluation=was_direct_evaluation,
            )
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="apply_prereq_relief")
    async def apply_prereq_relief(
        user_id: Annotated[str, Field(min_length=1)],
        source_concept_uid: Annotated[str, Field(min_length=1)],
        source_dimension: Annotated[str, Field(pattern="^(recognition|recall|explanation|application)$")],
        quality_q: Annotated[int, Field(ge=0, le=5)],
    ) -> dict[str, Any]:
        """Apply prerequisite review relief from child to parent concepts."""
        try:
            return await backend.apply_prereq_relief(
                user_id=user_id,
                source_concept_uid=source_concept_uid,
                source_dimension=source_dimension,
                quality_q=quality_q,
            )
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="get_sr_stats")
    async def get_sr_stats(
        user_id: Annotated[str, Field(min_length=1)],
    ) -> dict[str, Any]:
        """Fetch aggregate spaced repetition statistics for a user."""
        try:
            return await backend.get_sr_stats(user_id=user_id)
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="start_adaptive_session")
    async def start_adaptive_session(
        user_id: Annotated[str, Field(min_length=1)],
        query: str | None = None,
        episode_id: str | None = None,
        job_id: str | None = None,
        study_mode: Annotated[str, Field(pattern="^(hybrid|backlog|recovery|isolated)$")] = "hybrid",
        domain_hint: str | None = None,
        language: str = "es",
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start an adaptive pedagogical session grounded on strict tutor context."""
        try:
            return await backend.start_adaptive_session(
                user_id=user_id,
                query=query,
                episode_id=episode_id,
                job_id=job_id,
                study_mode=study_mode,
                domain_hint=domain_hint,
                language=language,
                constraints=constraints,
            )
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="submit_adaptive_block")
    async def submit_adaptive_block(
        session_id: Annotated[str, Field(min_length=1)],
        block_id: Annotated[str, Field(min_length=1)],
        submissions: list[dict[str, Any]],
        interaction_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Submit learner responses for the current adaptive block and replan."""
        try:
            return await backend.submit_adaptive_block(
                session_id=session_id,
                block_id=block_id,
                submissions=submissions,
                interaction_events=interaction_events,
            )
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    @mcp.tool(name="get_adaptive_session")
    async def get_adaptive_session(
        session_id: Annotated[str, Field(min_length=1)],
    ) -> dict[str, Any]:
        """Fetch the persisted adaptive session snapshot."""
        try:
            return await backend.get_adaptive_session(session_id=session_id)
        except MCPBackendError as exc:
            raise translate_backend_error(exc) from exc

    for tool_name in ("link_concepts", "preview_delete_relation", "delete_relation"):
        tool = mcp._tool_manager.get_tool(tool_name)
        if tool is None:
            continue
        original_run = tool.run

        async def run_tool(arguments: dict[str, Any], *args: Any, __original_run: Any = original_run, **kwargs: Any) -> Any:
            translated = dict(arguments)
            if "from" in translated and "from_" not in translated:
                translated["from_"] = translated.pop("from")
            return await __original_run(translated, *args, **kwargs)

        object.__setattr__(tool, "run", run_tool)

        properties = tool.parameters.get("properties", {})
        if "from_" in properties:
            properties["from"] = properties.pop("from_")
        required = tool.parameters.get("required")
        if isinstance(required, list):
            tool.parameters["required"] = ["from" if item == "from_" else item for item in required]

    return mcp


def create_app(
    *,
    settings: Settings | None = None,
    backend_client: MCPBackendProtocol | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    backend = backend_client or MCPBackendClient(settings)
    mcp = create_mcp_server(settings=settings, backend_client=backend)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        async with mcp.session_manager.run():
            try:
                yield
            finally:
                if backend_client is None:
                    await backend.close()

    app = FastAPI(title="knowledge-graph-mcp", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def require_mcp_bearer(request: Request, call_next):
        path = request.url.path
        if path == "/mcp" or path.startswith("/mcp/"):
            expected = f"Bearer {settings.mcp_bearer_token}"
            if request.headers.get("Authorization") != expected:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "invalid bearer token"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await call_next(request)

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready() -> JSONResponse:
        is_ready, payload = await backend.check_ready()
        return JSONResponse(status_code=200 if is_ready else 503, content=payload)

    app.mount("/", mcp.streamable_http_app())
    return app


app = create_app()
