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

    async def start_adaptive_session(
        self,
        *,
        user_id: str,
        query: str | None = None,
        episode_id: str | None = None,
        job_id: str | None = None,
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

    @mcp.tool(name="start_adaptive_session")
    async def start_adaptive_session(
        user_id: Annotated[str, Field(min_length=1)],
        query: str | None = None,
        episode_id: str | None = None,
        job_id: str | None = None,
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

    link_tool = mcp._tool_manager.get_tool("link_concepts")
    if link_tool is not None:
        original_run = link_tool.run

        async def run_link_tool(arguments: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            translated = dict(arguments)
            if "from" in translated and "from_" not in translated:
                translated["from_"] = translated.pop("from")
            return await original_run(translated, *args, **kwargs)

        object.__setattr__(link_tool, "run", run_link_tool)

        properties = link_tool.parameters.get("properties", {})
        if "from_" in properties:
            properties["from"] = properties.pop("from_")
        required = link_tool.parameters.get("required")
        if isinstance(required, list):
            link_tool.parameters["required"] = ["from" if item == "from_" else item for item in required]

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
