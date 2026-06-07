from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Any, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import Field

from app.agent_content import AgentContentGenerator, build_agent_content_generator
from app.agent_mcp_client import AgentMCPUpstreamClient, AgentMCPUpstreamError
from app.agent_proxy import AgentProxyService
from app.config import Settings, get_settings


class AgentUpstreamProtocol(Protocol):
    async def check_ready(self) -> tuple[bool, dict[str, Any]]: ...

    async def close(self) -> None: ...

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]: ...

    async def add_knowledge_fragment(
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
        evaluations: list[dict[str, Any]],
        domain_hint: str | None = None,
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


class AgentContentProtocol(Protocol):
    async def close(self) -> None: ...

    async def explain_topic(self, **kwargs: Any) -> Any: ...

    async def generate_quiz(self, **kwargs: Any) -> Any: ...

    async def evaluate_answer(self, **kwargs: Any) -> Any: ...


def create_agent_mcp_server(
    *,
    settings: Settings | None = None,
    upstream_client: AgentUpstreamProtocol | None = None,
    content_generator: AgentContentProtocol | None = None,
) -> FastMCP:
    settings = settings or get_settings()
    upstream = upstream_client or AgentMCPUpstreamClient(settings)
    generator = content_generator or build_agent_content_generator(settings)
    service = AgentProxyService(upstream_client=upstream, content_generator=generator)

    mcp = FastMCP(
        "knowledge-graph-agent-mcp",
        instructions=(
            "Agent-facing MCP proxy over the internal knowledge graph MCP. "
            "Use explain_topic, generate_quiz and evaluate_answer for grounded pedagogy. "
            "Use kg_* tools for direct graph inspection, pedagogical context and curation."
        ),
        host="0.0.0.0",
        stateless_http=True,
        json_response=True,
        streamable_http_path="/mcp",
    )

    def translate_error(exc: AgentMCPUpstreamError) -> ToolError:
        return ToolError(str(exc))

    @mcp.tool(name="explain_topic")
    async def explain_topic(
        query: Annotated[str, Field(min_length=1)],
        domain_hint: str | None = None,
        audience: Annotated[str, Field(pattern="^(beginner|intermediate|advanced)$")] = "intermediate",
        focus: str | None = None,
        include_examples: bool = True,
    ) -> dict[str, Any]:
        try:
            return (await service.explain_topic(
                query=query,
                domain_hint=domain_hint,
                audience=audience,
                focus=focus,
                include_examples=include_examples,
            )).model_dump()
        except AgentMCPUpstreamError as exc:
            raise translate_error(exc) from exc

    @mcp.tool(name="generate_quiz")
    async def generate_quiz(
        query: Annotated[str, Field(min_length=1)],
        domain_hint: str | None = None,
        difficulty: Annotated[str, Field(pattern="^(beginner|intermediate|advanced)$")] = "intermediate",
        question_count: Annotated[int, Field(ge=1, le=10)] = 5,
        question_type: Annotated[str, Field(pattern="^(mixed|multiple_choice|open)$")] = "mixed",
    ) -> dict[str, Any]:
        try:
            return (await service.generate_quiz(
                query=query,
                domain_hint=domain_hint,
                difficulty=difficulty,
                question_count=question_count,
                question_type=question_type,
            )).model_dump()
        except AgentMCPUpstreamError as exc:
            raise translate_error(exc) from exc

    @mcp.tool(name="evaluate_answer")
    async def evaluate_answer(
        query: Annotated[str, Field(min_length=1)],
        question: Annotated[str, Field(min_length=1)],
        learner_answer: Annotated[str, Field(min_length=1)],
        domain_hint: str | None = None,
        expected_answer: str | None = None,
    ) -> dict[str, Any]:
        try:
            return (await service.evaluate_answer(
                query=query,
                question=question,
                learner_answer=learner_answer,
                domain_hint=domain_hint,
                expected_answer=expected_answer,
            )).model_dump()
        except AgentMCPUpstreamError as exc:
            raise translate_error(exc) from exc

    @mcp.tool(name="kg_add_knowledge_fragment")
    async def kg_add_knowledge_fragment(
        text: Annotated[str, Field(min_length=1)],
        source_type: str = "manual_input",
        tags: list[str] | None = None,
        language: str = "es",
    ) -> dict[str, Any]:
        try:
            return await upstream.add_knowledge_fragment(
                text=text,
                source_type=source_type,
                tags=tags,
                language=language,
            )
        except AgentMCPUpstreamError as exc:
            raise translate_error(exc) from exc

    @mcp.tool(name="kg_search_candidates")
    async def kg_search_candidates(
        query: Annotated[str, Field(min_length=1)],
        domain_hint: str | None = None,
        limit: Annotated[int, Field(ge=1, le=50)] = 10,
    ) -> dict[str, Any]:
        try:
            return await upstream.search_candidates(query=query, domain_hint=domain_hint, limit=limit)
        except AgentMCPUpstreamError as exc:
            raise translate_error(exc) from exc

    @mcp.tool(name="kg_get_learning_context")
    async def kg_get_learning_context(
        query: Annotated[str, Field(min_length=1)],
        domain_hint: str | None = None,
        candidate_limit: Annotated[int, Field(ge=1, le=50)] = 8,
        concept_limit: Annotated[int, Field(ge=1, le=10)] = 3,
        claim_limit: Annotated[int, Field(ge=1, le=20)] = 6,
        episode_limit: Annotated[int, Field(ge=1, le=10)] = 3,
        include_neighborhood: bool = True,
        depth: Annotated[int, Field(ge=1, le=2)] = 1,
    ) -> dict[str, Any]:
        try:
            return await upstream.get_learning_context(
                query=query,
                domain_hint=domain_hint,
                candidate_limit=candidate_limit,
                concept_limit=concept_limit,
                claim_limit=claim_limit,
                episode_limit=episode_limit,
                include_neighborhood=include_neighborhood,
                depth=depth,
            )
        except AgentMCPUpstreamError as exc:
            raise translate_error(exc) from exc

    @mcp.tool(name="kg_get_tutor_context")
    async def kg_get_tutor_context(
        query: str | None = None,
        episode_id: str | None = None,
        job_id: str | None = None,
        depth: Annotated[int, Field(ge=1, le=1)] = 1,
        include_evidence: bool = True,
    ) -> dict[str, Any]:
        try:
            return await upstream.get_tutor_context(
                query=query,
                episode_id=episode_id,
                job_id=job_id,
                depth=depth,
                include_evidence=include_evidence,
            )
        except AgentMCPUpstreamError as exc:
            raise translate_error(exc) from exc

    @mcp.tool(name="kg_upsert_concept")
    async def kg_upsert_concept(
        canonical_name: Annotated[str, Field(min_length=1)],
        domain: Annotated[str, Field(min_length=1)],
        aliases: list[str] | None = None,
        description: str = "",
        uid: str | None = None,
    ) -> dict[str, Any]:
        try:
            return await upstream.upsert_concept(
                uid=uid,
                canonical_name=canonical_name,
                aliases=aliases,
                domain=domain,
                description=description,
            )
        except AgentMCPUpstreamError as exc:
            raise translate_error(exc) from exc

    @mcp.tool(name="kg_create_concept")
    async def kg_create_concept(
        canonical_name: Annotated[str, Field(min_length=1)],
        domain: Annotated[str, Field(min_length=1)],
        aliases: list[str] | None = None,
        description: str = "",
    ) -> dict[str, Any]:
        try:
            return await upstream.create_concept(
                canonical_name=canonical_name,
                aliases=aliases,
                domain=domain,
                description=description,
            )
        except AgentMCPUpstreamError as exc:
            raise translate_error(exc) from exc

    @mcp.tool(name="kg_link_concepts")
    async def kg_link_concepts(
        from_: Annotated[str, Field(min_length=1)],
        relation: Annotated[str, Field(min_length=1)],
        to: Annotated[str, Field(min_length=1)],
        evidence_episode_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            return await upstream.link_concepts(
                from_ref=from_,
                relation=relation,
                to_ref=to,
                evidence_episode_id=evidence_episode_id,
            )
        except AgentMCPUpstreamError as exc:
            raise translate_error(exc) from exc

    @mcp.tool(name="kg_get_neighborhood")
    async def kg_get_neighborhood(
        concept: Annotated[str, Field(min_length=1)],
        depth: Annotated[int, Field(ge=1, le=2)] = 1,
    ) -> dict[str, Any]:
        try:
            return await upstream.get_neighborhood(concept=concept, depth=depth)
        except AgentMCPUpstreamError as exc:
            raise translate_error(exc) from exc

    @mcp.tool(name="kg_get_pedagogical_context")
    async def kg_get_pedagogical_context(
        user_id: Annotated[str, Field(min_length=1)],
        domain: str | None = None,
        concept_uids: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            return await upstream.get_pedagogical_context(
                user_id=user_id,
                domain=domain,
                concept_uids=concept_uids,
            )
        except AgentMCPUpstreamError as exc:
            raise translate_error(exc) from exc

    @mcp.tool(name="kg_update_pedagogical_context")
    async def kg_update_pedagogical_context(
        user_id: Annotated[str, Field(min_length=1)],
        evaluations: list[dict[str, Any]],
        domain_hint: str | None = None,
        session_closed_at: str | None = None,
    ) -> dict[str, Any]:
        try:
            return await upstream.update_pedagogical_context(
                user_id=user_id,
                evaluations=evaluations,
                domain_hint=domain_hint,
                session_closed_at=session_closed_at,
            )
        except AgentMCPUpstreamError as exc:
            raise translate_error(exc) from exc

    @mcp.tool(name="kg_get_pedagogical_session_view")
    async def kg_get_pedagogical_session_view(
        user_id: Annotated[str, Field(min_length=1)],
        domain_hint: str | None = None,
        concept_uids: list[str] | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        try:
            return await upstream.get_pedagogical_session_view(
                user_id=user_id,
                domain_hint=domain_hint,
                concept_uids=concept_uids,
                query=query,
            )
        except AgentMCPUpstreamError as exc:
            raise translate_error(exc) from exc

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
        try:
            return await upstream.start_adaptive_session(
                user_id=user_id,
                query=query,
                episode_id=episode_id,
                job_id=job_id,
                domain_hint=domain_hint,
                language=language,
                constraints=constraints,
            )
        except AgentMCPUpstreamError as exc:
            raise translate_error(exc) from exc

    @mcp.tool(name="submit_adaptive_block")
    async def submit_adaptive_block(
        session_id: Annotated[str, Field(min_length=1)],
        block_id: Annotated[str, Field(min_length=1)],
        submissions: list[dict[str, Any]],
        interaction_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        try:
            return await upstream.submit_adaptive_block(
                session_id=session_id,
                block_id=block_id,
                submissions=submissions,
                interaction_events=interaction_events,
            )
        except AgentMCPUpstreamError as exc:
            raise translate_error(exc) from exc

    @mcp.tool(name="get_adaptive_session")
    async def get_adaptive_session(
        session_id: Annotated[str, Field(min_length=1)],
    ) -> dict[str, Any]:
        try:
            return await upstream.get_adaptive_session(session_id=session_id)
        except AgentMCPUpstreamError as exc:
            raise translate_error(exc) from exc

    link_tool = mcp._tool_manager.get_tool("kg_link_concepts")
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


def create_agent_app(
    *,
    settings: Settings | None = None,
    upstream_client: AgentUpstreamProtocol | None = None,
    content_generator: AgentContentProtocol | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    upstream = upstream_client or AgentMCPUpstreamClient(settings)
    generator = content_generator or build_agent_content_generator(settings)
    mcp = create_agent_mcp_server(
        settings=settings,
        upstream_client=upstream,
        content_generator=generator,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        async with mcp.session_manager.run():
            try:
                yield
            finally:
                if upstream_client is None:
                    await upstream.close()
                if content_generator is None and hasattr(generator, "close"):
                    await generator.close()

    app = FastAPI(title="knowledge-graph-agent-mcp", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def require_mcp_bearer(request: Request, call_next):
        path = request.url.path
        if path == "/mcp" or path.startswith("/mcp/"):
            expected = f"Bearer {settings.agent_mcp_bearer_token}"
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
        is_ready, payload = await upstream.check_ready()
        return JSONResponse(status_code=200 if is_ready else 503, content=payload)

    app.mount("/", mcp.streamable_http_app())
    return app


app = create_agent_app()
