from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request

from app.ai_provider import AIProvider, build_ai_provider
from app.arcadedb_client import ArcadeDBClient
from app.config import Settings, get_settings
from app.ingestion import IngestionService
from app.logging_config import setup_logging
from app.store import ArcadeKnowledgeStore, InMemoryKnowledgeStore, KnowledgeStore
from app.trace import bind, clear
from app.worker import IngestionWorker

logger = logging.getLogger(__name__)


@dataclass
class AppServices:
    settings: Settings
    store: KnowledgeStore
    ai_provider: AIProvider
    ingestion_service: IngestionService
    worker: IngestionWorker
    queue: asyncio.Queue[str]
    arcade_client: ArcadeDBClient | None = None
    bootstrap_error: str | None = None
    bootstrap_complete: bool = False
    bootstrap_task: asyncio.Task[None] | None = None


def get_services(app: FastAPI) -> AppServices:
    return app.state.services  # type: ignore[no-any-return]


def create_app(
    *,
    settings: Settings | None = None,
    store: KnowledgeStore | None = None,
    ai_provider: AIProvider | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    async def bootstrap_with_retry(services: AppServices) -> None:
        attempts = 0
        while True:
            attempts += 1
            try:
                await services.store.bootstrap_schema()
            except Exception as exc:
                services.bootstrap_error = str(exc)
                services.bootstrap_complete = False
                if attempts >= services.settings.bootstrap_max_attempts:
                    return
                await asyncio.sleep(services.settings.bootstrap_retry_delay_seconds)
                continue

            services.bootstrap_error = None
            services.bootstrap_complete = True
            return

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_level)
        logger.info("app starting", extra={"env": settings.app_env, "ai_provider": settings.ai_provider})
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=settings.queue_maxsize)
        arcade_client = None
        actual_store = store
        if actual_store is None:
            arcade_client = ArcadeDBClient(settings)
            actual_store = ArcadeKnowledgeStore(settings, arcade_client)
        actual_ai_provider = ai_provider or build_ai_provider(settings)
        ingestion_service = IngestionService(
            settings=settings,
            store=actual_store,
            ai_provider=actual_ai_provider,
            queue=queue,
        )
        worker = IngestionWorker(queue, ingestion_service)
        app.state.services = AppServices(
            settings=settings,
            store=actual_store,
            ai_provider=actual_ai_provider,
            ingestion_service=ingestion_service,
            worker=worker,
            queue=queue,
            arcade_client=arcade_client,
        )
        if arcade_client is None:
            await bootstrap_with_retry(app.state.services)
        else:
            app.state.services.bootstrap_task = asyncio.create_task(
                bootstrap_with_retry(app.state.services),
                name="arcadedb-bootstrap",
            )
        await worker.start()
        logger.info("app ready")
        try:
            yield
        finally:
            logger.info("app shutting down")
            await worker.stop()
            if app.state.services.bootstrap_task:
                app.state.services.bootstrap_task.cancel()
                try:
                    await app.state.services.bootstrap_task
                except asyncio.CancelledError:
                    pass
            if arcade_client:
                await arcade_client.close()
            await actual_ai_provider.close()

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_trace_middleware(request: Request, call_next):
        req_id = request.headers.get("x-request-id") or f"req-{uuid.uuid4().hex[:12]}"
        bind(run_id=req_id, step="request")
        response = await call_next(request)
        clear()
        response.headers["x-request-id"] = req_id
        return response

    from app.routers import adaptive, concepts, deletions, episodes, health, jobs, knowledge, pedagogical, search, sr

    app.include_router(health.router)
    app.include_router(knowledge.router)
    app.include_router(jobs.router)
    app.include_router(episodes.router)
    app.include_router(search.router)
    app.include_router(concepts.router)
    app.include_router(deletions.router)
    app.include_router(pedagogical.router)
    app.include_router(sr.router)
    app.include_router(adaptive.router)
    return app
app = create_app()
