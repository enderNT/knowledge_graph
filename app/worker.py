from __future__ import annotations

import asyncio
import contextlib
import logging

from app.ingestion import IngestionService


logger = logging.getLogger(__name__)


class IngestionWorker:
    def __init__(self, queue: asyncio.Queue[str], service: IngestionService) -> None:
        self.queue = queue
        self.service = service
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        if self._task:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="ingestion-worker")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        while self._running:
            job_id = await self.queue.get()
            try:
                await self.service.process_job(job_id)
            except Exception:
                logger.exception("Failed processing job %s", job_id)
            finally:
                self.queue.task_done()
