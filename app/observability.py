from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from app.arcadedb_client import ArcadeDBClient

logger = logging.getLogger(__name__)

_SKIP_LOGGERS = frozenset({"app.observability", "httpx", "asyncio", "uvicorn.access"})

# log event → (run_type, new_status)
_START_EVENTS = {
    "job started": ("ingestion", "running"),
    "adaptive session created": ("adaptive", "running"),
}
_END_EVENTS = {
    "job completed": "completed",
    "job failed": "failed",
}


class ObservabilityService:
    def __init__(self, client: ArcadeDBClient) -> None:
        self._client = client
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=10_000)
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self._drain_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._drain_task = asyncio.create_task(self._drain_loop(), name="obs-drain")

    async def stop(self) -> None:
        if self._drain_task:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass

    def enqueue(self, record: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            pass

    async def _drain_loop(self) -> None:
        while True:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=2.0)
                batch = [item]
                while not self._queue.empty() and len(batch) < 100:
                    batch.append(self._queue.get_nowait())
                await self._write_batch(batch)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def _write_batch(self, batch: list[dict[str, Any]]) -> None:
        for item in batch:
            try:
                await self._client.command(
                    "INSERT INTO ObservabilityLog SET "
                    "ts = :ts, ts_epoch_ms = :ts_epoch_ms, level = :level, "
                    "service = :service, logger_name = :logger_name, event = :event, "
                    "run_id = :run_id, step = :step, job_id = :job_id, "
                    "session_id = :session_id, episode_id = :episode_id, "
                    "block_id = :block_id, tool_name = :tool_name, path = :path, "
                    "status = :status, duration_ms = :duration_ms, "
                    "input_shape_json = :input_shape_json, "
                    "output_shape_json = :output_shape_json, "
                    "counts_json = :counts_json, "
                    "error_type = :error_type, error_message = :error_message",
                    {
                        "ts": item.get("ts", ""),
                        "ts_epoch_ms": item.get("ts_epoch_ms", int(time.time() * 1000)),
                        "level": item.get("level", ""),
                        "service": item.get("service", ""),
                        "logger_name": item.get("logger", ""),
                        "event": item.get("event", item.get("msg", "")),
                        "run_id": item.get("run_id", ""),
                        "step": item.get("step", ""),
                        "job_id": item.get("job_id", ""),
                        "session_id": item.get("session_id", ""),
                        "episode_id": item.get("episode_id", ""),
                        "block_id": item.get("block_id", ""),
                        "tool_name": item.get("tool_name", ""),
                        "path": item.get("path", ""),
                        "status": item.get("status", ""),
                        "duration_ms": item.get("duration_ms"),
                        "input_shape_json": _to_json(item.get("input_shape")),
                        "output_shape_json": _to_json(item.get("output_shape")),
                        "counts_json": _to_json(item.get("counts")),
                        "error_type": item.get("error_type", ""),
                        "error_message": item.get("error_message", ""),
                    },
                )
                # broadcast to SSE subscribers using the same shape as get_run_events:
                # logger_name + already-parsed input_shape/output_shape/counts objects.
                broadcast = dict(item)
                broadcast["logger_name"] = item.get("logger", item.get("logger_name", ""))
                broadcast.setdefault("input_shape", item.get("input_shape"))
                broadcast.setdefault("output_shape", item.get("output_shape"))
                broadcast.setdefault("counts", item.get("counts"))
                for q in list(self._subscribers):
                    try:
                        q.put_nowait(broadcast)
                    except asyncio.QueueFull:
                        pass
            except Exception:
                pass

            # handle run lifecycle transitions
            await self._handle_transition(item)

    async def _handle_transition(self, item: dict[str, Any]) -> None:
        event = item.get("event", item.get("msg", ""))
        run_id = item.get("run_id", "")
        if not run_id:
            return
        try:
            if event in _START_EVENTS:
                run_type, status = _START_EVENTS[event]
                await self._upsert_run(
                    run_id=run_id,
                    run_type=run_type,
                    status=status,
                    job_id=item.get("job_id", ""),
                    session_id=item.get("session_id", ""),
                    episode_id=item.get("episode_id", ""),
                    start_epoch_ms=item.get("ts_epoch_ms", int(time.time() * 1000)),
                    start_ts=item.get("ts", ""),
                )
            elif event in _END_EVENTS:
                status = _END_EVENTS[event]
                now_ms = item.get("ts_epoch_ms", int(time.time() * 1000))
                await self._upsert_run(
                    run_id=run_id,
                    status=status,
                    end_epoch_ms=now_ms,
                    end_ts=item.get("ts", ""),
                    duration_ms=item.get("duration_ms"),
                    episode_id=item.get("episode_id", ""),
                )
            elif event == "block submission complete" and item.get("session_status") == "closed":
                now_ms = item.get("ts_epoch_ms", int(time.time() * 1000))
                await self._upsert_run(
                    run_id=run_id,
                    status="completed",
                    end_epoch_ms=now_ms,
                    end_ts=item.get("ts", ""),
                    duration_ms=item.get("duration_ms"),
                )
        except Exception:
            pass

    async def _upsert_run(
        self,
        *,
        run_id: str,
        run_type: str = "",
        status: str,
        job_id: str = "",
        session_id: str = "",
        episode_id: str = "",
        start_epoch_ms: int | None = None,
        start_ts: str = "",
        end_epoch_ms: int | None = None,
        end_ts: str = "",
        duration_ms: int | None = None,
    ) -> None:
        existing = await self._client.query(
            "SELECT FROM LogRun WHERE run_id = :run_id", {"run_id": run_id}
        )
        if not existing:
            now_ms = start_epoch_ms or int(time.time() * 1000)
            now_ts = start_ts or datetime.now(timezone.utc).isoformat()
            await self._client.command(
                "INSERT INTO LogRun SET run_id = :run_id, run_type = :run_type, "
                "status = :status, job_id = :job_id, session_id = :session_id, "
                "episode_id = :episode_id, start_ts = :start_ts, start_epoch_ms = :start_epoch_ms",
                {
                    "run_id": run_id,
                    "run_type": run_type,
                    "status": status,
                    "job_id": job_id,
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "start_ts": now_ts,
                    "start_epoch_ms": now_ms,
                },
            )
        else:
            sets: list[str] = ["status = :status"]
            params: dict[str, Any] = {"run_id": run_id, "status": status}
            if end_epoch_ms is not None:
                sets.append("end_epoch_ms = :end_epoch_ms")
                params["end_epoch_ms"] = end_epoch_ms
            if end_ts:
                sets.append("end_ts = :end_ts")
                params["end_ts"] = end_ts
            if duration_ms is not None:
                sets.append("duration_ms = :duration_ms")
                params["duration_ms"] = duration_ms
            if episode_id:
                sets.append("episode_id = :episode_id")
                params["episode_id"] = episode_id
            await self._client.command(
                f"UPDATE LogRun SET {', '.join(sets)} WHERE run_id = :run_id", params
            )

    async def list_runs(
        self,
        *,
        sort: str = "desc",
        limit: int = 10,
        skip: int = 0,
        job_id: str | None = None,
        session_id: str | None = None,
        episode_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: dict[str, Any] = {"limit": min(limit, 10), "skip": max(skip, 0)}
        if job_id:
            where.append("job_id = :job_id")
            params["job_id"] = job_id
        if session_id:
            where.append("session_id = :session_id")
            params["session_id"] = session_id
        if episode_id:
            where.append("episode_id = :episode_id")
            params["episode_id"] = episode_id
        if since:
            where.append("start_ts >= :since")
            params["since"] = since
        if until:
            where.append("start_ts <= :until")
            params["until"] = until
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        order = "DESC" if sort == "desc" else "ASC"
        sql = f"SELECT FROM LogRun {where_sql} ORDER BY start_epoch_ms {order} LIMIT :limit SKIP :skip"
        return await self._client.query(sql, params)

    async def get_run_events(self, run_id: str) -> list[dict[str, Any]]:
        rows = await self._client.query(
            "SELECT FROM ObservabilityLog WHERE run_id = :run_id ORDER BY ts_epoch_ms ASC",
            {"run_id": run_id},
        )
        return [_deserialize_event(row) for row in rows]

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = await self._client.query(
            "SELECT FROM LogRun WHERE run_id = :run_id", {"run_id": run_id}
        )
        return rows[0] if rows else None

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=500)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass


def _deserialize_event(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    for field in ("input_shape", "output_shape", "counts"):
        json_val = row.pop(f"{field}_json", None)
        if json_val:
            try:
                row[field] = json.loads(json_val)
            except Exception:
                row[field] = None
        else:
            row.setdefault(field, None)
    return row


def _to_json(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, default=str)
    except Exception:
        return ""
