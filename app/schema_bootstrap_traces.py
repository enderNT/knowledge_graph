from __future__ import annotations

from app.arcadedb_client import ArcadeDBClient


async def ensure_trace_schema(client: ArcadeDBClient) -> None:
    for cmd in _build_commands():
        try:
            await client.command_idempotent(cmd)
        except Exception:
            pass


def _build_commands() -> list[str]:
    cmds: list[str] = ["CREATE DOCUMENT TYPE CanonicalTrace IF NOT EXISTS"]
    for prop, type_ in [
        ("trace_id", "STRING"),
        ("execution_type", "STRING"),
        ("execution_id", "STRING"),
        ("episode_id", "STRING"),
        ("status", "STRING"),
        ("started_at", "STRING"),
        ("ended_at", "STRING"),
        ("duration_ms", "LONG"),
        ("total_steps", "INTEGER"),
        ("total_decisions", "INTEGER"),
        ("error_count", "INTEGER"),
        ("status_counts_json", "STRING"),
        ("semantic_counts_json", "STRING"),
        ("domain", "STRING"),
    ]:
        cmds.append(f"CREATE PROPERTY CanonicalTrace.{prop} IF NOT EXISTS {type_}")
    cmds += [
        "CREATE INDEX ON CanonicalTrace (trace_id) UNIQUE",
        "CREATE INDEX ON CanonicalTrace (execution_id) NOTUNIQUE",
        "CREATE INDEX ON CanonicalTrace (episode_id) NOTUNIQUE",
        "CREATE INDEX ON CanonicalTrace (started_at) NOTUNIQUE",
        "CREATE DOCUMENT TYPE CanonicalTraceEvent IF NOT EXISTS",
    ]
    for prop, type_ in [
        ("event_id", "STRING"),
        ("trace_id", "STRING"),
        ("parent_event_id", "STRING"),
        ("sequence", "INTEGER"),
        ("type", "STRING"),
        ("role", "STRING"),
        ("status", "STRING"),
        ("title", "STRING"),
        ("summary", "STRING"),
        ("input_json", "STRING"),
        ("output_json", "STRING"),
        ("detail_json", "STRING"),
        ("boundary_payload_json", "STRING"),
        ("created_at", "STRING"),
    ]:
        cmds.append(f"CREATE PROPERTY CanonicalTraceEvent.{prop} IF NOT EXISTS {type_}")
    cmds += [
        "CREATE INDEX ON CanonicalTraceEvent (event_id) UNIQUE",
        "CREATE INDEX ON CanonicalTraceEvent (trace_id) NOTUNIQUE",
        "CREATE INDEX ON CanonicalTraceEvent (sequence) NOTUNIQUE",
    ]
    return cmds
