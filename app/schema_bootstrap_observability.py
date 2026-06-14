from __future__ import annotations

from app.arcadedb_client import ArcadeDBClient


async def ensure_observability_schema(client: ArcadeDBClient) -> None:
    if not await client.database_exists():
        await client.server_command(f"create database {client._database}")
    for cmd in _build_commands():
        try:
            await client.command_idempotent(cmd)
        except Exception:
            pass


def _build_commands() -> list[str]:
    cmds: list[str] = ["CREATE DOCUMENT TYPE ObservabilityLog IF NOT EXISTS"]
    for prop, type_ in [
        ("ts", "STRING"),
        ("ts_epoch_ms", "LONG"),
        ("level", "STRING"),
        ("service", "STRING"),
        ("logger_name", "STRING"),
        ("event", "STRING"),
        ("run_id", "STRING"),
        ("step", "STRING"),
        ("job_id", "STRING"),
        ("session_id", "STRING"),
        ("episode_id", "STRING"),
        ("block_id", "STRING"),
        ("tool_name", "STRING"),
        ("path", "STRING"),
        ("status", "STRING"),
        ("duration_ms", "LONG"),
        ("input_shape_json", "STRING"),
        ("output_shape_json", "STRING"),
        ("counts_json", "STRING"),
        ("error_type", "STRING"),
        ("error_message", "STRING"),
    ]:
        cmds.append(f"CREATE PROPERTY ObservabilityLog.{prop} IF NOT EXISTS {type_}")
    cmds += [
        "CREATE INDEX ON ObservabilityLog (run_id) NOTUNIQUE",
        "CREATE INDEX ON ObservabilityLog (ts_epoch_ms) NOTUNIQUE",
        "CREATE INDEX ON ObservabilityLog (job_id) NOTUNIQUE",
        "CREATE INDEX ON ObservabilityLog (session_id) NOTUNIQUE",
        "CREATE INDEX ON ObservabilityLog (episode_id) NOTUNIQUE",
        "CREATE DOCUMENT TYPE LogRun IF NOT EXISTS",
    ]
    for prop, type_ in [
        ("run_id", "STRING"),
        ("run_type", "STRING"),
        ("start_ts", "STRING"),
        ("start_epoch_ms", "LONG"),
        ("end_ts", "STRING"),
        ("end_epoch_ms", "LONG"),
        ("status", "STRING"),
        ("job_id", "STRING"),
        ("session_id", "STRING"),
        ("episode_id", "STRING"),
        ("duration_ms", "LONG"),
    ]:
        cmds.append(f"CREATE PROPERTY LogRun.{prop} IF NOT EXISTS {type_}")
    cmds += [
        "CREATE INDEX ON LogRun (run_id) UNIQUE",
        "CREATE INDEX ON LogRun (start_epoch_ms) NOTUNIQUE",
        "CREATE INDEX ON LogRun (job_id) NOTUNIQUE",
        "CREATE INDEX ON LogRun (session_id) NOTUNIQUE",
    ]
    return cmds
