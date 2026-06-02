from __future__ import annotations

from app.arcadedb_client import ArcadeDBClient
from app.config import Settings


VERTEX_TYPES = [
    "Concept",
    "Topic",
    "Domain",
    "Source",
    "Episode",
    "Claim",
    "Alias",
]

EDGE_TYPES = [
    "PART_OF",
    "IS_A",
    "RELATED_TO",
    "EXPLAINS",
    "CONTRASTS_WITH",
    "PREREQUISITE_FOR",
    "SUPPORTED_BY",
    "MENTIONED_IN",
    "ALIAS_OF",
]

DOCUMENT_TYPES = ["IngestionJob"]


def build_schema_commands(settings: Settings) -> list[str]:
    commands: list[str] = []
    commands.extend(f"CREATE VERTEX TYPE {name} IF NOT EXISTS" for name in VERTEX_TYPES)
    commands.extend(f"CREATE EDGE TYPE {name} IF NOT EXISTS" for name in EDGE_TYPES)
    commands.extend(f"CREATE DOCUMENT TYPE {name} IF NOT EXISTS" for name in DOCUMENT_TYPES)

    commands.extend(
        [
            "CREATE PROPERTY Concept.uid IF NOT EXISTS STRING",
            "CREATE PROPERTY Concept.canonical_name IF NOT EXISTS STRING",
            "CREATE PROPERTY Concept.normalized_name IF NOT EXISTS STRING",
            "CREATE PROPERTY Concept.description IF NOT EXISTS STRING",
            "CREATE PROPERTY Concept.domain IF NOT EXISTS STRING",
            "CREATE PROPERTY Concept.aliases IF NOT EXISTS LIST OF STRING",
            "CREATE PROPERTY Concept.embedding IF NOT EXISTS ARRAY_OF_FLOATS",
            "CREATE PROPERTY Concept.source_confidence IF NOT EXISTS DOUBLE",
            "CREATE PROPERTY Concept.created_at IF NOT EXISTS DATETIME",
            "CREATE PROPERTY Concept.updated_at IF NOT EXISTS DATETIME",
            "CREATE PROPERTY Episode.uid IF NOT EXISTS STRING",
            "CREATE PROPERTY Episode.text IF NOT EXISTS STRING",
            "CREATE PROPERTY Episode.source_type IF NOT EXISTS STRING",
            "CREATE PROPERTY Episode.tags IF NOT EXISTS LIST OF STRING",
            "CREATE PROPERTY Episode.language IF NOT EXISTS STRING",
            "CREATE PROPERTY Episode.status IF NOT EXISTS STRING",
            "CREATE PROPERTY Episode.error_message IF NOT EXISTS STRING",
            "CREATE PROPERTY Episode.embedding IF NOT EXISTS ARRAY_OF_FLOATS",
            "CREATE PROPERTY Episode.created_at IF NOT EXISTS DATETIME",
            "CREATE PROPERTY Claim.uid IF NOT EXISTS STRING",
            "CREATE PROPERTY Claim.text IF NOT EXISTS STRING",
            "CREATE PROPERTY Claim.normalized_text IF NOT EXISTS STRING",
            "CREATE PROPERTY Claim.confidence IF NOT EXISTS DOUBLE",
            "CREATE PROPERTY Claim.status IF NOT EXISTS STRING",
            "CREATE PROPERTY Claim.embedding IF NOT EXISTS ARRAY_OF_FLOATS",
            "CREATE PROPERTY Claim.created_at IF NOT EXISTS DATETIME",
            "CREATE PROPERTY Domain.uid IF NOT EXISTS STRING",
            "CREATE PROPERTY Domain.name IF NOT EXISTS STRING",
            "CREATE PROPERTY Domain.normalized_name IF NOT EXISTS STRING",
            "CREATE PROPERTY Domain.created_at IF NOT EXISTS DATETIME",
            "CREATE PROPERTY Topic.uid IF NOT EXISTS STRING",
            "CREATE PROPERTY Topic.name IF NOT EXISTS STRING",
            "CREATE PROPERTY Topic.normalized_name IF NOT EXISTS STRING",
            "CREATE PROPERTY Topic.created_at IF NOT EXISTS DATETIME",
            "CREATE PROPERTY Alias.uid IF NOT EXISTS STRING",
            "CREATE PROPERTY Alias.value IF NOT EXISTS STRING",
            "CREATE PROPERTY Alias.normalized_value IF NOT EXISTS STRING",
            "CREATE PROPERTY Alias.created_at IF NOT EXISTS DATETIME",
            "CREATE PROPERTY IngestionJob.uid IF NOT EXISTS STRING",
            "CREATE PROPERTY IngestionJob.episode_id IF NOT EXISTS STRING",
            "CREATE PROPERTY IngestionJob.status IF NOT EXISTS STRING",
            "CREATE PROPERTY IngestionJob.result IF NOT EXISTS STRING",
            "CREATE PROPERTY IngestionJob.error IF NOT EXISTS STRING",
            "CREATE PROPERTY IngestionJob.created_at IF NOT EXISTS DATETIME",
            "CREATE PROPERTY IngestionJob.updated_at IF NOT EXISTS DATETIME",
        ]
    )

    commands.extend(
        [
            "CREATE INDEX IF NOT EXISTS ON Concept (uid) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON Concept (normalized_name) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON Concept (domain) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON Concept (canonical_name) FULL_TEXT",
            "CREATE INDEX IF NOT EXISTS ON Episode (uid) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON Episode (created_at) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON Episode (text) FULL_TEXT",
            "CREATE INDEX IF NOT EXISTS ON Claim (uid) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON Claim (normalized_text) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON Claim (text) FULL_TEXT",
            "CREATE INDEX IF NOT EXISTS ON Domain (uid) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON Domain (normalized_name) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON Topic (uid) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON Topic (normalized_name) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON Alias (uid) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON Alias (normalized_value) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON Alias (value) FULL_TEXT",
            "CREATE INDEX IF NOT EXISTS ON IngestionJob (uid) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON IngestionJob (episode_id) NOTUNIQUE",
            (
                "CREATE INDEX IF NOT EXISTS ON Concept (embedding) LSM_VECTOR "
                f"METADATA {{dimensions: {settings.embedding_dimensions}, similarity: 'COSINE'}}"
            ),
            (
                "CREATE INDEX IF NOT EXISTS ON Episode (embedding) LSM_VECTOR "
                f"METADATA {{dimensions: {settings.embedding_dimensions}, similarity: 'COSINE'}}"
            ),
            (
                "CREATE INDEX IF NOT EXISTS ON Claim (embedding) LSM_VECTOR "
                f"METADATA {{dimensions: {settings.embedding_dimensions}, similarity: 'COSINE'}}"
            ),
        ]
    )
    return commands


async def ensure_database_and_schema(client: ArcadeDBClient, settings: Settings) -> None:
    if not await client.database_exists(settings.arcadedb_database):
        await client.server_command(f"create database {settings.arcadedb_database}")

    for command in build_schema_commands(settings):
        await client.command(command)
