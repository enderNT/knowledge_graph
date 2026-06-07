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

DOCUMENT_TYPES = [
    "IngestionJob",
    "UserConceptMastery",
    "UserDomainMastery",
    "UserEvaluationEvent",
    "AdaptiveSession",
    "AdaptiveBlockAttempt",
]


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
            "CREATE PROPERTY UserConceptMastery.user_id IF NOT EXISTS STRING",
            "CREATE PROPERTY UserConceptMastery.concept_uid IF NOT EXISTS STRING",
            "CREATE PROPERTY UserConceptMastery.concept_name IF NOT EXISTS STRING",
            "CREATE PROPERTY UserConceptMastery.domain IF NOT EXISTS STRING",
            "CREATE PROPERTY UserConceptMastery.mastery_score_0_to_100 IF NOT EXISTS DOUBLE",
            "CREATE PROPERTY UserConceptMastery.mastery_label IF NOT EXISTS STRING",
            "CREATE PROPERTY UserConceptMastery.dimensions_json IF NOT EXISTS STRING",
            "CREATE PROPERTY UserConceptMastery.confidence_0_to_1 IF NOT EXISTS DOUBLE",
            "CREATE PROPERTY UserConceptMastery.trend IF NOT EXISTS STRING",
            "CREATE PROPERTY UserConceptMastery.priority_score IF NOT EXISTS DOUBLE",
            "CREATE PROPERTY UserConceptMastery.last_block_id IF NOT EXISTS STRING",
            "CREATE PROPERTY UserConceptMastery.recent_history_json IF NOT EXISTS STRING",
            "CREATE PROPERTY UserConceptMastery.recent_stats_json IF NOT EXISTS STRING",
            "CREATE PROPERTY UserConceptMastery.weaknesses_json IF NOT EXISTS STRING",
            "CREATE PROPERTY UserConceptMastery.detected_gaps_json IF NOT EXISTS STRING",
            "CREATE PROPERTY UserConceptMastery.suggested_questions_json IF NOT EXISTS STRING",
            "CREATE PROPERTY UserConceptMastery.effective_depth_used IF NOT EXISTS INTEGER",
            "CREATE PROPERTY UserConceptMastery.last_evaluated_at IF NOT EXISTS DATETIME",
            "CREATE PROPERTY UserConceptMastery.updated_at IF NOT EXISTS DATETIME",
            "CREATE PROPERTY UserConceptMastery.recalculation_traces_json IF NOT EXISTS STRING",
            "CREATE PROPERTY UserDomainMastery.user_id IF NOT EXISTS STRING",
            "CREATE PROPERTY UserDomainMastery.domain IF NOT EXISTS STRING",
            "CREATE PROPERTY UserDomainMastery.mastery_score_0_to_100 IF NOT EXISTS DOUBLE",
            "CREATE PROPERTY UserDomainMastery.mastery_label IF NOT EXISTS STRING",
            "CREATE PROPERTY UserDomainMastery.concept_count IF NOT EXISTS INTEGER",
            "CREATE PROPERTY UserDomainMastery.weak_concept_uids_json IF NOT EXISTS STRING",
            "CREATE PROPERTY UserDomainMastery.recent_stats_json IF NOT EXISTS STRING",
            "CREATE PROPERTY UserDomainMastery.updated_at IF NOT EXISTS DATETIME",
            "CREATE PROPERTY UserDomainMastery.recalculation_traces_json IF NOT EXISTS STRING",
            "CREATE PROPERTY UserEvaluationEvent.uid IF NOT EXISTS STRING",
            "CREATE PROPERTY UserEvaluationEvent.user_id IF NOT EXISTS STRING",
            "CREATE PROPERTY UserEvaluationEvent.concept_uid IF NOT EXISTS STRING",
            "CREATE PROPERTY UserEvaluationEvent.concept_name IF NOT EXISTS STRING",
            "CREATE PROPERTY UserEvaluationEvent.domain IF NOT EXISTS STRING",
            "CREATE PROPERTY UserEvaluationEvent.score_0_to_100 IF NOT EXISTS DOUBLE",
            "CREATE PROPERTY UserEvaluationEvent.recorded_at IF NOT EXISTS DATETIME",
            "CREATE PROPERTY UserEvaluationEvent.source IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveSession.session_id IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveSession.user_id IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveSession.status IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveSession.resolved_reference_json IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveSession.domain_hint IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveSession.language IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveSession.constraints_json IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveSession.tutor_context_json IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveSession.current_block_json IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveSession.block_history_json IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveSession.summary_json IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveSession.opened_at IF NOT EXISTS DATETIME",
            "CREATE PROPERTY AdaptiveSession.updated_at IF NOT EXISTS DATETIME",
            "CREATE PROPERTY AdaptiveBlockAttempt.session_id IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveBlockAttempt.block_id IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveBlockAttempt.plan_json IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveBlockAttempt.items_json IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveBlockAttempt.answer_keys_json IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveBlockAttempt.submissions_json IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveBlockAttempt.interaction_events_json IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveBlockAttempt.block_result_json IF NOT EXISTS STRING",
            "CREATE PROPERTY AdaptiveBlockAttempt.created_at IF NOT EXISTS DATETIME",
            "CREATE PROPERTY AdaptiveBlockAttempt.updated_at IF NOT EXISTS DATETIME",
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
            "CREATE INDEX IF NOT EXISTS ON UserConceptMastery (user_id) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON UserConceptMastery (concept_uid) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON UserConceptMastery (domain) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON UserConceptMastery (updated_at) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON UserDomainMastery (user_id) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON UserDomainMastery (domain) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON UserDomainMastery (updated_at) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON UserEvaluationEvent (uid) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON UserEvaluationEvent (user_id) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON UserEvaluationEvent (concept_uid) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON UserEvaluationEvent (domain) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON UserEvaluationEvent (recorded_at) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON AdaptiveSession (session_id) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON AdaptiveSession (user_id) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON AdaptiveSession (updated_at) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON AdaptiveBlockAttempt (block_id) UNIQUE",
            "CREATE INDEX IF NOT EXISTS ON AdaptiveBlockAttempt (session_id) NOTUNIQUE",
            "CREATE INDEX IF NOT EXISTS ON AdaptiveBlockAttempt (updated_at) NOTUNIQUE",
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
