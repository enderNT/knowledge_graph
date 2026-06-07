from __future__ import annotations

from app.schema_bootstrap import build_schema_commands


def test_schema_commands_are_idempotent(settings):
    commands = build_schema_commands(settings)
    create_commands = [command for command in commands if command.startswith("CREATE ")]
    assert create_commands
    assert all("IF NOT EXISTS" in command for command in create_commands)


def test_schema_configures_vector_indexes(settings):
    commands = build_schema_commands(settings)
    vector_commands = [command for command in commands if "LSM_VECTOR" in command]
    assert len(vector_commands) == 3
    assert all(str(settings.embedding_dimensions) in command for command in vector_commands)


def test_schema_includes_spaced_repetition_document_and_indexes(settings):
    commands = build_schema_commands(settings)
    assert "CREATE DOCUMENT TYPE UserSpacedRepetition IF NOT EXISTS" in commands
    assert "CREATE PROPERTY UserSpacedRepetition.user_id IF NOT EXISTS STRING" in commands
    assert "CREATE PROPERTY UserSpacedRepetition.next_review_at IF NOT EXISTS DATETIME" in commands
    assert "CREATE INDEX IF NOT EXISTS ON UserSpacedRepetition (user_id, concept_uid, dimension) UNIQUE" in commands
    assert "CREATE INDEX IF NOT EXISTS ON UserSpacedRepetition (next_review_at) NOTUNIQUE" in commands
