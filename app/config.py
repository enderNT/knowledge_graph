from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "knowledge-graph-api"
    app_env: Literal["development", "test", "production"] = "development"
    api_key: str = Field(default="change-me", alias="API_KEY")
    log_level: str = "INFO"

    arcadedb_url: str = Field(default="http://arcadedb:2480", alias="ARCADEDB_URL")
    arcadedb_database: str = Field(default="knowledge", alias="ARCADEDB_DATABASE")
    arcadedb_root_username: str = Field(default="root", alias="ARCADEDB_ROOT_USERNAME")
    arcadedb_root_password: str = Field(default="change-me-please", alias="ARCADEDB_ROOT_PASSWORD")

    ai_provider: Literal["stub", "openai_compatible"] = Field(default="stub", alias="AI_PROVIDER")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_CHAT_MODEL")
    openai_embeddings_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDINGS_MODEL",
    )
    embedding_dimensions: int = Field(default=16, alias="EMBEDDING_DIMENSIONS")

    resolution_match_threshold: float = 0.91
    resolution_gap_threshold: float = 0.03
    candidate_limit_multiplier: int = 4
    queue_maxsize: int = 1000
    bootstrap_retry_delay_seconds: float = 3.0
    bootstrap_max_attempts: int = Field(default=20, alias="BOOTSTRAP_MAX_ATTEMPTS")
    mcp_port: int = Field(default=9000, alias="MCP_PORT")
    mcp_bearer_token: str = Field(default="change-me", alias="MCP_BEARER_TOKEN")
    kg_api_base_url: str = Field(default="http://api:8000", alias="KG_API_BASE_URL")
    kg_api_key: str = Field(default="change-me", alias="KG_API_KEY")
    mcp_poll_interval_seconds: float = Field(default=1.0, alias="MCP_POLL_INTERVAL_SECONDS")
    mcp_ingestion_timeout_seconds: float = Field(default=90.0, alias="MCP_INGESTION_TIMEOUT_SECONDS")


@lru_cache
def get_settings() -> Settings:
    return Settings()
