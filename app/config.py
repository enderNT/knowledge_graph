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
    agent_mcp_port: int = Field(default=9100, alias="AGENT_MCP_PORT")
    agent_mcp_bearer_token: str = Field(default="change-me-agent", alias="AGENT_MCP_BEARER_TOKEN")
    knowledge_mcp_base_url: str = Field(default="http://mcp:9000", alias="KNOWLEDGE_MCP_BASE_URL")
    knowledge_mcp_bearer_token: str = Field(default="change-me-knowledge", alias="KNOWLEDGE_MCP_BEARER_TOKEN")
    kg_api_base_url: str = Field(default="http://api:8000", alias="KG_API_BASE_URL")
    kg_api_key: str = Field(default="change-me", alias="KG_API_KEY")
    mcp_poll_interval_seconds: float = Field(default=1.0, alias="MCP_POLL_INTERVAL_SECONDS")
    mcp_ingestion_timeout_seconds: float = Field(default=90.0, alias="MCP_INGESTION_TIMEOUT_SECONDS")
    agent_openai_base_url: str | None = Field(default=None, alias="AGENT_OPENAI_BASE_URL")
    agent_openai_api_key: str | None = Field(default=None, alias="AGENT_OPENAI_API_KEY")
    agent_openai_chat_model: str | None = Field(default=None, alias="AGENT_OPENAI_CHAT_MODEL")
    learning_context_full_text_min_score: float = 0.84
    learning_context_vector_min_score: float = 0.88
    learning_context_low_score_threshold: float = 0.9
    learning_context_min_description_chars: int = 24
    learning_context_fragmentation_gap_threshold: float = 0.08
    learning_context_episode_excerpt_chars: int = 280

    @property
    def resolved_agent_openai_base_url(self) -> str:
        return self.agent_openai_base_url or self.openai_base_url

    @property
    def resolved_agent_openai_api_key(self) -> str | None:
        return self.agent_openai_api_key or self.openai_api_key

    @property
    def resolved_agent_openai_chat_model(self) -> str:
        return self.agent_openai_chat_model or self.openai_chat_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
