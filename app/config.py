from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
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

    ai_provider: Literal["stub", "openai_compatible", "anthropic"] = Field(default="stub", alias="AI_PROVIDER")
    embedding_provider: Literal["stub", "openai_compatible"] | None = Field(
        default=None,
        alias="EMBEDDING_PROVIDER",
    )
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_CHAT_MODEL")
    openai_embeddings_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDINGS_MODEL",
    )
    openai_timeout_seconds: float = Field(default=180.0, alias="OPENAI_TIMEOUT_SECONDS")
    anthropic_gateway_base_url: str = Field(
        default="http://anthropic-gateway:8081",
        alias="ANTHROPIC_GATEWAY_BASE_URL",
    )
    anthropic_gateway_bearer_token: str | None = Field(default=None, alias="ANTHROPIC_GATEWAY_BEARER_TOKEN")
    anthropic_chat_model: str | None = Field(default=None, alias="ANTHROPIC_CHAT_MODEL")
    anthropic_thinking_type: Literal["adaptive", "enabled"] | None = Field(
        default=None,
        alias="ANTHROPIC_THINKING_TYPE",
    )
    anthropic_thinking_budget_tokens: int | None = Field(
        default=None,
        alias="ANTHROPIC_THINKING_BUDGET_TOKENS",
    )
    anthropic_effort: Literal["low", "medium", "high", "max", "xhigh"] | None = Field(
        default=None,
        alias="ANTHROPIC_EFFORT",
    )
    anthropic_timeout_seconds: float = Field(default=180.0, alias="ANTHROPIC_TIMEOUT_SECONDS")
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

    @property
    def resolved_embedding_provider(self) -> Literal["stub", "openai_compatible"]:
        if self.embedding_provider:
            return self.embedding_provider
        if self.ai_provider == "anthropic":
            raise ValueError("EMBEDDING_PROVIDER is required when AI_PROVIDER=anthropic")
        return self.ai_provider

    @model_validator(mode="after")
    def validate_anthropic_thinking(self) -> "Settings":
        if self.anthropic_thinking_budget_tokens is not None and self.anthropic_thinking_budget_tokens <= 0:
            raise ValueError("ANTHROPIC_THINKING_BUDGET_TOKENS must be greater than 0")

        if self.anthropic_thinking_type == "adaptive" and self.anthropic_thinking_budget_tokens is not None:
            raise ValueError("ANTHROPIC_THINKING_BUDGET_TOKENS cannot be set when ANTHROPIC_THINKING_TYPE=adaptive")

        if self.anthropic_thinking_type == "enabled" and self.anthropic_thinking_budget_tokens is None:
            raise ValueError("ANTHROPIC_THINKING_BUDGET_TOKENS is required when ANTHROPIC_THINKING_TYPE=enabled")

        if self.anthropic_thinking_type is None and self.anthropic_thinking_budget_tokens is not None:
            raise ValueError("ANTHROPIC_THINKING_TYPE is required when ANTHROPIC_THINKING_BUDGET_TOKENS is set")

        model_name = (self.anthropic_chat_model or "").strip().lower()
        if "haiku" in model_name and self.anthropic_thinking_type == "adaptive":
            raise ValueError("ANTHROPIC_THINKING_TYPE=adaptive is not supported for Claude Haiku 4.5")

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
