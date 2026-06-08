"""
Application configuration loaded from environment variables.

Use pydantic-settings to validate and type-check all config.
Create a .env file in apps/backend/ based on .env.example
"""

from enum import StrEnum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalMode(StrEnum):
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


class Settings(BaseSettings):
    """All app settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # allow extra env vars without error
    )

    # ─── Database ─────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+psycopg2://postgres:postgres@localhost:5432/ecommerce",
        description="PostgreSQL connection string (SQLAlchemy format)",
    )

    # ─── Neo4j Graph Database ─────────────────────────────────────────────────
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt connection URI",
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j username",
    )
    neo4j_password: str = Field(
        default="password",
        description="Neo4j password",
    )

    # ─── LLM / Embeddings (GLM-4 via OpenAI-compatible API) ───────────────────
    openai_api_key: str = Field(
        default="",
        description="OpenAI-compatible API key (e.g., GLM-4-Flash from bigmodel.cn)",
    )
    openai_api_base: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4/",
        description="Base URL for the OpenAI-compatible API",
    )
    openai_model: str = Field(
        default="glm-4-flash",
        description="Chat model name",
    )
    embedding_model: str = Field(
        default="embedding-2",
        description="Embedding model name",
    )
    embedding_dim: int = Field(
        default=1024,
        description="Embedding vector dimension",
    )

    # ─── Redis (optional — for session memory in production) ──────────────────
    redis_url: str | None = Field(
        default=None,
        description="Redis connection string, e.g. redis://localhost:6379/0",
    )

    # ─── API / CORS ───────────────────────────────────────────────────────────
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        description="Comma-separated list of allowed CORS origins",
    )
    api_host: str = Field(default="0.0.0.0", description="FastAPI bind host")
    api_port: int = Field(default=8000, description="FastAPI bind port")

    # ─── Retrieval ────────────────────────────────────────────────────────────
    retrieval_mode: RetrievalMode = Field(
        default=RetrievalMode.HYBRID,
        description="Policy retrieval mode: vector | graph | hybrid",
    )

    # ─── Intent Classification ────────────────────────────────────────────────
    classification_mode: str = Field(
        default="keyword",
        description="Intent classifier: keyword (<100 products) | "
        "llm_hybrid (100-10K) | semantic (10K+)",
    )

    # ─── Checkpointer ─────────────────────────────────────────────────────────
    checkpoint_type: str = Field(
        default="postgres",
        description="Checkpointer backend: postgres | memory",
    )

    # ─── App Paths ────────────────────────────────────────────────────────────
    data_dir: str = Field(default="data", description="Directory for local data files")
    memory_filepath: str = Field(
        default="data/memory_store.json",
        description="Path to JSON memory store (dev only, legacy ReAct agent)",
    )

    # ─── Logging ──────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Python logging level")

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS string into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def pg_connection_raw(self) -> str:
        """
        Return a psycopg2-compatible connection string.
        Converts sqlalchemy 'postgresql+psycopg2://' to 'postgresql://' if needed.
        """
        raw = self.database_url
        if raw.startswith("postgresql+psycopg2://"):
            return raw.replace("postgresql+psycopg2://", "postgresql://", 1)
        return raw


# Global settings instance — imported by other modules
settings = Settings()
