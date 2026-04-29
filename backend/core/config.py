"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings for the platform."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "AI Decision Intelligence Platform"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"
    secret_key: str = Field("change-me-in-production", alias="SECRET_KEY")
    access_token_expire_minutes: int = 60 * 8

    database_url: str = Field(
        "sqlite:///./data/app.db",
        alias="DATABASE_URL",
    )

    cors_origins: str = Field("*", alias="CORS_ORIGINS")
    default_llm_provider: str = Field("openai", alias="DEFAULT_LLM_PROVIDER")
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", alias="OPENAI_MODEL")
    gemini_api_key: str | None = Field(None, alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-1.5-flash", alias="GEMINI_MODEL")
    embedding_model_name: str = Field(
        "all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME",
    )
    hf_token: str | None = Field(None, alias="HF_TOKEN")
    hf_persist_repo_id: str | None = Field(None, alias="HF_PERSIST_REPO_ID")
    hf_persist_repo_private: bool = Field(True, alias="HF_PERSIST_REPO_PRIVATE")
    free_deploy_target: str = Field("huggingface-spaces", alias="FREE_DEPLOY_TARGET")
    free_mode: bool = Field(True, alias="FREE_MODE")
    public_api_base_url: str | None = Field(None, alias="PUBLIC_API_BASE_URL")
    request_limit_per_minute: int = Field(90, alias="REQUEST_LIMIT_PER_MINUTE")
    max_document_chunks: int = Field(120, alias="MAX_DOCUMENT_CHUNKS")
    shap_reference_sample_size: int = Field(120, alias="SHAP_REFERENCE_SAMPLE_SIZE")
    recent_task_limit: int = Field(8, alias="RECENT_TASK_LIMIT")

    uploads_dir: Path = BASE_DIR / "data" / "uploads"
    model_dir: Path = BASE_DIR / "data" / "models"
    artifact_dir: Path = BASE_DIR / "data" / "artifacts"
    vector_store_dir: Path = BASE_DIR / "data" / "vector_store"
    data_dir: Path = BASE_DIR / "data"

    forecast_default_horizon: int = 14
    max_upload_size_mb: int = 20
    rag_chunk_size: int = 650
    rag_chunk_overlap: int = 120
    retrieval_top_k: int = 4

    @property
    def cors_origin_list(self) -> list[str]:
        """Return CORS origins as a normalized list."""

        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def sqlite_file_path(self) -> Path | None:
        """Return the local SQLite path when the configured database is file-based."""

        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            return None
        raw_path = self.database_url[len(prefix) :]
        return (BASE_DIR / raw_path[2:]).resolve() if raw_path.startswith("./") else Path(raw_path).resolve()


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
