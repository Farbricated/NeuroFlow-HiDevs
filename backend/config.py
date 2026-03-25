from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "neuroflow"
    postgres_user: str = "neuroflow"
    postgres_password: str

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # OpenTelemetry / Jaeger
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "neuroflow-api"

    # LLM Providers
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Application
    api_secret_key: str = "change-me-in-production"
    environment: str = "development"
    log_level: str = "INFO"

    # Rate limits
    ingest_rate_limit_per_hour: int = 10
    query_rate_limit_per_minute: int = 60

    # Queue backpressure thresholds
    queue_warn_depth: int = 50
    queue_block_depth: int = 100

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def asyncpg_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
