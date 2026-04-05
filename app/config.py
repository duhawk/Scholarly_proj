from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://user:password@db:5432/scholarly"
    anthropic_api_key: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 8
    max_iterations: int = 3
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    reranker_top_n: int = 5

    # Hardening
    api_key: str = ""                # Empty = auth disabled (dev mode)
    rate_limit_per_minute: int = 20  # 0 = disabled
    max_upload_size_mb: int = 50

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
