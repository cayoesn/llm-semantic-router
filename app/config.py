import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "LLM Semantic Router & Cache Engine"
    DEBUG: bool = False
    
    # Langfuse Observability
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-demo")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-demo")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    
    # Semantic Cache & Router Thresholds
    SIMILARITY_THRESHOLD: float = 0.88
    EMBEDDING_DIMENSION: int = 384
    DEFAULT_FAST_MODEL: str = "gemini-3.5-flash"
    DEFAULT_ADVANCED_MODEL: str = "gemini-3.5-pro"
    
    # Cost Constants (USD per token for cost metrics)
    COST_PER_1K_FAST: float = 0.0001
    COST_PER_1K_ADVANCED: float = 0.0015

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
