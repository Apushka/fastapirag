from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os
from typing import Literal

PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    qdrant_host: str
    qdrant_port: int
    ollama_host: str
    embedding_model: str
    embedding_vector_size: str
    chunk_size: int
    chunk_overlap: int
    llm_model: str
    collection_name: str

    langsmith_tracing: bool
    langsmith_endpoint: str
    langsmith_api_key: str
    langsmith_project: str

    log_level: Literal["INFO", "DEBUG"]

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8"
    )


settings = Settings()

os.environ["LANGSMITH_TRACING"] = "true" if settings.langsmith_tracing else "false"
os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
