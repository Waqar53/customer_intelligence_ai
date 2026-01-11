"""
Configuration Management for Customer Intelligence AI.

This module handles all application configuration using Pydantic Settings.
Configuration is loaded from environment variables and .env files.

Why this approach?
- Type-safe configuration with validation
- Easy to override via environment variables
- Works seamlessly with Docker
- Self-documenting with defaults
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be overridden by setting the corresponding
    environment variable (e.g., OPENAI_API_KEY=xxx).
    """
    
    # =========================================================
    # OpenAI Configuration
    # =========================================================
    openai_api_key: str = ""  # Required for LLM functionality
    
    # =========================================================
    # Model Configuration
    # =========================================================
    # Sentence transformer model for embeddings
    # "all-MiniLM-L6-v2" is fast and good quality for semantic search
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # OpenAI model for reasoning/summarization
    # "gpt-4o-mini" is cost-effective and capable
    llm_model: str = "gpt-4o-mini"
    
    # =========================================================
    # Application Settings
    # =========================================================
    app_env: str = "development"  # development, staging, production
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # =========================================================
    # Data & Storage Paths
    # =========================================================
    upload_dir: str = "./data/uploads"
    vector_store_path: str = "./data/vector_store"
    sample_data_path: str = "./data/sample"
    
    # =========================================================
    # ML/Clustering Settings
    # =========================================================
    default_num_clusters: int = 5  # Default K for K-Means
    min_cluster_size: int = 3  # Minimum docs to form a cluster
    
    # =========================================================
    # RAG Settings
    # =========================================================
    retrieval_top_k: int = 5  # Number of documents to retrieve
    max_context_length: int = 4000  # Max chars for LLM context
    
    # =========================================================
    # API Settings
    # =========================================================
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Pydantic v2 configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra env vars
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Using lru_cache ensures settings are only loaded once,
    improving performance and ensuring consistency.
    
    Returns:
        Settings: Application configuration object
    """
    return Settings()


# Convenience function for quick access
settings = get_settings()
