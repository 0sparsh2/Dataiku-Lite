"""
Configuration settings for the ML/DS Educational Platform
"""
import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # Groq API Configuration
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    
    # Database Configuration
    database_url: Optional[str] = None
    mongo_url: Optional[str] = None
    
    # Application Configuration
    debug: bool = True
    log_level: str = "INFO"
    max_file_size_mb: int = 100
    cache_ttl_hours: int = 24
    
    # Model Configuration
    max_features: int = 1000
    default_test_size: float = 0.2
    random_state: int = 42
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
