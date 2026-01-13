from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "gpt-5-chat"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = "text-embedding-ada-002"
    AZURE_OPENAI_API_VERSION: str = "2024-12-01-preview"
    
    # Application Settings
    APP_NAME: str = "AI Agent System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Vector Store Settings
    VECTOR_STORE_PATH: str = "./data/vectorstore"
    DOCUMENTS_PATH: str = "./documents"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 3
    
    # Agent Settings
    MAX_CONVERSATION_HISTORY: int = 10
    TEMPERATURE: float = 1.0
    MAX_TOKENS: int = 1000
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()