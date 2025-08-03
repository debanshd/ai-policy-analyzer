import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

class Settings(BaseSettings):
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    langsmith_api_key: Optional[str] = os.getenv("LANGSMITH_API_KEY")
    
    # LangSmith Configuration
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "multi-source-analysis-agent")
    langsmith_tracing: bool = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
    
    # Model Configuration
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    
    # Chunking Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Qdrant Configuration
    qdrant_location: str = ":memory:"  # For development, use in-memory
    qdrant_collection_name: str = "user_documents"
    
    # File Upload Configuration
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: list = [".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md"]
    upload_directory: str = "uploads"
    
    # API Configuration
    cors_origins: list = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    class Config:
        case_sensitive = False

settings = Settings()

# Validate required API keys
if not settings.openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

if not settings.tavily_api_key:
    raise ValueError("TAVILY_API_KEY environment variable is required")

# Set up LangSmith if configured
if settings.langsmith_api_key and settings.langsmith_tracing:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project 