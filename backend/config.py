import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # API Keys - now optional
    openai_api_key: str = Field("", description="OpenAI API key")
    tavily_api_key: str = Field("", description="Tavily API key")
    data_commons_api_key: str = Field("", description="Data Commons API key")
    langsmith_api_key: str = Field("", description="LangSmith API key")

    # LangSmith Configuration
    langsmith_project: str = Field("multi-source-analysis", description="LangSmith project name")
    langsmith_endpoint: str = Field("https://api.smith.langchain.com", description="LangSmith API endpoint")

    # CORS Origins
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://127.0.0.1:3000", "http://127.0.0.1:3001", "http://127.0.0.1:3002"],
        description="Allowed CORS origins"
    )

    # Vector DB Configuration
    vector_collection_name: str = Field("policy_documents", description="Qdrant collection name")
    qdrant_collection_name: str = Field("policy_documents", description="Qdrant collection name (alias)")
    qdrant_location: str = Field(":memory:", description="Qdrant database location (use ':memory:' for in-memory)")
    vector_size: int = Field(1536, description="Vector embedding size")

    # File Upload Configuration
    max_file_size: int = Field(10 * 1024 * 1024, description="Maximum file size in bytes (10MB)")
    allowed_file_types: List[str] = Field(
        default=[".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp"],
        description="Allowed file extensions"
    )

    # LLM Configuration
    model_name: str = Field("gpt-4o-mini", description="OpenAI model name")
    llm_model: str = Field("gpt-4o-mini", description="LLM model name (alias for model_name)")
    embedding_model: str = Field("text-embedding-3-small", description="OpenAI embedding model")
    temperature: float = Field(0.1, description="LLM temperature")
    max_tokens: int = Field(2000, description="Maximum tokens for LLM response")

    # Document Processing Configuration
    chunk_size: int = Field(1000, description="Document chunk size for text splitting")
    chunk_overlap: int = Field(200, description="Overlap between document chunks")

    # Tool Usage Configuration
    enable_external_tools: bool = Field(True, description="Enable external tools (Data Commons, Tavily) when documents may not have sufficient information")
    require_external_tools: bool = Field(False, description="Always use external tools even when documents have complete information")
    completeness_threshold: float = Field(0.5, description="Confidence threshold (0-1) for considering document answer complete")

    class Config:
        env_file = ".env"
        case_sensitive = False

    def validate_required_keys(self) -> dict:
        """Validate which API keys are missing and return status"""
        missing_keys = []
        optional_keys = []

        if not self.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")
        if not self.tavily_api_key:
            optional_keys.append("TAVILY_API_KEY")
        if not self.data_commons_api_key:
            optional_keys.append("DATA_COMMONS_API_KEY")
        if not self.langsmith_api_key:
            optional_keys.append("LANGSMITH_API_KEY")

        return {
            "missing_required": missing_keys,
            "missing_optional": optional_keys,
            "is_valid": len(missing_keys) == 0
        }

# Create settings instance
settings = Settings()

# Validate keys but don't fail immediately
key_status = settings.validate_required_keys()

if not key_status["is_valid"]:
    print("⚠️  Missing required API keys:")
    for key in key_status["missing_required"]:
        print(f"   - {key}")
    print("\n💡 You can set these keys:")
    print("   1. Via environment variables or .env file")
    print("   2. Via the frontend interface when it starts")
    print("   3. Or set them manually:\n")

    for key in key_status["missing_required"]:
        print(f'   export {key}="your_key_here"')

    print(f"\n🔗 Frontend will be available at: http://localhost:3001")
    print(f"🔗 Backend API docs at: http://localhost:8000/docs")

if key_status["missing_optional"]:
    print("\n📝 Optional API keys not set (features may be limited):")
    for key in key_status["missing_optional"]:
        print(f"   - {key}") 