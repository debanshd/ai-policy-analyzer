from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class ToolType(str, Enum):
    RAG_TOOL = "RAG_Tool"
    DATA_COMMONS_TOOL = "DataCommonsTool"
    TAVILY_TOOL = "TavilyTool"

class StreamingUpdate(BaseModel):
    """Model for streaming updates during agent execution"""
    type: str = Field(..., description="Type of update: 'tool', 'thought', 'result', 'error'")
    content: str = Field(..., description="Content of the update")
    tool: Optional[ToolType] = Field(None, description="Tool being used (if applicable)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class QueryRequest(BaseModel):
    """Model for query requests"""
    question: str = Field(..., description="User's question", min_length=1)
    session_id: Optional[str] = Field(None, description="Session ID for maintaining context")

class FileUploadResponse(BaseModel):
    """Response model for file uploads"""
    success: bool
    message: str
    file_id: str
    filename: str
    size: int

class DocumentChunk(BaseModel):
    """Model for document chunks"""
    content: str
    metadata: Dict[str, Any]
    page_number: Optional[int] = None

class RAGResult(BaseModel):
    """Model for RAG tool results"""
    relevant_chunks: List[DocumentChunk]
    answer: str
    confidence_score: Optional[float] = None

class DataCommonsResult(BaseModel):
    """Model for Data Commons tool results"""
    data_points: List[Dict[str, Any]]
    summary: str
    sources: List[str]

class TavilyResult(BaseModel):
    """Model for Tavily search results"""
    search_results: List[Dict[str, Any]]
    summary: str
    urls: List[str]

class AgentResponse(BaseModel):
    """Final agent response model"""
    answer: str
    sources: List[str]
    tools_used: List[ToolType]
    session_id: str
    processing_time: float
    
class ErrorResponse(BaseModel):
    """Error response model"""
    error: bool = True
    message: str
    details: Optional[Dict[str, Any]] = None 