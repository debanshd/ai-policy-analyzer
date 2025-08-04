import asyncio
import uuid
import time
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import settings, key_status
from agent import MultiSourceAnalysisAgent
from models import FileUploadResponse
from utils.document_processor import validate_file_upload

# Global variables
agent: Optional[MultiSourceAnalysisAgent] = None

# API Key update models
class ApiKeyUpdate(BaseModel):
    openai_api_key: str = ""
    tavily_api_key: str = ""
    data_commons_api_key: str = ""
    langsmith_api_key: str = ""

class ApiKeyStatus(BaseModel):
    api_keys: dict
    status: dict
    backend_ready: bool

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global agent
    
    # Only initialize agent if we have required keys
    if key_status["is_valid"]:
        print("🚀 Initializing Multi-Source Analysis Agent...")
        agent = MultiSourceAnalysisAgent()
        print("✅ Agent initialized successfully")
    else:
        print("⏳ Waiting for API keys to be configured...")
        agent = None
    
    yield
    
    # Cleanup
    cleanup_task = asyncio.create_task(cleanup_agent())
    await cleanup_task

async def cleanup_agent():
    """Cleanup agent resources"""
    global agent
    if agent:
        # Cleanup vector stores
        agent.session_stores.clear()
        print("🧹 Cleaned up agent resources")

# Create FastAPI app
app = FastAPI(
    title="Multi-Source Analysis Agent",
    description="AI-powered research assistant that analyzes uploaded documents, fetches statistical data, and provides web context",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multi-Source Analysis Agent API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload",
            "query": "/query",
            "session": "/session/{session_id}",
            "settings": "/settings",
            "health": "/health"
        },
        "setup_required": not key_status["is_valid"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "agent_ready": agent is not None,
        "keys_configured": key_status["is_valid"]
    }

@app.get("/settings", response_model=ApiKeyStatus)
async def get_settings():
    """Get current API key status and configuration"""
    # Don't return actual keys for security
    masked_keys = {}
    if settings.openai_api_key:
        masked_keys["openai_api_key"] = f"sk-...{settings.openai_api_key[-4:]}"
    if settings.tavily_api_key:
        masked_keys["tavily_api_key"] = f"tvly-...{settings.tavily_api_key[-4:]}"
    if settings.data_commons_api_key:
        masked_keys["data_commons_api_key"] = f"dc-...{settings.data_commons_api_key[-4:]}"
    if settings.langsmith_api_key:
        masked_keys["langsmith_api_key"] = f"ls-...{settings.langsmith_api_key[-4:]}"
    
    return ApiKeyStatus(
        api_keys=masked_keys,
        status=key_status,
        backend_ready=agent is not None
    )

@app.post("/settings")
async def update_settings(keys: ApiKeyUpdate):
    """Update API keys and reinitialize agent if needed"""
    global agent
    import os
    
    # Update settings and environment variables
    if keys.openai_api_key:
        settings.openai_api_key = keys.openai_api_key
        os.environ["OPENAI_API_KEY"] = keys.openai_api_key
    if keys.tavily_api_key:
        settings.tavily_api_key = keys.tavily_api_key
        os.environ["TAVILY_API_KEY"] = keys.tavily_api_key
    if keys.data_commons_api_key:
        settings.data_commons_api_key = keys.data_commons_api_key
        os.environ["DATA_COMMONS_API_KEY"] = keys.data_commons_api_key
    if keys.langsmith_api_key:
        settings.langsmith_api_key = keys.langsmith_api_key
        os.environ["LANGSMITH_API_KEY"] = keys.langsmith_api_key
    
    # Re-validate keys
    updated_status = settings.validate_required_keys()
    
    # Update global key status
    global key_status
    key_status = updated_status
    
    # Reinitialize agent if we now have all required keys
    if updated_status["is_valid"] and not agent:
        try:
            print("🚀 Initializing agent with new API keys...")
            agent = MultiSourceAnalysisAgent()
            print("✅ Agent initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize agent with provided keys: {str(e)}"
            )
    
    return {
        "success": True,
        "message": "API keys updated successfully",
        "status": updated_status,
        "agent_ready": agent is not None
    }

@app.post("/upload", response_model=List[FileUploadResponse])
async def upload_files(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Upload one or more files (PDFs, images) and process them for analysis
    
    Creates a new session if session_id is not provided
    """
    if not agent:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Please configure API keys first."
        )
    
    try:
        # Create new session if needed
        if not session_id:
            session_id = agent.create_new_session()
        
        # Validate and process files
        file_responses = []
        processed_files = []
        
        for file in files:
            # Read file content
            content = await file.read()
            
            # Validate file
            try:
                validate_file_upload(file.filename, len(content))
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            
            # Prepare file info for processing
            file_info = {
                'filename': file.filename,
                'content': content,
                'content_type': file.content_type,
                'size': len(content)
            }
            processed_files.append(file_info)
            
            # Create response
            file_response = FileUploadResponse(
                success=True,
                message=f"File {file.filename} uploaded successfully",
                file_id=str(uuid.uuid4()),
                filename=file.filename,
                size=len(content)
            )
            file_responses.append(file_response)
        
        if not processed_files:
            raise HTTPException(status_code=400, detail="No valid files uploaded")
        
        # Process files and create vectorstore
        try:
            vectorstore, file_info = agent.process_files(processed_files, session_id)
            agent.set_session_vectorstore(session_id, vectorstore, file_info)
            
            print(f"✅ Successfully processed {len(processed_files)} files for session {session_id}")
            return file_responses
            
        except Exception as e:
            print(f"❌ Error processing files: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing files: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error during upload: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during file upload: {str(e)}"
        )

@app.post("/query")
async def query_documents(
    question: str = Form(...),
    session_id: str = Form(...)
):
    """
    Query uploaded documents with streaming response
    
    Returns a streaming response with real-time updates as the agent processes
    """
    if not agent:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Please configure API keys first."
        )
    
    try:
        # Validate session
        session_info = agent.get_session_info(session_id)
        if not session_info:
            raise HTTPException(
                status_code=404, 
                detail="Session not found. Please upload documents first."
            )
        
        # Validate question
        if not question or len(question.strip()) < 1:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Stream the agent's processing
        async def generate_stream():
            async for update in agent.process_query_streaming(question, session_id):
                yield f"data: {update}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific session"""
    if not agent:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Please configure API keys first."
        )
    
    session_info = agent.get_session_info(session_id)
    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session_info

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its associated data"""
    if not agent:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Please configure API keys first."
        )
    
    success = agent.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"success": True, "message": f"Session {session_id} deleted successfully"}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    if not agent:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Please configure API keys first."
        )
    
    sessions = agent.list_sessions()
    return {"sessions": sessions}

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🔮 Policy Prism - Multi-Source Analysis Agent")
    print("="*60)
    
    if not key_status["is_valid"]:
        print("\n⚠️  Setup Required:")
        print("1. Configure API keys via the web interface")
        print("2. Or set environment variables before starting")
        print("\n🔗 Opening frontend at: http://localhost:3001")
        print("🔗 API documentation: http://localhost:8000/docs")
    else:
        print("✅ All required API keys configured")
        print("🚀 Ready for analysis!")
    
    print("\n" + "="*60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 