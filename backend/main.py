import os
import uuid
import asyncio
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import settings
from models import (
    QueryRequest, FileUploadResponse, ErrorResponse, 
    StreamingUpdate, AgentResponse
)
from utils.document_processor import DocumentProcessor, validate_file_upload
from agent import MultiSourceAnalysisAgent

# Global instances
agent = MultiSourceAnalysisAgent()
document_processor = DocumentProcessor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the FastAPI app"""
    # Startup
    print("🚀 Multi-Source Analysis Agent starting up...")
    
    # Create upload directory if it doesn't exist
    os.makedirs(settings.upload_directory, exist_ok=True)
    
    # Start background task for session cleanup
    cleanup_task = asyncio.create_task(cleanup_sessions_periodically())
    
    yield
    
    # Shutdown
    print("🔄 Multi-Source Analysis Agent shutting down...")
    cleanup_task.cancel()

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
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "version": "1.0.0"
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
            vectorstore = document_processor.process_uploaded_files(processed_files, session_id)
            
            # Store in agent session
            agent.set_session_vectorstore(session_id, vectorstore, processed_files)
            
            # Add session_id to all responses
            for response in file_responses:
                response.message += f" (Session: {session_id})"
            
            return file_responses
            
        except Exception as e:
            print(f"Error processing files: {str(e)}")  # Add logging
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to process files: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
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
    try:
        session_info = agent.get_session_info(session_id)
        
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Return sanitized session info (without vectorstore object)
        return {
            "session_id": session_id,
            "files": session_info.get('files', []),
            "created_at": session_info.get('created_at'),
            "status": "active"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving session info: {str(e)}"
        )

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session and its data"""
    try:
        session_info = agent.get_session_info(session_id)
        
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Remove from agent's session store
        if session_id in agent.session_stores:
            del agent.session_stores[session_id]
        
        return {"message": f"Session {session_id} deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting session: {str(e)}"
        )

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    try:
        sessions = []
        for session_id, session_data in agent.session_stores.items():
            sessions.append({
                "session_id": session_id,
                "file_count": len(session_data.get('files', [])),
                "created_at": session_data.get('created_at'),
                "status": "active"
            })
        
        return {"sessions": sessions, "total": len(sessions)}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing sessions: {str(e)}"
        )

# Background task for session cleanup
async def cleanup_sessions_periodically():
    """Background task to clean up old sessions"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            agent.cleanup_old_sessions(max_age_hours=24)
            print(f"✓ Session cleanup completed. Active sessions: {len(agent.session_stores)}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error during session cleanup: {e}")

# Custom exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return ErrorResponse(
        message=exc.detail,
        details={"status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """General exception handler for unexpected errors"""
    return ErrorResponse(
        message="An unexpected error occurred",
        details={"error": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    
    print(f"🚀 Starting Multi-Source Analysis Agent...")
    print(f"📊 Using model: {settings.llm_model}")
    print(f"🔧 Environment: Development")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 