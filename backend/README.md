# Multi-Source Analysis Agent - Backend

This is the FastAPI backend for the Multi-Source Analysis Agent, an AI-powered research assistant that analyzes uploaded documents, fetches statistical data, and provides web context for policy analysts.

## Features

- **Document Processing**: Upload and analyze PDF, image, and text files
- **RAG (Retrieval-Augmented Generation)**: Query uploaded documents using advanced retrieval techniques
- **Real Statistical Data**: Uses **data_gemma** library for actual Data Commons statistical data via RIG/RAG methodologies
- **Web Search**: Get broader context and definitions using Tavily search
- **Real-time Streaming**: See the agent's thought process in real-time
- **Session Management**: Maintain context across multiple queries
- **LangSmith Integration**: Monitor performance, cost, and latency

## Technology Stack

- **Framework**: FastAPI with async support
- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Database**: Qdrant (in-memory for development)
- **Document Processing**: PyMuPDF for PDFs, PIL for images, plain text for TXT/MD
- **Statistical Data**: data_gemma (RIG/RAG methodologies for Data Commons)
- **Web Search**: Tavily API
- **Monitoring**: LangSmith
- **Orchestration**: LangChain

## Requirements

- **Python 3.10+** (required for data_gemma integration)
- OpenAI API key (required)
- Tavily API key (required)

## Setup

### 1. Check Python Version

```bash
python --version  # Should be 3.10+
```

If you need to upgrade Python:
- **Direct install**: Download from [python.org](https://python.org)
- **Using pyenv**: `pyenv install 3.10.12 && pyenv local 3.10.12`
- **Using conda**: `conda create -n policy-agent python=3.10`

### 2. Install uv (Optional but Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver, written in Rust. It's significantly faster than pip.

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

### 3. Install Dependencies

```bash
cd backend

# Basic installation (without data_gemma)
uv sync

# Or using pip
pip install .

# For development dependencies
uv sync --extra dev

# For real Data Commons integration (requires Python 3.10+)
uv sync --extra datacommons
```

### 4. Environment Configuration

Create a `.env` file in the backend directory with the following required variables:

```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Required for REAL Data Commons (optional - will use simulation otherwise)
DATA_COMMONS_API_KEY=your_data_commons_api_key_here

# Optional: LangSmith for monitoring
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=multi-source-analysis-agent
LANGSMITH_TRACING=true

# Optional: Model configuration (defaults provided)
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Optional: File upload configuration
MAX_FILE_SIZE=52428800  # 50MB
UPLOAD_DIRECTORY=uploads

# Optional: CORS configuration
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
```

### 5. Install data_gemma for Real Data Commons Integration (Optional)

The agent can use the **data_gemma** library for real statistical data from Data Commons, or fall back to simulation mode.

**Check Python version first:**
```bash
python check_python.py  # Must show Python 3.10+
```

**Install data_gemma (choose one method):**

```bash
# Option 1: Using uv with optional dependency (recommended)
uv sync --extra datacommons

# Option 2: Use the install script
python install_data_gemma.py

# Option 3: Manual installation
uv add "data-gemma@git+https://github.com/datacommonsorg/llm-tools.git"
# or with pip:
pip install git+https://github.com/datacommonsorg/llm-tools.git
```

**Note**: 
- **Requires Python 3.10+** (data_gemma limitation)
- **Large installation**: Includes Google Cloud AI, transformers, and PyTorch (~2-3GB)
- data_gemma will download DataGemma models automatically on first use
- Initial model download may take several minutes  
- The agent will gracefully fall back to simulation mode if data_gemma is not available

**Installation size comparison:**
- Basic installation (simulation mode): ~200MB
- Full installation with data_gemma: ~2-3GB (includes ML models)

### 6. Get API Keys

- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Tavily API Key**: Get from [Tavily](https://tavily.com)
- **Data Commons API Key** (for real data): Get from [Data Commons API](https://datacommons.org/api)
- **LangSmith API Key** (optional): Get from [LangSmith](https://smith.langchain.com/)

### 7. Run the Application

```bash
# Development mode with auto-reload
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- **Interactive API docs**: `http://localhost:8000/docs`
- **ReDoc documentation**: `http://localhost:8000/redoc`

## API Endpoints

### Core Endpoints

- `POST /upload` - Upload files for analysis
- `POST /query` - Query documents with streaming response
- `GET /session/{session_id}` - Get session information
- `DELETE /session/{session_id}` - Delete a session
- `GET /sessions` - List all active sessions

### Utility Endpoints

- `GET /` - API information
- `GET /health` - Health check

## Usage Example

### 1. Upload Documents

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document.pdf" \
  -F "files=@image.png"
```

### 2. Query with Streaming

```bash
curl -X POST "http://localhost:8000/query" \
  -F "question=What are the main findings about climate policy?" \
  -F "session_id=your_session_id"
```

## Testing with Documents

### Method 1: Using the Test Script (Recommended)

```bash
# Install test dependencies
uv sync --extra test

# Test with a PDF document
python tests/test_workflow.py --file document.pdf --prompt "What are the main findings?"

# Test with multiple prompts on the same document
python tests/test_workflow.py --file document.pdf --prompt "Summarize the key points"
python tests/test_workflow.py --file document.pdf --prompt "What are the policy recommendations?" --session existing_session_id
```

Example output:
```
🚀 Testing Multi-Source Analysis Agent
==================================================
✅ Backend is healthy: healthy
📄 Uploading document: document.pdf
✅ Upload successful!
   📁 document.pdf (245760 bytes)
🔑 Session ID: 12345678-1234-5678-9012-123456789abc

🤖 Querying with prompt: 'What are the main findings?'
📡 Streaming agent response...

💭 Starting comprehensive analysis...
📚 [RAG_Tool] Analyzing uploaded documents...
✅ Found relevant information in 3 document sections
📊 [DataCommonsTool] Extracting key entities for statistical data search...
✅ Identified 5 key entities for statistical analysis
🌐 [TavilyTool] Searching web for broader context and definitions...
✅ Found 8 web sources
💭 Synthesizing comprehensive analysis...

🎯 FINAL ANSWER:
============================================================
Based on the uploaded document analysis, the main findings include...
[comprehensive answer here]
============================================================
📋 Sources: document.pdf, Data Commons API, news.example.com
🛠️  Tools used: RAG_Tool, DataCommonsTool, TavilyTool
⏱️  Processing time: 12.34s
```

### Method 2: Using curl (Command Line)

```bash
# 1. Upload a document
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document.pdf"

# Response will include session_id, e.g.:
# [{"success": true, "message": "File document.pdf uploaded successfully (Session: abc123)", ...}]

# 2. Query the document (streaming response)
curl -X POST "http://localhost:8000/query" \
  -F "question=What are the main findings?" \
  -F "session_id=abc123"
```

### Method 3: Using the Interactive API Documentation

1. Start the backend: `python run.py`
2. Open your browser to `http://localhost:8000/docs`
3. Use the **Upload** endpoint:
   - Click "Try it out"
   - Select your PDF/image file
   - Click "Execute"
   - Note the `session_id` from the response
4. Use the **Query** endpoint:
   - Enter your question and the session_id
   - Click "Execute" to see the streaming response

### Method 4: Test Multiple Documents

```bash
# Upload multiple documents to the same session
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf" \
  -F "files=@chart.png"

# Query across all uploaded documents
python tests/test_workflow.py --file document1.pdf --prompt "Compare findings from all uploaded documents" --session your_session_id
```

### Quick Test (No Document Required)

If you don't have a document ready, you can test the basic functionality:

```bash
# Test without documents (will show appropriate error)
python tests/test_workflow.py --file sample_document.txt --prompt "What are the main policy recommendations?"

# Use the provided sample document (convert to PDF first if needed)
# The sample_document.txt contains renewable energy policy content for testing
```

### Example Test Questions

Once you have a document uploaded, try these types of questions:

**Analysis Questions:**
- "What are the main findings of this document?"
- "Summarize the key recommendations"
- "What statistical data is presented?"

**Specific Queries:**
- "What does this document say about [specific topic]?"
- "How does this relate to current policy trends?"
- "What are the economic implications mentioned?"

**Comparative Analysis:**
- "How do these findings compare to current industry standards?"
- "What are the policy implications of these results?"

### Troubleshooting

**Backend not responding?**
```bash
# Check if backend is running
curl http://localhost:8000/health

# Start backend if needed
python run.py
```

**Upload failing?**
- Check file size (max 50MB)
- Ensure file type is supported (.pdf, .png, .jpg, .jpeg, .txt, .md)
- Verify API keys are set in `.env` file

**Empty responses?**
- Ensure OPENAI_API_KEY is valid and has credits
- Check TAVILY_API_KEY is set (required for web search)
- Verify document was processed successfully (check upload response)

**Python/data_gemma issues?**
```bash
# Check Python version compatibility
python run.py --check-python

# Test environment setup
python run.py --test-env

# If data_gemma fails to install (Python < 3.10):
# The agent will use simulation mode automatically
# No action needed - functionality will be preserved
```

**data_gemma installation error?**
- **Most common**: Python version < 3.10 (upgrade Python or use simulation mode)
- **Network issues**: Try manual installation: `pip install git+https://github.com/datacommonsorg/llm-tools.git`
- **Permission issues**: Try with `--user` flag: `pip install --user git+...`
```

## Project Structure

```
backend/
├── main.py                 # FastAPI application entry point
├── config.py              # Configuration settings
├── models.py              # Pydantic models
├── agent.py               # Main agent orchestration
├── pyproject.toml         # Project configuration and dependencies
├── run.py                 # Development server script
├── test_backend.py        # Test suite
├── utils/
│   ├── __init__.py
│   └── document_processor.py  # Document processing utilities
└── tools/
    ├── __init__.py
    ├── rag_tool.py        # RAG implementation
    ├── tavily_tool.py     # Web search tool
    └── data_commons_tool.py   # Statistical data tool (placeholder)
```

## Agent Workflow

The Multi-Source Analysis Agent follows a three-step process:

1. **Analyze User Context**: Uses RAG to find relevant information in uploaded documents
2. **Augment with Live Data**: Extracts key entities and fetches statistical data
3. **Enrich with Web Context**: Searches for definitions and broader context

## Development Notes

### Chunking Strategy

Based on the notebook patterns, we use:
- `RecursiveCharacterTextSplitter`
- Chunk size: 1000 characters
- Overlap: 200 characters

### Vector Database

- Uses Qdrant in-memory mode for development
- Collection per session: `user_documents_{session_id}`
- OpenAI embeddings with `text-embedding-3-small`

### Streaming Implementation

The agent provides real-time updates during processing:
- `thought`: Agent reasoning steps
- `tool`: Current tool being used
- `result`: Intermediate results
- `error`: Error messages

### Session Management

- Sessions automatically expire after 24 hours
- Background cleanup task runs hourly
- Each session maintains its own vectorstore

## Production Considerations

1. **Vector Database**: Replace in-memory Qdrant with persistent storage
2. **File Storage**: Implement proper file storage (AWS S3, etc.)
3. **Authentication**: Add user authentication and authorization
4. **Rate Limiting**: Implement API rate limiting
5. **Caching**: Add Redis for session and query caching
6. **Data Commons**: Implement actual Data Commons API integration
7. **Error Handling**: Enhanced error reporting and logging
8. **Scaling**: Consider async processing with Celery for heavy workloads

## Monitoring

When LangSmith is configured, the application automatically tracks:
- Query performance and latency
- Token usage and costs
- Tool execution times
- Error rates and debugging information

Visit your LangSmith dashboard to view detailed analytics.

## License

This project is part of the AI Policy Analyzer certification challenge.
