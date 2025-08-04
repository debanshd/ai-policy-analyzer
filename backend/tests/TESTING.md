# Testing Guide - Multi-Source Analysis Agent

Quick reference for testing the backend with documents and prompts.

## Prerequisites

1. **Set up environment:**
   ```bash
   cd backend
   uv sync --extra test  # or pip install .[test]
   ```

2. **Create `.env` file with API keys:**
   ```bash
   OPENAI_API_KEY=your_openai_key_here
   TAVILY_API_KEY=your_tavily_key_here
   ```

3. **Start the backend:**
   ```bash
   python run.py  # or python main.py
   ```

## Quick Testing Methods

### Method 1: Built-in Test Script (Easiest)
```bash
# Test with any PDF or image
python tests/test_workflow.py --file document.pdf --prompt "What are the main findings?"

# Or use the run script
python run.py --test-doc document.pdf --prompt "Summarize the key points"
```

### Method 2: Interactive API Docs (Visual)
1. Open `http://localhost:8000/docs`
2. Upload file using `/upload` endpoint
3. Copy the session_id from response
4. Query using `/query` endpoint with your prompt

### Method 3: curl Commands (Command Line)
```bash
# Upload document
curl -X POST "http://localhost:8000/upload" -F "files=@document.pdf"

# Query (replace session_id with actual ID from upload response)
curl -X POST "http://localhost:8000/query" \
  -F "question=What are the main findings?" \
  -F "session_id=your_session_id_here"
```

## What You'll See

The agent provides **real-time streaming** of its thought process:

```
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
[Comprehensive analysis combining document content, 
 statistical data insights, and web context]
============================================================
📋 Sources: document.pdf, Data Commons API, web sources
🛠️  Tools used: RAG_Tool, DataCommonsTool, TavilyTool
⏱️  Processing time: 12.34s
```

## Good Test Prompts

**General Analysis:**
- "What are the main findings of this document?"
- "Summarize the key recommendations"
- "What are the most important statistics mentioned?"

**Specific Questions:**
- "What does this document say about [specific topic]?"
- "What are the economic implications discussed?"
- "How do these findings relate to current policy trends?"

**Comparative Analysis:**
- "How do these results compare to industry standards?"
- "What are the policy implications of these findings?"
- "What are the potential risks and benefits mentioned?"

## Testing Different Document Types

**PDFs:** Research papers, policy documents, reports
**Images:** Charts, graphs, infographics, screenshots  
**Text Files:** .txt and .md files with policy content, research notes

## Troubleshooting

**❌ "Cannot connect to backend"**
- Make sure server is running: `python run.py`
- Check: `curl http://localhost:8000/health`

**❌ "No documents uploaded"**
- Upload file first using `/upload` endpoint
- Ensure session_id matches upload response

**❌ "API key error"**
- Check `.env` file has valid OPENAI_API_KEY
- Verify API key has sufficient credits

**❌ "File upload failed"**
- Check file size (max 50MB)
- Ensure supported format (.pdf, .png, .jpg, .jpeg, .txt, .md)

**❌ "Empty or error responses"**
- Verify TAVILY_API_KEY is set
- Check document was processed successfully

## Sample Document

Use `sample_document.txt` for testing if you don't have your own documents:

```bash
# Convert sample to PDF first, or test with any policy/research document
python tests/test_workflow.py --file sample_document.txt --prompt "What are the main policy recommendations?"
```

## Advanced Testing

**Multiple Documents:**
```bash
# Upload multiple files to same session
curl -X POST "http://localhost:8000/upload" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "files=@chart.png"
```

**Session Management:**
```bash
# List active sessions
curl http://localhost:8000/sessions

# Get session info
curl http://localhost:8000/session/your_session_id

# Delete session
curl -X DELETE http://localhost:8000/session/your_session_id
```

## Expected Performance

- **Upload time:** 1-5 seconds per document
- **Query processing:** 10-30 seconds depending on complexity
- **Response quality:** Combines document analysis + live data + web context
- **Sources:** Always shows which tools and sources were used

Happy testing! 🚀 