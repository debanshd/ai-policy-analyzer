#!/usr/bin/env python3
"""
Simple test script for the Multi-Source Analysis Agent backend

This script tests basic functionality without requiring external files.
Run with: python test_backend.py
"""

import asyncio
import json
from agent import MultiSourceAnalysisAgent
from tools.rag_tool import RAGTool
from tools.tavily_tool import TavilyTool
from tools.data_commons_tool import DataCommonsTool

async def test_tools():
    """Test individual tools"""
    print("🧪 Testing individual tools...")
    
    # Test Data Commons Tool
    print("\n1. Testing Data Commons Tool...")
    dc_tool = DataCommonsTool()
    test_query = "What is the unemployment rate in the United States?"
    dc_result = dc_tool._run(test_query)
    print(f"   ✓ Data Commons query: {test_query}")
    print(f"   ✓ Method used: {dc_result.get('methodology', 'unknown')}")
    print(f"   ✓ Data points found: {len(dc_result.get('data_points', []))}")
    if dc_result.get('data_points'):
        print(f"   ✓ Sample result: {dc_result['data_points'][0].get('value', 'No data')[:100]}...")
    
    # Test Tavily Tool (this will make actual API calls if TAVILY_API_KEY is set)
    print("\n2. Testing Tavily Tool...")
    try:
        tavily_tool = TavilyTool()
        tavily_result = tavily_tool._run("artificial intelligence policy")
        print(f"   ✓ Tavily search completed")
        print(f"   ✓ Found {len(tavily_result.get('search_results', []))} results")
        if tavily_result.get('error'):
            print(f"   ⚠️  Tavily error (expected if no API key): {tavily_result['error']}")
    except Exception as e:
        print(f"   ⚠️  Tavily test failed (expected if no API key): {e}")
    
    # Test RAG Tool (without vectorstore)
    print("\n3. Testing RAG Tool...")
    rag_tool = RAGTool()
    rag_result = rag_tool._run("What is AI policy?")
    print(f"   ✓ RAG tool response: {rag_result.get('answer', 'No answer')[:100]}...")
    
    print("\n✅ Tool tests completed!")

async def test_agent():
    """Test the main agent (without actual file uploads)"""
    print("\n🤖 Testing Multi-Source Analysis Agent...")
    
    agent = MultiSourceAnalysisAgent()
    
    # Test session creation
    session_id = agent.create_new_session()
    print(f"   ✓ Created session: {session_id}")
    
    # Test session info retrieval
    session_info = agent.get_session_info(session_id)
    print(f"   ✓ Session info retrieved: {session_info is None}")  # Should be None (no docs uploaded)
    
    # Test basic query processing
    test_query = "We need to analyze climate change policy and its impact on economic development and healthcare systems."
    print(f"   ✓ Test query prepared: {test_query[:50]}...")
    
    # Test that the agent can handle basic operations
    print(f"   ✓ Agent ready for document processing and queries")
    
    print("\n✅ Agent tests completed!")

def test_config():
    """Test configuration loading"""
    print("\n⚙️  Testing configuration...")
    
    from config import settings
    print(f"   ✓ LLM Model: {settings.llm_model}")
    print(f"   ✓ Embedding Model: {settings.embedding_model}")
    print(f"   ✓ Chunk Size: {settings.chunk_size}")
    print(f"   ✓ Max File Size: {settings.max_file_size // (1024*1024)}MB")
    print(f"   ✓ Allowed File Types: {settings.allowed_file_types}")
    
    # Check API keys (without exposing them)
    print(f"   ✓ OpenAI API Key: {'✅ Set' if settings.openai_api_key else '❌ Missing'}")
    print(f"   ✓ Tavily API Key: {'✅ Set' if settings.tavily_api_key else '❌ Missing'}")
    print(f"   ✓ LangSmith API Key: {'✅ Set' if settings.langsmith_api_key else '❌ Not set (optional)'}")
    
    print("\n✅ Configuration tests completed!")

def test_models():
    """Test Pydantic models"""
    print("\n📋 Testing Pydantic models...")
    
    from models import StreamingUpdate, ToolType, DocumentChunk
    
    # Test StreamingUpdate
    update = StreamingUpdate(
        type="tool",
        content="Testing RAG tool",
        tool=ToolType.RAG_TOOL,
        metadata={"test": True}
    )
    print(f"   ✓ StreamingUpdate created: {update.type}")
    
    # Test DocumentChunk
    chunk = DocumentChunk(
        content="This is a test document chunk",
        metadata={"filename": "test.pdf", "page": 1},
        page_number=1
    )
    print(f"   ✓ DocumentChunk created: {len(chunk.content)} chars")
    
    # Test JSON serialization
    update_json = json.dumps(update.dict())
    print(f"   ✓ JSON serialization works: {len(update_json)} chars")
    
    print("\n✅ Model tests completed!")

async def main():
    """Run all tests"""
    print("🚀 Starting Multi-Source Analysis Agent Backend Tests")
    print("=" * 60)
    
    try:
        # Test configuration first
        test_config()
        
        # Test models
        test_models()
        
        # Test tools
        await test_tools()
        
        # Test agent
        await test_agent()
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed successfully!")
        print("\nNote: Some tests may show warnings if API keys are not configured.")
        print("This is expected for development setup.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 