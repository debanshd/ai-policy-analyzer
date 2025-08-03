#!/usr/bin/env python3
"""
Simple environment test to check if .env file is loaded correctly
"""

import os
from dotenv import load_dotenv

def test_environment():
    print("🔧 Environment Test")
    print("=" * 30)
    
    # Load environment variables
    print("📂 Loading .env file...")
    load_dotenv()
    
    # Check for .env file existence
    if os.path.exists('.env'):
        print("✅ .env file found")
        
        # Read and show file size
        with open('.env', 'r') as f:
            content = f.read()
            lines = content.strip().split('\n')
            non_empty_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
            
        print(f"   📄 File size: {len(content)} bytes")
        print(f"   📋 Total lines: {len(lines)}")
        print(f"   🔑 Config lines: {len(non_empty_lines)}")
    else:
        print("❌ .env file not found")
        return False
    
    # Check specific environment variables
    print("\n🔑 Checking API keys...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"✅ OpenAI API key: {openai_key[:12]}...{openai_key[-12:]} ({len(openai_key)} chars)")
    else:
        print("❌ OPENAI_API_KEY not found")
    
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        print(f"✅ Tavily API key: {tavily_key[:12]}...{tavily_key[-12:]} ({len(tavily_key)} chars)")
    else:
        print("❌ TAVILY_API_KEY not found")
    
    # Check other config
    print("\n⚙️ Checking other config...")
    llm_model = os.getenv("LLM_MODEL", "default")
    print(f"✅ LLM Model: {llm_model}")
    
    embed_model = os.getenv("EMBEDDING_MODEL", "default")
    print(f"✅ Embedding Model: {embed_model}")
    
    # Test if we can import the main components
    print("\n📦 Testing imports...")
    
    try:
        from config import settings
        print(f"✅ Config imported: LLM={settings.llm_model}")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from langchain_openai import OpenAIEmbeddings
        print("✅ OpenAI embeddings imported")
    except Exception as e:
        print(f"❌ OpenAI import failed: {e}")
        return False
        
    try:
        from langchain_openai import ChatOpenAI
        print("✅ ChatOpenAI imported")
    except Exception as e:
        print(f"❌ ChatOpenAI import failed: {e}")
        return False
    
    print("\n✅ Environment test completed!")
    
    if openai_key and tavily_key:
        print("🚀 Ready to run full tests!")
        return True
    else:
        print("⚠️ Missing API keys - some tests may fail")
        return False

if __name__ == "__main__":
    test_environment() 