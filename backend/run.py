#!/usr/bin/env python3
"""
Simple run script for the Multi-Source Analysis Agent backend

Usage:
    python run.py              # Start development server
    python run.py --prod        # Start production server
    python run.py --test        # Run tests
    python run.py --help        # Show help
"""

import argparse
import asyncio
import uvicorn
from config import settings

def run_dev_server():
    """Run development server with auto-reload"""
    print(f"🚀 Starting Multi-Source Analysis Agent (Development Mode)")
    print(f"📊 Using model: {settings.llm_model}")
    print(f"🔧 Environment: Development")
    print(f"📍 URL: http://localhost:8000")
    print(f"📖 Docs: http://localhost:8000/docs")
    print("-" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

def run_prod_server():
    """Run production server"""
    print(f"🚀 Starting Multi-Source Analysis Agent (Production Mode)")
    print(f"📊 Using model: {settings.llm_model}")
    print(f"🔧 Environment: Production")
    print(f"📍 URL: http://localhost:8000")
    print("-" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,  # Adjust based on CPU cores
        log_level="warning"
    )

async def run_tests():
    """Run the test suite"""
    print("🧪 Running Multi-Source Analysis Agent Tests")
    print("-" * 50)
    print("💡 Use the following commands for comprehensive testing:")
    print()
    print("   🌐 Full workflow test:")
    print("      uv run python tests/test_workflow.py --file sample_document.txt --prompt 'Your question'")
    print()
    print("   ⚡ Quick test:")
    print("      uv run python tests/test_workflow_fast.py --file sample_document.txt --prompt 'Your question'")
    print()
    print("   🔧 Environment test:")
    print("      uv run python run.py --test-env")
    print()
    print("   🐍 Python version check:")
    print("      uv run python run.py --check-python")
    print()
    print("✅ Use the commands above for targeted testing")

def run_env_test():
    """Run environment test"""
    print("🔧 Environment Test")
    print("=" * 30)
    
    import os
    from dotenv import load_dotenv
    
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

def run_python_check():
    """Run Python version check"""
    import sys
    
    print("🐍 Python Version Check")
    print("=" * 25)
    
    version = sys.version_info
    current_version = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"Current Python version: {current_version}")
    print(f"Full version info: {sys.version}")
    
    if version >= (3, 10):
        print("✅ Python version is compatible with data_gemma!")
        print("🚀 You can proceed with installation")
        return True
    else:
        print("❌ Python version is too old for data_gemma")
        print("📝 data_gemma requires Python 3.10 or higher")
        print()
        print("🔧 Upgrade options:")
        print("1. Direct install: https://python.org")
        print("2. Using pyenv: pyenv install 3.10.12 && pyenv local 3.10.12")
        print("3. Using conda: conda create -n policy-agent python=3.10")
        return False

async def run_document_test(file_path: str, prompt: str):
    """Run document test"""
    print("📄 Running Multi-Source Analysis Agent Document Test")
    print("-" * 50)
    
    try:
        import subprocess
        import sys
        
        # Run the test script with the provided arguments
        result = subprocess.run([
            sys.executable, 
            "tests/test_workflow.py", 
            "--file", file_path, 
            "--prompt", prompt
        ], capture_output=False)
        
        if result.returncode != 0:
            print(f"❌ Document test failed with exit code {result.returncode}")
        
    except ImportError:
        print("❌ tests/test_workflow.py not found")
    except Exception as e:
        print(f"❌ Document test failed: {e}")

async def run_fast_test(file_path: str, prompt: str):
    """Run fast document test"""
    print("⚡ Running Multi-Source Analysis Agent Fast Test")
    print("-" * 50)
    
    try:
        import subprocess
        import sys
        
        # Run the fast test script with the provided arguments
        result = subprocess.run([
            sys.executable, 
            "tests/test_workflow_fast.py", 
            "--file", file_path, 
            "--prompt", prompt
        ], capture_output=False)
        
        if result.returncode != 0:
            print(f"❌ Fast test failed with exit code {result.returncode}")
        
    except ImportError:
        print("❌ tests/test_workflow_fast.py not found")
    except Exception as e:
        print(f"❌ Fast test failed: {e}")

def show_help():
    """Show help information"""
    help_text = """
Multi-Source Analysis Agent - Backend

Usage:
    python run.py              Start development server (default)
    python run.py --dev         Start development server with auto-reload
    python run.py --prod        Start production server
    python run.py --test        Run test suite
    python run.py --test-env    Test environment setup (.env file)
    python run.py --check-python  Check Python version for data_gemma compatibility
    python run.py --test-doc FILE --prompt "TEXT"  Test with document (full)
    python run.py --test-fast FILE --prompt "TEXT"  Fast test with document (RAG only)
    python run.py --help        Show this help

Environment Setup:
    1. Create a .env file with required API keys:
       OPENAI_API_KEY=your_key_here
       TAVILY_API_KEY=your_key_here
    
    2. Install dependencies:
       uv sync  # or pip install .
    
    3. Run the server:
       python run.py

API Endpoints:
    GET  /                      API information
    GET  /health               Health check
    GET  /docs                 Interactive API documentation
    POST /upload               Upload files for analysis
    POST /query                Query documents (streaming response)
    GET  /session/{id}         Get session information
    DELETE /session/{id}       Delete session

For more information, visit: http://localhost:8000/docs
"""
    print(help_text)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-Source Analysis Agent Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--dev", action="store_true", help="Run development server")
    parser.add_argument("--prod", action="store_true", help="Run production server")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--test-env", action="store_true", help="Test environment setup")
    parser.add_argument("--check-python", action="store_true", help="Check Python version compatibility")
    parser.add_argument("--test-doc", metavar="FILE", help="Test with document file")
    parser.add_argument("--test-fast", metavar="FILE", help="Fast test with document file (RAG only)")
    parser.add_argument("--prompt", metavar="TEXT", help="Prompt for document testing")
    parser.add_argument("--help-extended", action="store_true", help="Show extended help")
    
    args = parser.parse_args()
    
    if args.help_extended:
        show_help()
    elif args.prod:
        run_prod_server()
    elif args.test:
        asyncio.run(run_tests())
    elif args.test_env:
        run_env_test()
    elif args.check_python:
        run_python_check()
    elif args.test_doc:
        if not args.prompt:
            print("❌ --prompt is required when using --test-doc")
            return
        asyncio.run(run_document_test(args.test_doc, args.prompt))
    elif args.test_fast:
        if not args.prompt:
            print("❌ --prompt is required when using --test-fast")
            return
        asyncio.run(run_fast_test(args.test_fast, args.prompt))
    else:
        # Default to development server
        run_dev_server()

if __name__ == "__main__":
    main() 