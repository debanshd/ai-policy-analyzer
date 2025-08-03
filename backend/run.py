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
    
    try:
        from test_backend import main as test_main
        await test_main()
    except ImportError:
        print("❌ test_backend.py not found")
    except Exception as e:
        print(f"❌ Tests failed: {e}")

def run_env_test():
    """Run environment test"""
    print("🔧 Running Environment Test")
    print("-" * 30)
    
    try:
        from test_env import test_environment
        test_environment()
    except ImportError:
        print("❌ test_env.py not found")
    except Exception as e:
        print(f"❌ Environment test failed: {e}")

def run_python_check():
    """Run Python version check"""
    print("🐍 Running Python Version Check")
    print("-" * 35)
    
    try:
        from check_python import check_python_version
        check_python_version()
    except ImportError:
        print("❌ check_python.py not found")
    except Exception as e:
        print(f"❌ Python check failed: {e}")

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
            "test_with_document.py", 
            "--file", file_path, 
            "--prompt", prompt
        ], capture_output=False)
        
        if result.returncode != 0:
            print(f"❌ Document test failed with exit code {result.returncode}")
        
    except ImportError:
        print("❌ test_with_document.py not found")
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
            "test_with_document_fast.py", 
            "--file", file_path, 
            "--prompt", prompt
        ], capture_output=False)
        
        if result.returncode != 0:
            print(f"❌ Fast test failed with exit code {result.returncode}")
        
    except ImportError:
        print("❌ test_with_document_fast.py not found")
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