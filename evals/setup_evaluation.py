#!/usr/bin/env python3
"""
Setup script for Golden Dataset Evaluation

This script validates the environment and prepares for running the consolidated
golden_dataset_evaluation.py implementation.

Requires backend integration - fails if backend components are not available.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_uv_installation():
    """Check if uv is installed"""
    print("🔍 Checking uv installation...")
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ uv installed: {result.stdout.strip()}")
            return True
        else:
            print("   ❌ uv not found")
            return False
    except FileNotFoundError:
        print("   ❌ uv not found")
        return False

def install_dependencies():
    """Install dependencies using uv"""
    print("\n📦 Installing dependencies...")
    try:
        result = subprocess.run(['uv', 'sync'], check=True, capture_output=True, text=True)
        print("   ✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install dependencies: {e}")
        print(f"   Error output: {e.stderr}")
        return False

def check_backend_integration():
    """Check backend integration - REQUIRED"""
    print("\n🔧 Checking backend integration (REQUIRED)...")
    
    backend_path = Path(__file__).parent.parent / "backend"
    if not backend_path.exists():
        print("   ❌ Backend directory not found")
        print("   Backend integration is REQUIRED for evaluation")
        return False
    
    print(f"   📁 Backend directory: {backend_path}")
    
    # Add backend to path
    sys.path.append(str(backend_path))
    
    # Check required backend components
    try:
        from tools.rag_tool import RAGTool
        print("   ✅ Backend RAG tool available")
    except ImportError as e:
        print(f"   ❌ Backend RAG tool not available: {e}")
        return False
    
    try:
        from utils.document_processor import DocumentProcessor
        print("   ✅ Backend document processor available")
    except ImportError as e:
        print(f"   ❌ Backend document processor not available: {e}")
        return False
    
    try:
        from config import settings
        print("   ✅ Backend configuration available")
        print(f"      • Qdrant location: {settings.qdrant_location}")
        print(f"      • Collection name: {settings.qdrant_collection_name}")
        print(f"      • LLM model: {settings.llm_model}")
        print(f"      • Embedding model: {settings.embedding_model}")
    except ImportError as e:
        print(f"   ❌ Backend configuration not available: {e}")
        return False
    
    print("   ✅ All backend components available")
    return True

def create_env_template():
    """Create .env template if it doesn't exist"""
    print("\n📝 Checking environment configuration...")
    
    env_file = Path(".env")
    if env_file.exists():
        print(f"   ✅ .env file exists: {env_file.absolute()}")
        return True
    
    env_template = """# Policy Analyzer Evaluation Environment Variables
# Copy from your backend .env or set these values

# Required for evaluation
OPENAI_API_KEY=your_openai_api_key_here

# Optional for LangSmith tracking
# LANGSMITH_API_KEY=your_langsmith_api_key_here
# LANGSMITH_PROJECT=policy-analyzer-evals
"""
    
    with open(env_file, 'w') as f:
        f.write(env_template)
    
    print(f"   📝 Created .env template: {env_file.absolute()}")
    print("   ⚠️  Please edit this file with your actual API keys")
    return False

def check_documents_directory():
    """Check if documents directory exists and has files"""
    print("\n📄 Checking documents...")
    
    docs_dir = Path("../documents")
    if not docs_dir.exists():
        print(f"   📁 Creating documents directory: {docs_dir.absolute()}")
        docs_dir.mkdir(parents=True, exist_ok=True)
        print("   ⚠️  Please add PDF or TXT policy documents to this directory")
        return False
    
    files = list(docs_dir.glob("*.pdf")) + list(docs_dir.glob("*.txt"))
    if not files:
        print(f"   ⚠️  No documents found in: {docs_dir.absolute()}")
        print("   Please add PDF or TXT policy documents")
        return False
    
    print(f"   ✅ Found {len(files)} document(s):")
    for file in files[:5]:  # Show first 5
        print(f"      • {file.name}")
    if len(files) > 5:
        print(f"      • ... and {len(files) - 5} more")
    
    return True

def test_imports():
    """Test if we can import required packages"""
    print("\n🧪 Testing package imports...")
    
    packages = [
        ("langchain_openai", "LangChain OpenAI"),
        ("ragas", "RAGAS"),
        ("qdrant_client", "Qdrant Client"),
        ("datasets", "Datasets"),
        ("pydantic", "Pydantic")
    ]
    
    all_good = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} not available")
            all_good = False
    
    return all_good

def check_api_keys():
    """Check if required API keys are configured"""
    print("\n🔑 Checking API key configuration...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("   ⚠️  No .env file found")
        return False
    
    # Load .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "your_openai_api_key_here":
        print("   ⚠️  OPENAI_API_KEY not configured")
        return False
    
    print("   ✅ OPENAI_API_KEY configured")
    
    # Check optional LangSmith key
    langsmith_key = os.getenv("LANGSMITH_API_KEY")
    if langsmith_key and langsmith_key != "your_langsmith_api_key_here":
        print("   ✅ LANGSMITH_API_KEY configured (optional)")
    else:
        print("   ⚠️  LANGSMITH_API_KEY not configured (optional - LangSmith features will be skipped)")
    
    return True

def main():
    """Main setup validation"""
    print("=" * 60)
    print("Golden Dataset Evaluation - Setup Validation")
    print("BACKEND INTEGRATION REQUIRED")
    print("=" * 60)
    
    checks = [
        ("UV Installation", check_uv_installation),
        ("Dependencies", install_dependencies),
        ("Environment Template", create_env_template),
        ("Backend Integration", check_backend_integration),
        ("Documents Directory", check_documents_directory),
        ("Package Imports", test_imports),
        ("API Keys", check_api_keys)
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    print("\n" + "=" * 60)
    print("Setup Summary")
    print("=" * 60)
    
    critical_checks = ["UV Installation", "Dependencies", "Backend Integration", "Package Imports"]
    has_critical_failures = False
    
    for name, success in results.items():
        if name in critical_checks and not success:
            has_critical_failures = True
        
        status = "✅ PASS" if success else "❌ FAIL" if name in critical_checks else "⚠️ NEEDS ATTENTION"
        print(f"{name:20s}: {status}")
    
    if has_critical_failures:
        print(f"\n❌ CRITICAL FAILURES - Cannot run evaluation")
        print(f"\nRequired fixes:")
        if not results["UV Installation"]:
            print(f"   • Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
        if not results["Backend Integration"]:
            print(f"   • Ensure backend directory exists at ../backend/")
            print(f"   • Run 'cd ../backend && uv sync' to setup backend")
        if not results["Package Imports"]:
            print(f"   • Run 'uv sync' to install missing packages")
        
        sys.exit(1)
    
    elif not all(results.values()):
        print(f"\n⚠️  Some optional features need attention")
        print(f"\nOptional fixes:")
        if not results["Environment Template"]:
            print(f"   • Edit .env file with your API keys")
        if not results["Documents Directory"]:
            print(f"   • Add policy documents to ../documents/")
        if not results["API Keys"]:
            print(f"   • Configure OPENAI_API_KEY in .env file")
        
        print(f"\n🎯 You can run evaluation but some features may be limited")
    
    else:
        print(f"\n🎉 Setup complete! Ready to run evaluation:")
    
    print(f"\n📖 Usage:")
    print(f"   uv run golden_dataset_evaluation.py")
    print(f"   uv run golden_dataset_evaluation.py --testset-size 30")
    print(f"   uv run golden_dataset_evaluation.py --skip-langsmith")

if __name__ == "__main__":
    main() 