#!/usr/bin/env python3
"""
Install script for data_gemma library from Data Commons
"""

import subprocess
import sys
import os

def install_data_gemma():
    """Install data_gemma library"""
    print("🔧 Installing data_gemma library...")
    print("📦 This may take a few minutes...")
    
    # Check Python version first
    python_version = sys.version_info
    if python_version < (3, 10):
        print(f"❌ Python {python_version.major}.{python_version.minor} detected")
        print("❌ data_gemma requires Python 3.10 or higher")
        print("📝 CRITICAL: data_gemma is REQUIRED for this application to function")
        print("📝 Please upgrade your Python version:")
        print("   - Install Python 3.10+ from https://python.org")
        print("   - Or use pyenv: pyenv install 3.10.12 && pyenv local 3.10.12")
        print("❌ Application will NOT work without data_gemma")
        return False
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor} is compatible")
    
    try:
        # Install using uv (preferred) or pip
        if os.system("which uv") == 0:  # uv is available
            print("✅ Using uv for installation...")
            result = subprocess.run([
                "uv", "add", "data-gemma@git+https://github.com/datacommonsorg/llm-tools.git"
            ], capture_output=True, text=True)
        else:
            print("✅ Using pip for installation...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/datacommonsorg/llm-tools.git"
            ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ data_gemma installation completed!")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def verify_installation():
    """Verify data_gemma is properly installed"""
    print("\n🔍 Verifying installation...")
    
    try:
        # Test imports
        from data_gemma.rig import RigDataGemma
        from data_gemma.rag import RagDataGemma
        print("✅ data_gemma imports successful")
        
        # Try to initialize (this might fail if models aren't available, which is expected)
        try:
            rig_model = RigDataGemma()
            print("✅ RIG model initialization successful")
        except Exception as e:
            print(f"⚠️ RIG model initialization warning: {e}")
            print("   This is expected if DataGemma models aren't downloaded yet")
        
        try:
            rag_model = RagDataGemma()
            print("✅ RAG model initialization successful")
        except Exception as e:
            print(f"⚠️ RAG model initialization warning: {e}")
            print("   This is expected if DataGemma models aren't downloaded yet")
        
        print("\n🎉 data_gemma verification completed!")
        print("📝 Note: You may see warnings about model downloads - this is normal")
        print("📝 The models will be downloaded automatically when first used")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("   data_gemma may not be properly installed")
        return False
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

def main():
    print("🚀 Data Gemma Installation Script")
    print("=" * 40)
    
    # Install
    if install_data_gemma():
        # Verify
        if verify_installation():
            print("\n✅ All good! data_gemma is ready to use")
            print("\n🔧 Next steps:")
            print("1. Start your backend: python run.py")
            print("2. Test with documents: python run.py --test-doc sample_document.txt --prompt 'Your question'")
            print("3. The agent will now use real Data Commons data via data_gemma!")
        else:
            print("\n❌ CRITICAL: Installation completed but verification failed")
            print("❌ data_gemma is REQUIRED - the agent will NOT start without it")
    else:
        print("\n❌ CRITICAL: data_gemma installation failed")
        print("❌ The application REQUIRES data_gemma to function")
        print("   You can try manual installation:")
        print("   pip install git+https://github.com/datacommonsorg/llm-tools.git")
        print("❌ The agent will NOT START without data_gemma")

if __name__ == "__main__":
    main() 