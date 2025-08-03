#!/usr/bin/env python3
"""
Simple Python version checker for data_gemma compatibility
"""

import sys

def check_python_version():
    """Check if Python version meets requirements"""
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

if __name__ == "__main__":
    check_python_version() 