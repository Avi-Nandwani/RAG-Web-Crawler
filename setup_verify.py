"""
Setup verification script for RAG Web Crawler
Run this to verify your environment is correctly configured
"""

import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.9 or higher"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"✗ Python 3.9+ required, but {version.major}.{version.minor} found")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    required_packages = [
        "fastapi",
        "uvicorn",
        "beautifulsoup4",
        "requests",
        "chromadb",
        "sentence_transformers",
        "ollama",
        "loguru",
        "pydantic",
        "yaml",
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == "yaml":
                __import__("yaml")
            elif package == "beautifulsoup4":
                __import__("bs4")
            elif package == "sentence_transformers":
                __import__("sentence_transformers")
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True


def check_ollama():
    """Check if Ollama is accessible"""
    print("\nChecking Ollama...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama is running")
            models = response.json().get("models", [])
            if models:
                print(f"  Installed models: {', '.join([m['name'] for m in models])}")
                # Check for llama3.2
                if any("llama3.2" in m["name"] for m in models):
                    print("✓ llama3.2 model found")
                else:
                    print("⚠ llama3.2 model not found. Run: ollama pull llama3.2:3b")
            else:
                print("⚠ No models installed. Run: ollama pull llama3.2:3b")
            return True
        else:
            print("✗ Ollama returned unexpected status")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("  Install from: https://ollama.ai")
        return False


def check_directories():
    """Check if required directories exist"""
    print("\nChecking directories...")
    required_dirs = [
        "src/api",
        "src/crawler", 
        "src/rag",
        "src/llm",
        "src/utils",
        "tests",
        "data/raw",
        "data/processed",
        "data/chroma_db",
        "notebooks"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - MISSING")
            all_exist = False
    
    return all_exist


def check_config_files():
    """Check if configuration files exist"""
    print("\nChecking configuration files...")
    config_files = [
        "config.yaml",
        "requirements.txt",
        ".gitignore",
        ".env.example"
    ]
    
    all_exist = True
    for file_path in config_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist


def test_imports():
    """Test if project modules can be imported"""
    print("\nTesting project imports...")
    try:
        from src.utils.config import config
        print("✓ src.utils.config")
        
        from src.utils.helpers import normalize_url
        print("✓ src.utils.helpers")
        
        print(f"\nConfiguration loaded: {config.get('crawler.max_pages')} max pages")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def main():
    """Run all verification checks"""
    print("=" * 60)
    print("RAG Web Crawler - Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Ollama", check_ollama),
        ("Directories", check_directories),
        ("Config Files", check_config_files),
        ("Project Imports", test_imports),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ {name} check failed with error: {e}")
            results[name] = False
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! Your environment is ready.")
        print("\nNext steps:")
        print("1. Start Ollama: ollama serve (if not already running)")
        print("2. Pull model: ollama pull llama3.2:3b")
        print("3. Start API: uvicorn src.api.routes:app --reload")
        return True
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
