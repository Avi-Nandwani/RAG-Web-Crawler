# Getting Started with RAG Web Crawler

This guide will help you set up and run the RAG Web Crawler project for the first time.

## Quick Start (5 minutes)

### Step 1: Install Python Dependencies

```bash
# Make sure you're in the project directory
cd RAG-Web-Crawler

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Install Ollama

1. Download Ollama from [https://ollama.ai](https://ollama.ai)
2. Install it (usually automatic on launch)
3. Pull the required model:

```bash
ollama pull llama3.2:3b
```

### Step 3: Verify Setup

Run the setup verification script:

```bash
python setup_verify.py
```

This will check:
- ✓ Python version (3.9+)
- ✓ All dependencies installed
- ✓ Ollama running and accessible
- ✓ Directory structure
- ✓ Configuration files
- ✓ Project imports working

### Step 4: Configure Environment (Optional)

```bash
# Copy example environment file
cp .env.example .env

# Edit .env if you need custom settings
# (default values work for most cases)
```

### Step 5: Test the Setup

```bash
# Test configuration loading
python -m src.utils.config

# Test logger
python -m src.utils.logger

# Test helpers
python -m src.utils.helpers
```

## What's Next?

Week 1 ✅ Complete! You now have:
- Complete project structure
- All dependencies installed
- Configuration system ready
- Logging system setup
- Utility functions ready

**Next: Week 2-3** - Implement the web crawler
- Create `src/crawler/scraper.py`
- Add robots.txt checking
- Implement domain boundaries
- Add content extraction

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Make sure virtual environment is activated and dependencies installed
```bash
pip install -r requirements.txt
```

### Issue: "Cannot connect to Ollama"
**Solution**: Make sure Ollama is running
```bash
# Check if running
curl http://localhost:11434/api/tags

# If not, Ollama usually auto-starts on system boot
# Or manually start it (varies by OS)
```

### Issue: "Config file not found"
**Solution**: Make sure you're running from project root directory
```bash
# Check current directory
pwd  # Linux/Mac
cd   # Windows

# Should be in RAG-Web-Crawler/
```

### Issue: Import errors like "ImportError: cannot import name 'config'"
**Solution**: Make sure you're running from the project root and Python can find the src module
```bash
# Run from project root
python -m src.utils.config

# Or add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
```

## Configuration Overview

All settings are in [config.yaml](config.yaml):

- **Crawler settings**: max_pages (30), crawl_delay (500ms), max_depth (3)
- **Chunking**: chunk_size (1000 chars), overlap (200 chars)
- **Embeddings**: Model (all-MiniLM-L6-v2), batch_size (32)
- **LLM**: Model (llama3.2:3b), temperature (0.1)
- **API**: Port (8000), host (0.0.0.0)

You can modify these values or override them with environment variables in `.env`

## Useful Commands

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Deactivate virtual environment
deactivate

# Verify setup
python setup_verify.py

# Test individual modules
python -m src.utils.config
python -m src.utils.logger
python -m src.utils.helpers

# Run tests (when available)
pytest tests/ -v

# Format code
black src/ tests/

# Check code quality
flake8 src/ tests/
```

## Project Structure Reminder

```
RAG-Web-Crawler/
├── src/
│   ├── api/              # FastAPI routes (Week 7)
│   ├── crawler/          # Web scraping (Week 2-3)
│   ├── rag/              # Chunking, embeddings, retrieval (Week 4-5)
│   ├── llm/              # LLM integration (Week 6)
│   └── utils/            # Config, logging, helpers ✓
├── tests/                # Tests (Week 8+)
├── data/                 # Data storage
│   ├── raw/              # Scraped HTML
│   ├── processed/        # Cleaned text
│   └── chroma_db/        # Vector database
├── notebooks/            # Jupyter demos (Week 8)
├── config.yaml           # Configuration ✓
├── requirements.txt      # Dependencies ✓
└── README.md             # Documentation ✓
```

## Ready to Code!

You've completed Week 1 setup. The foundation is ready. Now you can start implementing the crawler in Week 2.

Happy coding! 🚀
