# RAG Web Crawler

A Retrieval-Augmented Generation (RAG) service that crawls websites, indexes content, and answers questions grounded strictly in the collected pages with source citations.

## Overview

This project implements a complete RAG pipeline that:
- Crawls websites while respecting robots.txt and domain boundaries
- Extracts and chunks textual content
- Generates embeddings and stores them in a vector database
- Answers questions using retrieved context with explicit source URLs
- Refuses to answer when information is not found in crawled content

## Features

- ✅ Polite web crawler with configurable delays and robots.txt compliance
- ✅ Domain-bounded crawling with page limits
- ✅ Text extraction with boilerplate removal
- ✅ Semantic chunking with overlap for context preservation
- ✅ Vector-based similarity search using ChromaDB
- ✅ Grounded question answering with source citations
- ✅ RESTful API with FastAPI
- ✅ Comprehensive logging and error handling
- ✅ Completely free and offline-capable

## Tech Stack

- **API Framework**: FastAPI with Pydantic validation
- **Web Scraping**: BeautifulSoup4 + Requests
- **Vector Database**: ChromaDB (persistent, local)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Ollama (Llama 3.2 3B - free, offline)
- **Logging**: Loguru
- **Testing**: Pytest

## Project Structure

```
RAG-Web-Crawler/
├── src/
│   ├── api/              # FastAPI routes and models
│   ├── crawler/          # Web scraping logic
│   ├── rag/              # RAG pipeline (chunking, embeddings, retrieval)
│   ├── llm/              # LLM integration and prompt management
│   └── utils/            # Logging, config, helpers
├── tests/                # Unit and integration tests
├── data/
│   ├── raw/              # Scraped HTML content
│   ├── processed/        # Cleaned and chunked text
│   └── chroma_db/        # Vector database storage
├── notebooks/            # Jupyter notebooks for demos
├── config.yaml           # Configuration settings
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Ollama (for local LLM inference)

### Installation

1. **Clone the repository**
   ```bash
   cd RAG-Web-Crawler
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and setup Ollama**
   
   Download and install Ollama from [https://ollama.ai](https://ollama.ai)
   
   Then pull the required model:
   ```bash
   ollama pull llama3.2:3b
   ```

5. **Configure environment variables**
   ```bash
   # Copy example env file
   cp .env.example .env
   
   # Edit .env if needed (default values should work)
   ```

6. **Verify installation**
   ```bash
   # Check Ollama is running
   curl http://localhost:11434/api/tags
   
   # Should return list of installed models
   ```

## Usage

### Running the API Server

```bash
# Start the FastAPI server
uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

Interactive API documentation: `http://localhost:8000/docs`

### API Endpoints

#### 1. Crawl a Website

```bash
curl -X POST "http://localhost:8000/crawl" \
  -H "Content-Type: application/json" \
  -d '{
    "start_url": "https://example.com",
    "max_pages": 30,
    "max_depth": 3,
    "crawl_delay_ms": 500
  }'
```

**Response:**
```json
{
  "page_count": 25,
  "skipped_count": 2,
  "urls": ["https://example.com", "https://example.com/about", ...]
}
```

#### 2. Index Crawled Content

```bash
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "all-MiniLM-L6-v2"
  }'
```

**Response:**
```json
{
  "vector_count": 487,
  "errors": []
}
```

#### 3. Ask Questions

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main purpose of this website?",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "answer": "Based on the crawled content, this website...",
  "sources": [
    {
      "url": "https://example.com/about",
      "snippet": "Our mission is to...",
      "similarity_score": 0.85
    }
  ],
  "timings": {
    "retrieval_ms": 45,
    "generation_ms": 1200,
    "total_ms": 1245
  }
}
```

### Using CLI (Coming Soon)

Command-line interface will be added in Week 7.

## Configuration

All configuration is managed through [config.yaml](config.yaml). Key settings:

- **Crawler**: `max_pages`, `crawl_delay_ms`, `max_depth`, `user_agent`
- **Chunking**: `chunk_size` (1000 chars), `chunk_overlap` (200 chars)
- **Embeddings**: `model_name`, `batch_size`, `device`
- **Retrieval**: `top_k` (5), `similarity_threshold` (0.3)
- **LLM**: `model_name`, `temperature`, `max_tokens`

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Formatting

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

## Architecture & Design Decisions

- **FastAPI over Flask**: Auto-generated API docs, built-in validation, async support
- **BeautifulSoup over Scrapy**: Simpler for <50 page crawls, easier debugging
- **ChromaDB over FAISS**: Persistent storage, metadata filtering, zero config
- **all-MiniLM-L6-v2**: Lightweight (80MB), fast on CPU, good quality for cost
- **Ollama + Llama 3.2**: Free, offline, good for dev/demos, no API limits
- **800-1000 char chunks**: Balances context vs precision, preserves sentences
- **200 char overlap**: Prevents information loss at chunk boundaries
- **Grounded refusals**: Maintains factual accuracy, prevents hallucinations
- **Source citations**: Enables verification, builds trust, educational value
- **Sync crawler initially**: Simpler to implement, async in Week 12 optimization

## Tradeoffs

- **CPU-based embeddings**: Slower than GPU but more accessible, no hardware requirements
- **Local LLM**: Slower than cloud APIs but free, offline, no rate limits
- **Page limit (30-50)**: Ensures reasonable runtime, sufficient for demos
- **Simple chunking**: Character-based vs semantic, easier to implement and tune
- **No JavaScript rendering**: Simplifies crawler, covers 80% of content

## Troubleshooting

**Ollama not running**
```bash
# Check if Ollama service is running
curl http://localhost:11434/api/tags

# Start Ollama (varies by OS)
# Usually starts automatically on install
```

**ChromaDB errors**
```bash
# Clear vector database if corrupted
rm -rf data/chroma_db/*
```

**Import errors**
```bash
# Ensure you're in virtual environment
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## License

MIT License

## Author

Avi Nandwani

## Acknowledgments

- FastAPI documentation and community
- sentence-transformers by UKPLab
- ChromaDB team
- Ollama project
