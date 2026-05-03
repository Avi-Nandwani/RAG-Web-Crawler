from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl


class CrawlRequest(BaseModel):
    start_url: HttpUrl
    max_pages: Optional[int] = Field(default=None, ge=1, le=500)
    max_depth: Optional[int] = Field(default=None, ge=0, le=10)
    crawl_delay_ms: Optional[int] = Field(default=None, ge=0, le=10000)


class CrawlResponse(BaseModel):
    page_count: int
    skipped_count: int
    failed_count: int
    total_words: int
    urls: List[str]


class IndexRequest(BaseModel):
    chunk_size: Optional[int] = Field(default=None, ge=50, le=5000)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=2000)
    min_chunk_size: Optional[int] = Field(default=None, ge=1, le=2000)
    embedding_model: Optional[str] = None


class IndexResponse(BaseModel):
    vector_count: int
    indexed_chunks: int
    indexed_pages: int
    skipped_pages: int
    errors: List[str]


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SourceItem(BaseModel):
    url: str
    title: Optional[str] = ""
    chunk_index: Optional[int] = 0
    similarity_score: Optional[float] = 0.0
    snippet: Optional[str] = ""
    exact_snippet: Optional[str] = ""
    highlighted_snippet: Optional[str] = ""
    relevance_span: Optional[Dict[str, int]] = None


class AskResponse(BaseModel):
    answer: str
    refused: bool
    reason: str
    confidence_score: float
    similarity_threshold: float
    sources: List[SourceItem]
    timings: Dict[str, Any]


class StatsResponse(BaseModel):
    total_requests: int
    endpoint_counts: Dict[str, int]
    average_latency_ms: float
    crawl_runs: int
    last_crawl_pages: int
    last_crawl_failed: int
    last_crawl_skipped: int
    embedding_runs: int
    last_embedding_ms: float
    total_embedding_ms: float
    llm_calls: int
    llm_prompt_tokens: int
    llm_completion_tokens: int
    llm_total_tokens: int


class HealthResponse(BaseModel):
    status: str
    api: str
    version: str
    vector_count: int
    model: str
