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


class AskResponse(BaseModel):
    answer: str
    refused: bool
    reason: str
    sources: List[SourceItem]
    timings: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    api: str
    version: str
    vector_count: int
    model: str
