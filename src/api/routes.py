import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.models import (
    AskRequest,
    AskResponse,
    CrawlRequest,
    CrawlResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    StatsResponse,
)
from src.crawler.crawler import CrawlResult, WebCrawler
from src.llm.qa import GroundedQAService
from src.rag.chunker import TextChunker
from src.rag.cleaner import TextCleaner
from src.rag.embedder import Embedder
from src.rag.vectorstore import VectorStore
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_crawler() -> WebCrawler:
    return WebCrawler()


def get_cleaner() -> TextCleaner:
    return TextCleaner()


def get_embedder() -> Embedder:
    return Embedder()


def get_vectorstore() -> VectorStore:
    return VectorStore()


def get_qa_service() -> GroundedQAService:
    return GroundedQAService()


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Web Crawler API",
        version="0.1.0",
        description="Week 7 API layer for crawl, index, ask, and health endpoints.",
    )
    app.state.last_crawl_result = None
    app.state.metrics = {
        "total_requests": 0,
        "endpoint_counts": {},
        "total_latency_ms": 0.0,
        "crawl_runs": 0,
        "last_crawl_pages": 0,
        "last_crawl_failed": 0,
        "last_crawl_skipped": 0,
        "embedding_runs": 0,
        "last_embedding_ms": 0.0,
        "total_embedding_ms": 0.0,
        "llm_calls": 0,
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "llm_total_tokens": 0,
    }

    @app.middleware("http")
    async def unhandled_error_middleware(request: Request, call_next):
        started = time.perf_counter()
        try:
            response = await call_next(request)

            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            metrics = request.app.state.metrics
            metrics["total_requests"] += 1
            metrics["total_latency_ms"] += elapsed_ms
            path = request.url.path
            endpoint_counts = metrics["endpoint_counts"]
            endpoint_counts[path] = endpoint_counts.get(path, 0) + 1

            logger.info(
                f"request {request.method} {path} status={response.status_code} duration_ms={elapsed_ms}"
            )
            return response
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            metrics = request.app.state.metrics
            metrics["total_requests"] += 1
            metrics["total_latency_ms"] += elapsed_ms
            path = request.url.path
            endpoint_counts = metrics["endpoint_counts"]
            endpoint_counts[path] = endpoint_counts.get(path, 0) + 1
            logger.exception(f"Unhandled API error on {request.url.path}: {exc}")
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "error": str(exc),
                    "path": request.url.path,
                },
            )

    @app.get("/stats", response_model=StatsResponse)
    def stats(request: Request):
        metrics = request.app.state.metrics
        total_requests = metrics.get("total_requests", 0)
        avg_latency = (
            round(metrics.get("total_latency_ms", 0.0) / total_requests, 2)
            if total_requests > 0
            else 0.0
        )

        return StatsResponse(
            total_requests=total_requests,
            endpoint_counts=metrics.get("endpoint_counts", {}),
            average_latency_ms=avg_latency,
            crawl_runs=metrics.get("crawl_runs", 0),
            last_crawl_pages=metrics.get("last_crawl_pages", 0),
            last_crawl_failed=metrics.get("last_crawl_failed", 0),
            last_crawl_skipped=metrics.get("last_crawl_skipped", 0),
            embedding_runs=metrics.get("embedding_runs", 0),
            last_embedding_ms=metrics.get("last_embedding_ms", 0.0),
            total_embedding_ms=metrics.get("total_embedding_ms", 0.0),
            llm_calls=metrics.get("llm_calls", 0),
            llm_prompt_tokens=metrics.get("llm_prompt_tokens", 0),
            llm_completion_tokens=metrics.get("llm_completion_tokens", 0),
            llm_total_tokens=metrics.get("llm_total_tokens", 0),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Validation error",
                "errors": exc.errors(),
                "path": request.url.path,
            },
        )

    @app.get("/health", response_model=HealthResponse)
    def health(vectorstore: VectorStore = Depends(get_vectorstore)):
        return HealthResponse(
            status="ok",
            api="up",
            version="0.1.0",
            vector_count=vectorstore.count(),
            model=config.get("llm.model_name", "llama3.2:3b"),
        )

    @app.post("/crawl", response_model=CrawlResponse)
    def crawl_site(payload: CrawlRequest, request: Request, crawler: WebCrawler = Depends(get_crawler)):
        crawler.max_pages = payload.max_pages if payload.max_pages is not None else crawler.max_pages
        crawler.max_depth = payload.max_depth if payload.max_depth is not None else crawler.max_depth
        if payload.crawl_delay_ms is not None:
            crawler.default_delay_s = payload.crawl_delay_ms / 1000.0

        try:
            result = crawler.crawl(str(payload.start_url))
            request.app.state.last_crawl_result = result
            metrics = request.app.state.metrics
            metrics["crawl_runs"] += 1
            metrics["last_crawl_pages"] = result.total_pages
            metrics["last_crawl_failed"] = len(result.failed_urls)
            metrics["last_crawl_skipped"] = len(result.skipped_urls)
            return CrawlResponse(
                page_count=result.total_pages,
                skipped_count=len(result.skipped_urls),
                failed_count=len(result.failed_urls),
                total_words=result.total_words,
                urls=[page.url for page in result.pages],
            )
        finally:
            crawler.close()

    @app.post("/index", response_model=IndexResponse)
    def index_content(
        payload: IndexRequest,
        request: Request,
        cleaner: TextCleaner = Depends(get_cleaner),
        embedder: Embedder = Depends(get_embedder),
        vectorstore: VectorStore = Depends(get_vectorstore),
    ):
        crawl_result: Optional[CrawlResult] = request.app.state.last_crawl_result
        if crawl_result is None or not crawl_result.pages:
            return JSONResponse(
                status_code=400,
                content={"detail": "No crawled pages available. Call /crawl first."},
            )

        chunking_cfg = config.get("chunking", {})
        effective_chunk_size = payload.chunk_size if payload.chunk_size is not None else chunking_cfg.get("chunk_size", 1000)
        effective_chunk_overlap = (
            payload.chunk_overlap if payload.chunk_overlap is not None else chunking_cfg.get("chunk_overlap", 200)
        )
        effective_min_chunk_size = (
            payload.min_chunk_size if payload.min_chunk_size is not None else chunking_cfg.get("min_chunk_size", 100)
        )

        if effective_chunk_overlap >= effective_chunk_size:
            return JSONResponse(
                status_code=422,
                content={
                    "detail": "Invalid chunking configuration: chunk_overlap must be smaller than chunk_size.",
                    "values": {
                        "chunk_size": effective_chunk_size,
                        "chunk_overlap": effective_chunk_overlap,
                        "min_chunk_size": effective_min_chunk_size,
                    },
                },
            )

        if effective_min_chunk_size > effective_chunk_size:
            return JSONResponse(
                status_code=422,
                content={
                    "detail": "Invalid chunking configuration: min_chunk_size must be less than or equal to chunk_size.",
                    "values": {
                        "chunk_size": effective_chunk_size,
                        "chunk_overlap": effective_chunk_overlap,
                        "min_chunk_size": effective_min_chunk_size,
                    },
                },
            )

        chunker = TextChunker(
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_chunk_overlap,
            min_chunk_size=effective_min_chunk_size,
        )

        # Optional per-request embedding model override.
        # If the embedder is already instantiated, reset lazy-loaded model
        # so the new model name is applied on the next embed() call.
        if payload.embedding_model:
            embedder.model_name = payload.embedding_model
            if hasattr(embedder, "_model"):
                embedder._model = None

        processed_dir = Path(config.get("paths.processed_data", "./data/processed"))
        processed_dir.mkdir(parents=True, exist_ok=True)

        all_chunks = []
        indexed_pages = 0
        skipped_pages = 0
        errors = []

        for page in crawl_result.pages:
            try:
                cleaned = cleaner.clean(page.text)
                if not cleaned:
                    skipped_pages += 1
                    continue

                page_chunks = chunker.chunk(cleaned, url=page.url, title=page.title)
                if not page_chunks:
                    skipped_pages += 1
                    continue

                vectorstore.delete_by_url(page.url)
                all_chunks.extend(page_chunks)
                indexed_pages += 1
                _save_processed_page(processed_dir, page.url, page.title, cleaned, len(page_chunks))
            except Exception as exc:
                errors.append(f"{page.url}: {exc}")
                logger.error(f"Indexing failed for {page.url}: {exc}")

        if not all_chunks:
            return IndexResponse(
                vector_count=vectorstore.count(),
                indexed_chunks=0,
                indexed_pages=indexed_pages,
                skipped_pages=skipped_pages,
                errors=errors,
            )

        texts = [chunk.text for chunk in all_chunks]
        embed_started = time.perf_counter()
        embeddings = embedder.embed(texts)
        embedding_ms = round((time.perf_counter() - embed_started) * 1000, 2)
        metrics = request.app.state.metrics
        metrics["embedding_runs"] += 1
        metrics["last_embedding_ms"] = embedding_ms
        metrics["total_embedding_ms"] += embedding_ms
        logger.info(
            f"embedding_generation chunks={len(texts)} model={embedder.model_name} duration_ms={embedding_ms}"
        )
        added = vectorstore.add(all_chunks, embeddings)

        return IndexResponse(
            vector_count=vectorstore.count(),
            indexed_chunks=added,
            indexed_pages=indexed_pages,
            skipped_pages=skipped_pages,
            errors=errors,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask_question(payload: AskRequest, qa_service: GroundedQAService = Depends(get_qa_service)):
        started = time.perf_counter()
        result = qa_service.ask(
            question=payload.question,
            top_k=payload.top_k,
            similarity_threshold=payload.similarity_threshold,
        )
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        usage = getattr(result, "llm_usage", {}) or {}

        metrics = app.state.metrics
        if usage:
            metrics["llm_calls"] += 1
            metrics["llm_prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            metrics["llm_completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            metrics["llm_total_tokens"] += int(usage.get("total_tokens", 0) or 0)

        return AskResponse(
            answer=result.answer,
            refused=result.refused,
            reason=result.reason,
            confidence_score=getattr(result, "confidence_score", 0.0),
            similarity_threshold=getattr(result, "similarity_threshold", 0.0),
            sources=result.sources,
            timings={
                "retrieval_ms": getattr(result, "retrieval_ms", None),
                "generation_ms": getattr(result, "generation_ms", None),
                "llm_usage": usage,
                "total_ms": total_ms,
            },
        )

    return app


app = create_app()


def _save_processed_page(processed_dir: Path, url: str, title: str, cleaned_text: str, chunk_count: int):
    url_hash = hashlib.sha1(url.encode()).hexdigest()[:16]
    output_path = processed_dir / f"{url_hash}.json"
    payload = {
        "url": url,
        "title": title,
        "cleaned_text": cleaned_text,
        "chunk_count": chunk_count,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
