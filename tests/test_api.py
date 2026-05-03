from dataclasses import dataclass, field
from typing import Any, Dict, List

from fastapi.testclient import TestClient

from src.api.routes import create_app, get_cleaner, get_crawler, get_embedder, get_qa_service, get_vectorstore
from src.crawler.crawler import CrawlResult
from src.crawler.parser import ParsedPage
from src.llm.qa import QAResult


class DummyCrawler:
    def __init__(self, result: CrawlResult = None, error: Exception = None):
        self.result = result
        self.error = error
        self.max_pages = 30
        self.max_depth = 3
        self.default_delay_s = 0.5
        self.closed = False

    def crawl(self, start_url: str):
        if self.error:
            raise self.error
        return self.result

    def close(self):
        self.closed = True


class DummyCleaner:
    def __init__(self, short_urls=None):
        self.short_urls = short_urls or set()

    def clean(self, text: str):
        return text if text else None


class DummyEmbedder:
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"
        self._model = object()

    def embed(self, texts: List[str]):
        return [[0.1, 0.2, 0.3] for _ in texts]


class DummyVectorStore:
    def __init__(self):
        self.docs = []
        self.deleted_urls = []

    def count(self):
        return len(self.docs)

    def delete_by_url(self, url: str):
        self.deleted_urls.append(url)
        self.docs = [doc for doc in self.docs if doc.url != url]
        return 0

    def add(self, chunks, embeddings):
        self.docs.extend(chunks)
        return len(chunks)


class DummyQAService:
    def __init__(self, result: QAResult = None):
        self.result = result or QAResult(
            answer="RAG uses retrieval plus generation [1].",
            sources=[{
                "url": "https://example.com/rag",
                "title": "RAG",
                "chunk_index": 0,
                "similarity_score": 0.9,
                "snippet": "RAG uses retrieval...",
                "exact_snippet": "RAG uses retrieval...",
                "highlighted_snippet": "RAG uses <<retrieval>>...",
                "relevance_span": {"start": 9, "end": 18},
            }],
            used_context_chunks=1,
            refused=False,
            reason="",
        )

    def ask(self, question: str, top_k=None, similarity_threshold=None):
        return self.result


def make_crawl_result():
    pages = [
        ParsedPage(url="https://example.com", title="Home", text="This is enough text for indexing and testing." * 3, links=[], word_count=20),
        ParsedPage(url="https://example.com/about", title="About", text="About page content for indexing and testing." * 3, links=[], word_count=20),
    ]
    return CrawlResult(start_url="https://example.com", pages=pages, failed_urls=[], skipped_urls=[])


class TestAPI:
    def setup_method(self):
        self.app = create_app()
        self.client = TestClient(self.app)

    def teardown_method(self):
        self.app.dependency_overrides = {}

    def test_health(self):
        store = DummyVectorStore()
        self.app.dependency_overrides[get_vectorstore] = lambda: store

        resp = self.client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["vector_count"] == 0

    def test_stats_endpoint(self):
        # Trigger one request so middleware metrics are non-empty.
        self.client.get("/health")

        resp = self.client.get("/stats")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_requests"] >= 1
        assert "/health" in body["endpoint_counts"]
        assert "average_latency_ms" in body
        assert "crawl_runs" in body
        assert "llm_total_tokens" in body

    def test_crawl_endpoint(self):
        crawler = DummyCrawler(result=make_crawl_result())
        self.app.dependency_overrides[get_crawler] = lambda: crawler

        resp = self.client.post("/crawl", json={"start_url": "https://example.com", "max_pages": 10})

        assert resp.status_code == 200
        body = resp.json()
        assert body["page_count"] == 2
        assert "https://example.com/about" in body["urls"]
        assert crawler.closed is True

    def test_crawl_unhandled_error_returns_500(self):
        crawler = DummyCrawler(error=RuntimeError("boom"))
        self.app.dependency_overrides[get_crawler] = lambda: crawler

        resp = self.client.post("/crawl", json={"start_url": "https://example.com"})

        assert resp.status_code == 500
        assert resp.json()["detail"] == "Internal server error"

    def test_index_requires_prior_crawl(self):
        resp = self.client.post("/index", json={})

        assert resp.status_code == 400
        assert "Call /crawl first" in resp.json()["detail"]

    def test_index_endpoint(self):
        self.app.state.last_crawl_result = make_crawl_result()
        self.app.dependency_overrides[get_cleaner] = lambda: DummyCleaner()
        self.app.dependency_overrides[get_embedder] = lambda: DummyEmbedder()
        store = DummyVectorStore()
        self.app.dependency_overrides[get_vectorstore] = lambda: store

        resp = self.client.post("/index", json={"chunk_size": 120, "chunk_overlap": 20, "min_chunk_size": 20})

        assert resp.status_code == 200
        body = resp.json()
        assert body["indexed_pages"] == 2
        assert body["indexed_chunks"] >= 2
        assert body["vector_count"] >= 2

    def test_index_embedding_model_override(self):
        self.app.state.last_crawl_result = make_crawl_result()
        self.app.dependency_overrides[get_cleaner] = lambda: DummyCleaner()

        embedder = DummyEmbedder()
        self.app.dependency_overrides[get_embedder] = lambda: embedder

        store = DummyVectorStore()
        self.app.dependency_overrides[get_vectorstore] = lambda: store

        resp = self.client.post(
            "/index",
            json={
                "embedding_model": "all-mpnet-base-v2",
                "chunk_size": 120,
                "chunk_overlap": 20,
                "min_chunk_size": 20,
            },
        )

        assert resp.status_code == 200
        assert embedder.model_name == "all-mpnet-base-v2"
        assert embedder._model is None

    def test_index_with_empty_payload_uses_config_defaults(self):
        self.app.state.last_crawl_result = make_crawl_result()
        self.app.dependency_overrides[get_cleaner] = lambda: DummyCleaner()
        self.app.dependency_overrides[get_embedder] = lambda: DummyEmbedder()
        store = DummyVectorStore()
        self.app.dependency_overrides[get_vectorstore] = lambda: store

        resp = self.client.post("/index", json={})

        assert resp.status_code == 200
        body = resp.json()
        assert body["indexed_pages"] == 2
        assert body["indexed_chunks"] >= 1

    def test_index_rejects_invalid_chunking_configuration(self):
        self.app.state.last_crawl_result = make_crawl_result()
        self.app.dependency_overrides[get_cleaner] = lambda: DummyCleaner()
        self.app.dependency_overrides[get_embedder] = lambda: DummyEmbedder()
        store = DummyVectorStore()
        self.app.dependency_overrides[get_vectorstore] = lambda: store

        resp = self.client.post(
            "/index",
            json={
                "chunk_size": 100,
                "chunk_overlap": 120,
                "min_chunk_size": 50,
            },
        )

        assert resp.status_code == 422
        body = resp.json()
        assert "chunk_overlap" in body["detail"]

    def test_ask_endpoint(self):
        qa = DummyQAService()
        self.app.dependency_overrides[get_qa_service] = lambda: qa

        resp = self.client.post("/ask", json={"question": "What is RAG?", "top_k": 3})

        assert resp.status_code == 200
        body = resp.json()
        assert body["refused"] is False
        assert "RAG uses retrieval" in body["answer"]
        assert len(body["sources"]) == 1
        assert "total_ms" in body["timings"]

    def test_ask_refusal_propagates(self):
        qa = DummyQAService(result=QAResult(answer="I do not know.", sources=[], used_context_chunks=0, refused=True, reason="no_context"))
        self.app.dependency_overrides[get_qa_service] = lambda: qa

        resp = self.client.post("/ask", json={"question": "Unknown?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["refused"] is True
        assert body["reason"] == "no_context"

    def test_ask_includes_week11_timings_and_usage(self):
        qa_result = QAResult(
            answer="RAG uses retrieval plus generation [1].",
            sources=[{"url": "https://example.com/rag"}],
            used_context_chunks=1,
            confidence_score=0.88,
            similarity_threshold=0.3,
            retrieval_ms=15.25,
            generation_ms=222.5,
            llm_usage={"prompt_tokens": 10, "completion_tokens": 24, "total_tokens": 34},
            refused=False,
            reason="",
        )
        self.app.dependency_overrides[get_qa_service] = lambda: DummyQAService(result=qa_result)

        resp = self.client.post("/ask", json={"question": "What is RAG?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["timings"]["retrieval_ms"] == 15.25
        assert body["timings"]["generation_ms"] == 222.5
        assert body["timings"]["llm_usage"]["total_tokens"] == 34

    def test_validation_error_shape(self):
        resp = self.client.post("/ask", json={"question": ""})

        assert resp.status_code == 422
        body = resp.json()
        assert body["detail"] == "Validation error"
        assert body["path"] == "/ask"
