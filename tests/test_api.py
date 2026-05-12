import pytest
from httpx import AsyncClient

from dataclasses import dataclass, field
from typing import Any, Dict, List

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

    async def crawl(self, start_url: str):
        if self.error:
            raise self.error
        return self.result

    async def close(self):
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


@pytest.mark.asyncio
class TestAPI:
    @pytest.fixture(autouse=True)
    async def setup_client(self):
        """Setup an async client for testing the API."""
        self.app = create_app()
        # Use AsyncClient for async app testing
        async with AsyncClient(app=self.app, base_url="http://test") as client:
            self.client = client
            yield

    def _override_dependencies(self, crawler=None, cleaner=None, embedder=None, vectorstore=None, qa_service=None):
        if crawler:
            self.app.dependency_overrides[get_crawler] = lambda: crawler
        if cleaner:
            self.app.dependency_overrides[get_cleaner] = lambda: cleaner
        if embedder:
            self.app.dependency_overrides[get_embedder] = lambda: embedder
        if vectorstore:
            self.app.dependency_overrides[get_vectorstore] = lambda: vectorstore
        if qa_service:
            self.app.dependency_overrides[get_qa_service] = lambda: qa_service

    async def test_health_check(self):
        response = await self.client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    async def test_crawl_success(self):
        crawl_result = make_crawl_result()
        dummy_crawler = DummyCrawler(result=crawl_result)
        dummy_cleaner = DummyCleaner()
        dummy_embedder = DummyEmbedder()
        dummy_vectorstore = DummyVectorStore()

        self._override_dependencies(
            crawler=dummy_crawler,
            cleaner=dummy_cleaner,
            embedder=dummy_embedder,
            vectorstore=dummy_vectorstore,
        )

        response = await self.client.post("/crawl", json={"url": "https://example.com"})

        assert response.status_code == 200
        data = response.json()
        assert data["start_url"] == "https://example.com"
        assert len(data["pages"]) == 2
        assert data["stats"]["pages_crawled"] == 2
        assert data["stats"]["pages_indexed"] == 2
        assert dummy_vectorstore.count() == 2 * 3  # 2 pages, 3 chunks each based on dummy text length

    async def test_crawl_with_existing_docs(self):
        crawl_result = make_crawl_result()
        dummy_crawler = DummyCrawler(result=crawl_result)
        dummy_vectorstore = DummyVectorStore()
        # Simulate that one URL has been indexed before
        dummy_vectorstore.add([{"url": "https://example.com/about"}], [])

        self._override_dependencies(
            crawler=dummy_crawler,
            vectorstore=dummy_vectorstore,
            cleaner=DummyCleaner(),
            embedder=DummyEmbedder(),
        )

        response = await self.client.post("/crawl", json={"url": "https://example.com"})

        assert response.status_code == 200
        # Check that the existing URL was deleted before adding new docs
        assert "https://example.com/about" in dummy_vectorstore.deleted_urls
        assert dummy_vectorstore.count() == 2 * 3

    async def test_crawl_crawler_error(self):
        dummy_crawler = DummyCrawler(error=Exception("Test crawl error"))
        self._override_dependencies(crawler=dummy_crawler)

        response = await self.client.post("/crawl", json={"url": "https://example.com"})

        assert response.status_code == 500
        assert "Test crawl error" in response.json()["detail"]

    async def test_ask_question_success(self):
        dummy_qa = DummyQAService()
        self._override_dependencies(qa_service=dummy_qa)

        response = await self.client.post("/ask", json={"question": "What is RAG?"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == dummy_qa.result.answer
        assert len(data["sources"]) == 1

    async def test_ask_question_refused(self):
        refused_result = QAResult(answer="", sources=[], used_context_chunks=0, refused=True, reason="Not relevant")
        dummy_qa = DummyQAService(result=refused_result)
        self._override_dependencies(qa_service=dummy_qa)

        response = await self.client.post("/ask", json={"question": "Irrelevant question"})

        assert response.status_code == 200
        data = response.json()
        assert data["refused"] is True
        assert data["reason"] == "Not relevant"

    async def test_get_stats(self):
        dummy_vectorstore = DummyVectorStore()
        dummy_vectorstore.add([{"url": "https://a.com"}], [])
        dummy_vectorstore.add([{"url": "https://b.com"}], [])
        self._override_dependencies(vectorstore=dummy_vectorstore)

        response = await self.client.get("/stats")

        assert response.status_code == 200
        assert response.json() == {"doc_count": 2}

    async def test_app_lifecycle(self):
        """
        Tests if the startup and shutdown events work correctly.
        Specifically, it checks if the crawler's close method is called on shutdown.
        """
        dummy_crawler = DummyCrawler()
        self._override_dependencies(crawler=dummy_crawler)

        # The AsyncClient context manager handles the app lifespan
        async with AsyncClient(app=self.app, base_url="http://test") as client:
            await client.get("/health")  # Make a request to ensure the app is running

        # After the client exits, the shutdown event should have fired
        assert dummy_crawler.closed is True
