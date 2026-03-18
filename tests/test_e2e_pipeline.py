from dataclasses import dataclass
from typing import List

from fastapi.testclient import TestClient

from src.api.routes import (
    create_app,
    get_cleaner,
    get_crawler,
    get_embedder,
    get_qa_service,
    get_vectorstore,
)
from src.crawler.crawler import CrawlResult
from src.crawler.parser import ParsedPage


@dataclass
class QAResultStub:
    answer: str
    refused: bool
    reason: str
    sources: list


class DummyCrawler:
    def __init__(self, result: CrawlResult):
        self.result = result
        self.max_pages = 30
        self.max_depth = 3
        self.default_delay_s = 0.5

    def crawl(self, start_url: str):
        return self.result

    def close(self):
        return None


class DummyCleaner:
    def clean(self, text: str):
        return text if text else ""


class DummyEmbedder:
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"
        self._model = object()

    def embed(self, texts: List[str]):
        # Keep deterministic pseudo-embeddings for stable tests.
        return [[float(len(text) % 10), 0.1, 0.2] for text in texts]


class InMemoryVectorStore:
    def __init__(self):
        self.docs = []

    def count(self):
        return len(self.docs)

    def delete_by_url(self, url: str):
        original = len(self.docs)
        self.docs = [doc for doc in self.docs if doc.url != url]
        return original - len(self.docs)

    def add(self, chunks, embeddings):
        self.docs.extend(chunks)
        return len(chunks)


class DummyQAService:
    def __init__(self, vectorstore: InMemoryVectorStore):
        self.vectorstore = vectorstore

    def ask(self, question: str, top_k=None, similarity_threshold=None):
        if not self.vectorstore.docs:
            return QAResultStub(
                answer="I do not know based on the indexed content.",
                refused=True,
                reason="no_context",
                sources=[],
            )

        first = self.vectorstore.docs[0]
        snippet = first.text[:120]
        answer = f"Based on indexed content, here is a relevant finding: {snippet} [1]"
        return QAResultStub(
            answer=answer,
            refused=False,
            reason="",
            sources=[
                {
                    "url": first.url,
                    "title": first.title,
                    "chunk_index": first.chunk_index,
                    "similarity_score": 0.9,
                    "snippet": snippet,
                }
            ],
        )


def make_crawl_result() -> CrawlResult:
    pages = [
        ParsedPage(
            url="https://example.com",
            title="Home",
            text=("Example home page with project details and contact information. " * 8).strip(),
            links=[],
            word_count=80,
        ),
        ParsedPage(
            url="https://example.com/about",
            title="About",
            text=("About page describes goals, scope, and timeline for the project. " * 8).strip(),
            links=[],
            word_count=80,
        ),
    ]
    return CrawlResult(start_url="https://example.com", pages=pages, failed_urls=[], skipped_urls=[])


def test_e2e_pipeline_crawl_index_ask():
    app = create_app()
    store = InMemoryVectorStore()

    app.dependency_overrides[get_crawler] = lambda: DummyCrawler(make_crawl_result())
    app.dependency_overrides[get_cleaner] = lambda: DummyCleaner()
    app.dependency_overrides[get_embedder] = lambda: DummyEmbedder()
    app.dependency_overrides[get_vectorstore] = lambda: store
    app.dependency_overrides[get_qa_service] = lambda: DummyQAService(store)

    client = TestClient(app)

    crawl_resp = client.post(
        "/crawl",
        json={
            "start_url": "https://example.com",
            "max_pages": 10,
            "max_depth": 2,
            "crawl_delay_ms": 10,
        },
    )
    assert crawl_resp.status_code == 200
    assert crawl_resp.json()["page_count"] == 2

    index_resp = client.post(
        "/index",
        json={
            "chunk_size": 120,
            "chunk_overlap": 20,
            "min_chunk_size": 20,
        },
    )
    assert index_resp.status_code == 200
    index_body = index_resp.json()
    assert index_body["indexed_pages"] == 2
    assert index_body["indexed_chunks"] >= 2
    assert index_body["vector_count"] >= 2

    ask_resp = client.post(
        "/ask",
        json={
            "question": "What does the site say about project scope?",
            "top_k": 3,
            "similarity_threshold": 0.1,
        },
    )
    assert ask_resp.status_code == 200
    ask_body = ask_resp.json()
    assert ask_body["refused"] is False
    assert "Based on indexed content" in ask_body["answer"]
    assert len(ask_body["sources"]) == 1
    assert ask_body["sources"][0]["url"].startswith("https://example.com")
