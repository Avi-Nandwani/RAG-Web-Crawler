import hashlib
from unittest.mock import MagicMock, patch
import pytest

from src.rag.chunker import TextChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(url: str = "https://example.com/page", idx: int = 0, text: str = "Sample text for testing purposes.") -> TextChunk:
    return TextChunk(
        text=text,
        url=url,
        title="Test Page",
        chunk_index=idx,
        total_chunks=3,
        char_start=idx * 100,
        char_end=(idx + 1) * 100,
    )


def fake_embedding(dim: int = 8) -> list:
    """Return a deterministic unit-ish vector."""
    val = 1.0 / dim ** 0.5
    return [val] * dim


# ---------------------------------------------------------------------------
# Embedder tests
# ---------------------------------------------------------------------------

class TestEmbedder:
    """Tests for src.rag.embedder.Embedder — model is fully mocked."""

    def _make_embedder(self, mock_model):
        from src.rag.embedder import Embedder
        emb = Embedder(model_name="all-MiniLM-L6-v2", batch_size=2, device="cpu")
        emb._model = mock_model
        return emb

    def _mock_model(self, dim: int = 8):
        import numpy as np
        m = MagicMock()
        m.get_sentence_embedding_dimension.return_value = dim
        # encode() returns a numpy array of rows
        m.encode.side_effect = lambda texts, **kw: np.array([fake_embedding(dim) for _ in texts])
        return m

    def test_embed_returns_list_of_lists(self):
        model = self._mock_model()
        emb = self._make_embedder(model)
        result = emb.embed(["hello world"])
        assert isinstance(result, list)
        assert isinstance(result[0], list)

    def test_embed_correct_count(self):
        model = self._mock_model()
        emb = self._make_embedder(model)
        texts = ["one", "two", "three"]
        result = emb.embed(texts)
        assert len(result) == 3

    def test_embed_empty_returns_empty(self):
        model = self._mock_model()
        emb = self._make_embedder(model)
        assert emb.embed([]) == []

    def test_embed_batch_logic(self):
        """With batch_size=2 and 5 texts, encode() should be called 3 times."""
        model = self._mock_model()
        emb = self._make_embedder(model)
        emb.batch_size = 2
        emb.embed(["a", "b", "c", "d", "e"])
        assert model.encode.call_count == 3

    def test_embed_one_returns_single_vector(self):
        model = self._mock_model(dim=8)
        emb = self._make_embedder(model)
        result = emb.embed_one("hello")
        assert isinstance(result, list)
        assert len(result) == 8

    def test_dimension_property(self):
        model = self._mock_model(dim=384)
        emb = self._make_embedder(model)
        assert emb.dimension == 384

    def test_lazy_loading(self):
        """Model should NOT be loaded before embed() is called."""
        from src.rag.embedder import Embedder
        emb = Embedder(model_name="all-MiniLM-L6-v2")
        assert emb._model is None  # Not loaded yet


# ---------------------------------------------------------------------------
# VectorStore tests
# ---------------------------------------------------------------------------

class TestVectorStore:
    """Tests for src.rag.vectorstore.VectorStore — uses chromadb EphemeralClient."""

    @pytest.fixture
    def store(self):
        """Return a fresh in-memory VectorStore for each test."""
        import uuid
        import chromadb
        from src.rag.vectorstore import VectorStore

        # Use a unique collection name so that EphemeralClient's shared
        # in-memory store does not bleed data between tests.
        client = chromadb.EphemeralClient()
        cname = f"test_{uuid.uuid4().hex}"
        vs = VectorStore.__new__(VectorStore)
        vs.collection_name = cname
        vs.persist_directory = ":memory:"
        vs._client = client
        vs._collection = client.get_or_create_collection(
            name=cname,
            metadata={"hnsw:space": "cosine"},
        )
        yield vs
        try:
            client.delete_collection(cname)
        except Exception:
            pass

    def _embeddings(self, n: int, dim: int = 8):
        return [fake_embedding(dim) for _ in range(n)]

    def test_add_returns_count(self, store):
        chunks = [make_chunk(idx=i) for i in range(3)]
        embeddings = self._embeddings(3)
        result = store.add(chunks, embeddings)
        assert result == 3

    def test_add_empty_returns_zero(self, store):
        assert store.add([], []) == 0

    def test_add_mismatched_lengths_raises(self, store):
        chunks = [make_chunk()]
        with pytest.raises(ValueError):
            store.add(chunks, [])

    def test_count_after_add(self, store):
        chunks = [make_chunk(idx=i) for i in range(5)]
        store.add(chunks, self._embeddings(5))
        assert store.count() == 5

    def test_search_returns_results(self, store):
        chunk = make_chunk(url="https://example.com/a", idx=0, text="machine learning is great")
        store.add([chunk], self._embeddings(1))
        results = store.search(fake_embedding(), top_k=1, similarity_threshold=0.0)
        assert len(results) == 1
        assert results[0].url == "https://example.com/a"

    def test_search_result_has_correct_fields(self, store):
        chunk = make_chunk(url="https://example.com/test", idx=0)
        store.add([chunk], self._embeddings(1))
        results = store.search(fake_embedding(), top_k=1, similarity_threshold=0.0)
        r = results[0]
        assert r.url == "https://example.com/test"
        assert r.title == "Test Page"
        assert isinstance(r.similarity_score, float)
        assert 0.0 <= r.similarity_score <= 1.0

    def test_search_respects_threshold(self, store):
        """With threshold above the max possible score (1.0), nothing is returned."""
        chunk = make_chunk()
        store.add([chunk], self._embeddings(1))
        results = store.search(fake_embedding(), top_k=5, similarity_threshold=1.1)
        assert results == []

    def test_search_empty_store_returns_empty(self, store):
        results = store.search(fake_embedding(), top_k=5, similarity_threshold=0.0)
        assert results == []

    def test_search_top_k_limits_results(self, store):
        chunks = [make_chunk(url=f"https://example.com/{i}", idx=i) for i in range(10)]
        store.add(chunks, self._embeddings(10))
        results = store.search(fake_embedding(), top_k=3, similarity_threshold=0.0)
        assert len(results) <= 3

    def test_delete_by_url(self, store):
        url_a = "https://example.com/a"
        url_b = "https://example.com/b"
        chunks_a = [make_chunk(url=url_a, idx=i) for i in range(3)]
        chunks_b = [make_chunk(url=url_b, idx=i) for i in range(2)]
        store.add(chunks_a + chunks_b, self._embeddings(5))

        deleted = store.delete_by_url(url_a)
        assert deleted == 3
        assert store.count() == 2

    def test_delete_nonexistent_url_returns_zero(self, store):
        deleted = store.delete_by_url("https://nothere.com")
        assert deleted == 0

    def test_upsert_overwrites_same_chunk(self, store):
        """Adding the same chunk twice should not increase the count."""
        chunk = make_chunk(url="https://example.com/page", idx=0)
        store.add([chunk], self._embeddings(1))
        store.add([chunk], self._embeddings(1))  # Same ID → upsert
        assert store.count() == 1

    def test_clear_wipes_all_data(self, store):
        chunks = [make_chunk(idx=i) for i in range(5)]
        store.add(chunks, self._embeddings(5))
        store.clear()
        assert store.count() == 0

    def test_chunk_index_in_metadata(self, store):
        chunk = make_chunk(url="https://example.com/p", idx=7)
        store.add([chunk], self._embeddings(1))
        results = store.search(fake_embedding(), top_k=1, similarity_threshold=0.0)
        assert results[0].chunk_index == 7
