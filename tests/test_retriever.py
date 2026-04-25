from unittest.mock import MagicMock

from src.rag.retriever import Retriever, RetrievedChunk
from src.rag.vectorstore import SearchResult


class TestRetriever:
    def _build_retriever(self):
        mock_embedder = MagicMock()
        mock_store = MagicMock()
        retriever = Retriever(embedder=mock_embedder, vectorstore=mock_store)
        return retriever, mock_embedder, mock_store

    def _sr(self, text: str, score: float, url: str = "https://example.com/a", idx: int = 0, title: str = "Doc"):
        return SearchResult(
            chunk_text=text,
            url=url,
            title=title,
            chunk_index=idx,
            similarity_score=score,
            metadata={"url": url, "chunk_index": idx},
        )

    def test_empty_query_returns_empty(self):
        retriever, mock_embedder, mock_store = self._build_retriever()

        result = retriever.retrieve("   ")

        assert result == []
        mock_embedder.embed_one.assert_not_called()
        mock_store.search.assert_not_called()

    def test_calls_embedder_and_vectorstore(self):
        retriever, mock_embedder, mock_store = self._build_retriever()
        mock_embedder.embed_one.return_value = [0.1, 0.2, 0.3]
        mock_store.search.return_value = []

        retriever.retrieve("what is rag?")

        mock_embedder.embed_one.assert_called_once_with("what is rag?")
        mock_store.search.assert_called_once()

    def test_passes_top_k_and_threshold(self):
        retriever, mock_embedder, mock_store = self._build_retriever()
        mock_embedder.embed_one.return_value = [0.1, 0.2]
        mock_store.search.return_value = []

        retriever.retrieve("query", top_k=7, similarity_threshold=0.55)

        kwargs = mock_store.search.call_args.kwargs
        assert kwargs["top_k"] == 7
        assert kwargs["similarity_threshold"] == 0.55

    def test_non_enforced_threshold_queries_with_zero_cutoff(self):
        retriever, mock_embedder, mock_store = self._build_retriever()
        mock_embedder.embed_one.return_value = [0.1, 0.2]
        mock_store.search.return_value = []

        retriever.retrieve("query", similarity_threshold=0.7, enforce_threshold=False)

        kwargs = mock_store.search.call_args.kwargs
        assert kwargs["similarity_threshold"] == 0.0

    def test_results_are_sorted_descending(self):
        retriever, mock_embedder, mock_store = self._build_retriever()
        mock_embedder.embed_one.return_value = [0.1]
        mock_store.search.return_value = [
            self._sr("low", 0.41, idx=1),
            self._sr("high", 0.91, idx=2),
            self._sr("mid", 0.66, idx=3),
        ]

        result = retriever.retrieve("query", similarity_threshold=0.0)

        scores = [r.similarity_score for r in result]
        assert scores == [0.91, 0.66, 0.41]

    def test_defensive_threshold_filter(self):
        retriever, mock_embedder, mock_store = self._build_retriever()
        mock_embedder.embed_one.return_value = [0.1]
        mock_store.search.return_value = [
            self._sr("kept", 0.80),
            self._sr("drop", 0.20),
        ]

        result = retriever.retrieve("query", similarity_threshold=0.5)

        assert len(result) == 1
        assert result[0].text == "kept"

    def test_returns_retrievedchunk_objects(self):
        retriever, mock_embedder, mock_store = self._build_retriever()
        mock_embedder.embed_one.return_value = [0.1]
        mock_store.search.return_value = [self._sr("text", 0.7)]

        result = retriever.retrieve("query", similarity_threshold=0.0)

        assert len(result) == 1
        assert isinstance(result[0], RetrievedChunk)

    def test_format_context_empty(self):
        retriever, _, _ = self._build_retriever()
        assert retriever.format_context([]) == ""

    def test_format_context_includes_sources(self):
        retriever, _, _ = self._build_retriever()
        results = [
            RetrievedChunk(
                text="Chunk one",
                url="https://example.com/1",
                title="Page One",
                chunk_index=0,
                similarity_score=0.8,
                metadata={},
            ),
            RetrievedChunk(
                text="Chunk two",
                url="https://example.com/2",
                title="Page Two",
                chunk_index=1,
                similarity_score=0.7,
                metadata={},
            ),
        ]

        context = retriever.format_context(results)

        assert "[1] Page One (https://example.com/1)" in context
        assert "Chunk one" in context
        assert "[2] Page Two (https://example.com/2)" in context
        assert "Chunk two" in context

    def test_build_sources_shape(self):
        retriever, _, _ = self._build_retriever()
        results = [
            RetrievedChunk(
                text="A" * 300,
                url="https://example.com/source",
                title="Source",
                chunk_index=2,
                similarity_score=0.88,
                metadata={},
            )
        ]

        sources = retriever.build_sources(results)

        assert len(sources) == 1
        assert sources[0]["url"] == "https://example.com/source"
        assert sources[0]["title"] == "Source"
        assert sources[0]["chunk_index"] == 2
        assert sources[0]["similarity_score"] == 0.88
        assert len(sources[0]["snippet"]) == 220

    def test_confidence_score_range(self):
        retriever, _, _ = self._build_retriever()
        results = [
            RetrievedChunk(
                text="Chunk one",
                url="https://example.com/1",
                title="Page One",
                chunk_index=0,
                similarity_score=0.9,
                metadata={},
            ),
            RetrievedChunk(
                text="Chunk two",
                url="https://example.com/2",
                title="Page Two",
                chunk_index=1,
                similarity_score=0.7,
                metadata={},
            ),
        ]

        score = retriever.confidence_score(results)

        assert score >= 0.0
        assert score <= 1.0
        assert score > 0.0
