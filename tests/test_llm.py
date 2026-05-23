import pytest
from unittest.mock import MagicMock

from src.llm.client import OllamaClient
from src.llm.prompts import REFUSAL_MESSAGE, build_system_prompt, build_user_prompt
from src.llm.qa import GroundedQAService
from src.rag.retriever import RetrievedChunk


class TestPrompts:
    def test_system_prompt_has_grounding_rules(self):
        prompt = build_system_prompt()
        assert "Use only the provided context" in prompt
        assert "refuse" in prompt.lower()
        assert "[1]" in prompt

    def test_user_prompt_contains_question_and_context(self):
        p = build_user_prompt("What is RAG?", "[1] Doc (https://x.com)\nRAG means...")
        assert "What is RAG?" in p
        assert "Context Chunks" in p
        assert "RAG means" in p


class TestOllamaClient:
    def test_generate_success(self):
        fake_client = MagicMock()
        fake_client.chat.return_value = {
            "message": {"content": "RAG is retrieval-augmented generation [1]."}
        }

        client = OllamaClient(client=fake_client, model_name="llama3.2:3b")
        resp = client.generate("q", "system")

        assert "retrieval-augmented" in resp.text
        assert resp.model == "llama3.2:3b"
        fake_client.chat.assert_called_once()

    def test_generate_raises_runtimeerror_on_failure(self):
        fake_client = MagicMock()
        fake_client.chat.side_effect = Exception("connection failed")

        client = OllamaClient(client=fake_client)

        with pytest.raises(RuntimeError):
            client.generate("q")

    def test_generate_extracts_usage(self):
        fake_client = MagicMock()
        fake_client.chat.return_value = {
            "message": {"content": "Answer [1]"},
            "prompt_eval_count": 12,
            "eval_count": 34,
        }

        client = OllamaClient(client=fake_client)
        resp = client.generate("q")

        assert resp.usage["prompt_tokens"] == 12
        assert resp.usage["completion_tokens"] == 34
        assert resp.usage["total_tokens"] == 46
        assert resp.generation_ms >= 0.0


class TestGroundedQAService:
    def _chunk(self, idx=0, score=0.8):
        return RetrievedChunk(
            text="RAG combines retrieval with generation.",
            url="https://example.com/rag",
            title="RAG Basics",
            chunk_index=idx,
            similarity_score=score,
            metadata={},
        )

    def _service(self):
        mock_retriever = MagicMock()
        mock_retriever.get_effective_threshold.return_value = 0.3
        mock_retriever.format_context.return_value = "Context chunks"
        mock_retriever.build_sources.return_value = [
            {
                "url": "https://example.com/rag",
                "title": "RAG Basics",
                "chunk_index": 0,
                "similarity_score": 0.9,
                "snippet": "RAG combines retrieval with generation.",
            }
        ]
        mock_retriever.confidence_score.return_value = 0.8
        mock_llm = MagicMock()

        service = GroundedQAService(retriever=mock_retriever, llm_client=mock_llm)
        return service, mock_retriever, mock_llm

    def test_empty_question_refuses(self):
        service, mock_retriever, mock_llm = self._service()

        result = service.ask("   ")

        assert result.refused is True
        assert result.reason == "empty_question"
        assert result.answer == REFUSAL_MESSAGE
        mock_retriever.retrieve.assert_not_called()
        mock_llm.generate.assert_not_called()

    def test_no_context_refuses(self):
        service, mock_retriever, mock_llm = self._service()
        mock_retriever.retrieve.return_value = []
        mock_retriever.get_effective_threshold.return_value = 0.3

        result = service.ask("What is RAG?")

        assert result.refused is True
        assert result.reason == "no_context"
        mock_retriever.retrieve.assert_called_once_with(
            query="What is RAG?",
            top_k=None,
            similarity_threshold=0.3,
            enforce_threshold=False,
        )
        mock_llm.generate.assert_not_called()

    def test_below_similarity_threshold_refuses(self):
        service, mock_retriever, mock_llm = self._service()
        mock_retriever.retrieve.return_value = [self._chunk(score=0.1)]
        mock_retriever.get_effective_threshold.return_value = 0.5

        result = service.ask("What is RAG?")

        assert result.refused is True
        assert result.reason == "below_similarity_threshold"

    def test_llm_error_refuses(self):
        service, mock_retriever, mock_llm = self._service()
        mock_retriever.retrieve.return_value = [self._chunk()]
        mock_llm.generate.side_effect = Exception("boom")

        result = service.ask("What is RAG?")

        assert result.refused is True
        assert result.reason == "llm_error"

    def test_successful_answer(self):
        service, mock_retriever, mock_llm = self._service()
        mock_retriever.retrieve.return_value = [self._chunk(idx=0, score=0.9)]
        mock_llm.generate.return_value = MagicMock(
            text="RAG is great [1].",
            generation_ms=5.0,
            usage={"total_tokens": 100},
        )

        result = service.ask("What is RAG?")

        assert result.refused is False
        assert "RAG is great" in result.answer
        assert len(result.sources) == 1
        assert result.sources[0]["url"] == "https://example.com/rag"
        assert result.used_context_chunks == 1
        mock_llm.generate.assert_called_once()

    def test_source_formatting(self):
        service, mock_retriever, mock_llm = self._service()
        mock_retriever.retrieve.return_value = [self._chunk(idx=0, score=0.9)]
        mock_retriever.build_sources.return_value = [
            {
                "url": "https://example.com/rag",
                "title": "RAG Basics",
                "chunk_index": 0,
                "similarity_score": 0.9,
                "snippet": "RAG combines retrieval with generation.",
            }
        ]
        mock_llm.generate.return_value = MagicMock(text="Answer without citations.")

        result = service.ask("What is RAG?")

        assert "Sources:" in result.answer
        assert len(result.sources) == 1
