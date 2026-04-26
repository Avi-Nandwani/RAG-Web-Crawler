from unittest.mock import MagicMock, patch

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

        raised = False
        try:
            client.generate("q")
        except RuntimeError:
            raised = True

        assert raised is True


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
        mock_llm = MagicMock()
        return GroundedQAService(retriever=mock_retriever, llm_client=mock_llm), mock_retriever, mock_llm

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
        assert result.answer == REFUSAL_MESSAGE
        mock_llm.generate.assert_not_called()

    def test_llm_error_refuses_with_sources(self):
        service, mock_retriever, mock_llm = self._service()
        retrieved = [self._chunk()]
        mock_retriever.get_effective_threshold.return_value = 0.3
        mock_retriever.confidence_score.return_value = 0.78
        mock_retriever.retrieve.return_value = retrieved
        mock_retriever.format_context.return_value = "[1] RAG Basics ..."
        mock_retriever.build_sources.return_value = [{"url": "https://example.com/rag"}]
        mock_llm.generate.side_effect = RuntimeError("boom")

        result = service.ask("What is RAG?")

        assert result.refused is True
        assert result.reason == "llm_error"
        assert len(result.sources) == 1

    def test_empty_llm_output_refuses(self):
        service, mock_retriever, mock_llm = self._service()
        retrieved = [self._chunk()]
        mock_retriever.get_effective_threshold.return_value = 0.3
        mock_retriever.confidence_score.return_value = 0.78
        mock_retriever.retrieve.return_value = retrieved
        mock_retriever.format_context.return_value = "context"
        mock_retriever.build_sources.return_value = [{"url": "https://example.com/rag"}]
        mock_llm.generate.return_value = MagicMock(text="")

        result = service.ask("What is RAG?")

        assert result.refused is True
        assert result.reason == "empty_llm_output"

    def test_success_returns_answer_and_sources(self):
        service, mock_retriever, mock_llm = self._service()
        retrieved = [self._chunk()]
        mock_retriever.get_effective_threshold.return_value = 0.3
        mock_retriever.confidence_score.return_value = 0.82
        mock_retriever.retrieve.return_value = retrieved
        mock_retriever.format_context.return_value = "context"
        mock_retriever.build_sources.return_value = [
            {"title": "RAG Basics", "url": "https://example.com/rag", "chunk_index": 0, "similarity_score": 0.8, "snippet": "RAG ..."}
        ]
        mock_llm.generate.return_value = MagicMock(text="RAG is retrieval augmented generation [1].")

        result = service.ask("What is RAG?")

        assert result.refused is False
        assert "RAG is retrieval" in result.answer
        assert len(result.sources) == 1
        assert result.used_context_chunks == 1
        assert result.confidence_score == 0.82
        assert result.similarity_threshold == 0.3

    def test_appends_source_footer_if_model_misses_citations(self):
        service, mock_retriever, mock_llm = self._service()
        retrieved = [self._chunk()]
        mock_retriever.get_effective_threshold.return_value = 0.3
        mock_retriever.confidence_score.return_value = 0.82
        mock_retriever.retrieve.return_value = retrieved
        mock_retriever.format_context.return_value = "context"
        mock_retriever.build_sources.return_value = [
            {"title": "RAG Basics", "url": "https://example.com/rag", "chunk_index": 0, "similarity_score": 0.8, "snippet": "RAG ..."}
        ]
        mock_llm.generate.return_value = MagicMock(text="RAG is retrieval augmented generation.")

        result = service.ask("What is RAG?")

        assert result.refused is False
        assert "Sources:" in result.answer
        assert "[1] RAG Basics (https://example.com/rag)" in result.answer

    def test_below_similarity_threshold_refuses_hard(self):
        service, mock_retriever, mock_llm = self._service()
        retrieved = [self._chunk(score=0.2)]
        mock_retriever.get_effective_threshold.return_value = 0.5
        mock_retriever.confidence_score.return_value = 0.24
        mock_retriever.retrieve.return_value = retrieved
        mock_retriever.build_sources.return_value = [{"url": "https://example.com/rag"}]

        result = service.ask("What is RAG?", similarity_threshold=0.5)

        assert result.refused is True
        assert result.reason == "below_similarity_threshold"
        assert result.answer == REFUSAL_MESSAGE
        assert result.confidence_score == 0.24
        assert result.similarity_threshold == 0.5
        mock_llm.generate.assert_not_called()

    def test_passes_question_for_source_highlighting(self):
        service, mock_retriever, mock_llm = self._service()
        question = "What are the project goals?"
        retrieved = [self._chunk(score=0.82)]

        mock_retriever.get_effective_threshold.return_value = 0.3
        mock_retriever.confidence_score.return_value = 0.8
        mock_retriever.retrieve.return_value = retrieved
        mock_retriever.format_context.return_value = "context"
        mock_retriever.build_sources.return_value = [{"url": "https://example.com/rag"}]
        mock_llm.generate.return_value = MagicMock(text="Answer [1]")

        result = service.ask(question)

        assert result.refused is False
        mock_retriever.build_sources.assert_called_with(retrieved, query=question)
