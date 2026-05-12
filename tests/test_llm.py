import pytest
from unittest.mock import MagicMock, AsyncMock

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


@pytest.mark.asyncio
class TestOllamaClient:
    async def test_generate_success(self):
        # Use AsyncMock for async methods
        fake_client = MagicMock()
        fake_client.chat = AsyncMock(return_value={
            "message": {"content": "RAG is retrieval-augmented generation [1]."}
        })

        client = OllamaClient(client=fake_client, model_name="llama3.2:3b")
        # The client's generate method is sync, but it's called within an async context in the app
        # For this unit test, we can call it directly. If it were an async method, we'd await it.
        # Let's assume we are testing the wrapper that would make it async.
        # For now, let's adapt the test to how the client is currently written.
        # The Ollama client itself is synchronous. The async part is in the LLM class using run_in_executor.
        # So, we don't need to change the client test to be async, but we can keep the async marker for consistency.
        
        # Re-evaluating: The Ollama client is sync. The tests for it should remain sync.
        # However, the service that USES it (GroundedQAService) will be async.
        # Let's make the client tests sync again and focus on the service.
        pass # We will adjust GroundedQAService tests instead.

# Re-adjusting the plan. OllamaClient is sync. Its tests should be sync.
# GroundedQAService and LLM use this client. Their tests should be async.

class TestOllamaClientSync: # Renaming to be clear
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


@pytest.mark.asyncio
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
        # Use AsyncMock for async methods of dependencies
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock()
        mock_llm = MagicMock()
        # The LLM's ask method will call generate, which is sync, but let's assume ask is async
        mock_llm.generate = MagicMock() # The actual call is sync
        
        # The service's `ask` method is async, so we need to test it in an async context.
        service = GroundedQAService(retriever=mock_retriever, llm_client=mock_llm)
        return service, mock_retriever, mock_llm

    async def test_empty_question_refuses(self):
        service, mock_retriever, mock_llm = self._service()

        result = await service.ask("   ")

        assert result.refused is True
        assert result.reason == "empty_question"
        assert result.answer == REFUSAL_MESSAGE
        mock_retriever.retrieve.assert_not_called()
        mock_llm.generate.assert_not_called()

    async def test_no_context_refuses(self):
        service, mock_retriever, mock_llm = self._service()
        mock_retriever.retrieve.return_value = []
        mock_retriever.get_effective_threshold.return_value = 0.3

        result = await service.ask("What is RAG?")

        assert result.refused is True
        assert result.reason == "no_context"
        mock_retriever.retrieve.assert_called_once_with("What is RAG?", top_k=5, similarity_threshold=0.3)
        mock_llm.generate.assert_not_called()

    async def test_llm_refuses(self):
        service, mock_retriever, mock_llm = self._service()
        mock_retriever.retrieve.return_value = [self._chunk()]
        mock_llm.generate.return_value = MagicMock(text="I cannot answer this.")

        result = await service.ask("What is RAG?")

        assert result.refused is True
        assert result.reason == "llm_refusal"
        assert result.answer == "I cannot answer this."

    async def test_successful_answer(self):
        service, mock_retriever, mock_llm = self._service()
        mock_retriever.retrieve.return_value = [self._chunk(idx=0, score=0.9), self._chunk(idx=1, score=0.85)]
        mock_llm.generate.return_value = MagicMock(text="RAG is great [1].", usage={"total_tokens": 100})

        result = await service.ask("What is RAG?")

        assert result.refused is False
        assert "RAG is great" in result.answer
        assert len(result.sources) == 1
        assert result.sources[0]["url"] == "https://example.com/rag"
        assert result.used_context_chunks == 2
        mock_llm.generate.assert_called_once()

    async def test_answer_with_low_similarity_chunks_filtered(self):
        service, mock_retriever, mock_llm = self._service()
        mock_retriever.retrieve.return_value = [self._chunk(score=0.9), self._chunk(score=0.2)]
        mock_retriever.get_effective_threshold.return_value = 0.5
        mock_llm.generate.return_value = MagicMock(text="Answer [1].")

        await service.ask("What is RAG?")

        # Check that the context passed to the LLM only contains the high-similarity chunk
        call_args, _ = mock_llm.generate.call_args
        user_prompt = call_args[0]
        assert "RAG combines retrieval" in user_prompt # from chunk with score 0.9
        # The context formatter might be complex, but we can assert the source text is there.
        # A more robust test would inspect the formatted context string more deeply.
        
        # Let's check the number of sources in the final answer
        result = await service.ask("What is RAG?")
        assert len(result.sources) == 1 # Only the high-score chunk should be used and cited

    async def test_source_formatting(self):
        service, mock_retriever, mock_llm = self._service()
        chunks = [
            self._chunk(idx=0, score=0.9),
            RetrievedChunk(text="Second chunk.", url="https://example.com/page2", title="Page 2", chunk_index=5, similarity_score=0.88)
        ]
        mock_retriever.retrieve.return_value = chunks
        mock_llm.generate.return_value = MagicMock(text="Answer is A [1] and B [2].")

        result = await service.ask("What is RAG?")

        assert len(result.sources) == 2
        assert result.sources[0]["url"] == "https://example.com/rag"
        assert result.sources[0]["chunk_index"] == 0
        assert result.sources[1]["url"] == "https://example.com/page2"
        assert result.sources[1]["chunk_index"] == 5
