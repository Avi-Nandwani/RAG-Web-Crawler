from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.llm.client import LLMResponse, OllamaClient
from src.llm.prompts import REFUSAL_MESSAGE, build_system_prompt, build_user_prompt
from src.rag.retriever import RetrievedChunk, Retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QAResult:
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    used_context_chunks: int = 0
    refused: bool = False
    reason: str = ""


class GroundedQAService:
    """
    Week 6 grounded QA orchestration.

    Flow:
      1. Retrieve relevant chunks
      2. Build grounded prompt
      3. Ask LLM
      4. Return answer + sources

    If retrieval has no results or the LLM fails, service returns a refusal.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm_client: Optional[OllamaClient] = None,
    ):
        self.retriever = retriever or Retriever()
        self.llm_client = llm_client or OllamaClient()

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> QAResult:
        question = (question or "").strip()
        if not question:
            return QAResult(
                answer=REFUSAL_MESSAGE,
                sources=[],
                used_context_chunks=0,
                refused=True,
                reason="empty_question",
            )

        retrieved = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        if not retrieved:
            return QAResult(
                answer=REFUSAL_MESSAGE,
                sources=[],
                used_context_chunks=0,
                refused=True,
                reason="no_context",
            )

        context = self.retriever.format_context(retrieved)
        user_prompt = build_user_prompt(question, context)
        system_prompt = build_system_prompt()

        try:
            llm_response: LLMResponse = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
        except Exception as exc:
            logger.error(f"LLM call failed: {exc}")
            return QAResult(
                answer=REFUSAL_MESSAGE,
                sources=self.retriever.build_sources(retrieved),
                used_context_chunks=len(retrieved),
                refused=True,
                reason="llm_error",
            )

        if not llm_response.text:
            return QAResult(
                answer=REFUSAL_MESSAGE,
                sources=self.retriever.build_sources(retrieved),
                used_context_chunks=len(retrieved),
                refused=True,
                reason="empty_llm_output",
            )

        sources = self.retriever.build_sources(retrieved)
        answer = self._ensure_citations(llm_response.text, sources)

        return QAResult(
            answer=answer,
            sources=sources,
            used_context_chunks=len(retrieved),
            refused=False,
            reason="",
        )

    def _ensure_citations(self, answer: str, sources: List[Dict[str, Any]]) -> str:
        """
        Ensure answer includes source references.
        If the model forgot inline citations, append a compact source footer.
        """
        if "[" in answer and "]" in answer:
            return answer

        if not sources:
            return answer

        footer_lines = ["", "Sources:"]
        for idx, src in enumerate(sources[:3], start=1):
            title = src.get("title") or "Untitled"
            url = src.get("url", "")
            footer_lines.append(f"[{idx}] {title} ({url})")

        return answer.rstrip() + "\n" + "\n".join(footer_lines)
