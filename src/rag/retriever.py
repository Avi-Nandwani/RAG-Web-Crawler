from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional

from src.rag.embedder import Embedder
from src.rag.vectorstore import SearchResult, VectorStore
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """Normalized retrieval output used by downstream QA/LLM layers."""

    text: str
    url: str
    title: str
    chunk_index: int
    similarity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class Retriever:
    """
    Week 5 retrieval pipeline.

    Responsibilities:
    - Embed user query
    - Query vector store
    - Normalize and sort results by similarity
    - Apply optional similarity threshold guardrail
    - Build formatted context for answer generation
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vectorstore: Optional[VectorStore] = None,
    ):
        self.embedder = embedder or Embedder()
        self.vectorstore = vectorstore or VectorStore()

        retrieval_cfg = config.get("retrieval", {})
        self.default_top_k: int = retrieval_cfg.get("top_k", 5)
        self.default_threshold: float = retrieval_cfg.get("similarity_threshold", 0.3)


    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        enforce_threshold: bool = True,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a user query.

        Args:
            query: User question or search query.
            top_k: Max number of chunks to return.
            similarity_threshold: Minimum score (0..1).
            enforce_threshold: If True, filter out chunks below threshold.

        Returns:
            List[RetrievedChunk] sorted by descending similarity.
        """
        query = (query or "").strip()
        if not query:
            logger.warning("Empty query passed to Retriever.retrieve()")
            return []

        k = top_k if top_k is not None else self.default_top_k
        threshold = self.get_effective_threshold(similarity_threshold)
        search_threshold = threshold if enforce_threshold else 0.0

        query_embedding = self.embedder.embed_one(query)

        raw_results: List[SearchResult] = self.vectorstore.search(
            query_embedding=query_embedding,
            top_k=k,
            similarity_threshold=search_threshold,
        )

        normalized: List[RetrievedChunk] = []
        for item in raw_results:
            # Defensive filtering in case the underlying store implementation changes
            if enforce_threshold and item.similarity_score < threshold:
                continue
            normalized.append(
                RetrievedChunk(
                    text=item.chunk_text,
                    url=item.url,
                    title=item.title,
                    chunk_index=item.chunk_index,
                    similarity_score=item.similarity_score,
                    metadata=item.metadata,
                )
            )

        normalized.sort(key=lambda r: r.similarity_score, reverse=True)

        logger.debug(
            f"Retriever query='{query[:40]}' k={k} threshold={threshold} enforce={enforce_threshold} -> {len(normalized)} result(s)"
        )
        return normalized

    def get_effective_threshold(self, similarity_threshold: Optional[float] = None) -> float:
        """Resolve effective similarity threshold from request or config default."""
        return similarity_threshold if similarity_threshold is not None else self.default_threshold

    def confidence_score(self, results: List[RetrievedChunk]) -> float:
        """
        Compute retrieval confidence score in [0, 1].

        Uses a blend of:
        - best match strength,
        - top-3 average relevance,
        - retrieval coverage (how many chunks were retrieved vs default k).
        """
        if not results:
            return 0.0

        sorted_results = sorted(results, key=lambda r: r.similarity_score, reverse=True)
        top_score = sorted_results[0].similarity_score

        top_n = min(3, len(sorted_results))
        top_avg = sum(item.similarity_score for item in sorted_results[:top_n]) / top_n

        denom = max(1, self.default_top_k)
        coverage = min(len(sorted_results) / denom, 1.0)

        score = (0.6 * top_score) + (0.3 * top_avg) + (0.1 * coverage)
        return round(max(0.0, min(1.0, score)), 4)

    def format_context(self, results: List[RetrievedChunk]) -> str:
        """
        Build a citation-friendly context block for LLM prompts.
        """
        if not results:
            return ""

        lines: List[str] = []
        for idx, result in enumerate(results, start=1):
            title = result.title or "Untitled"
            lines.append(f"[{idx}] {title} ({result.url})")
            lines.append(result.text)
            lines.append("")

        return "\n".join(lines).strip()

    def build_sources(self, results: List[RetrievedChunk], query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Convert retrieved chunks into API-ready source objects.
        """
        sources: List[Dict[str, Any]] = []
        for item in results:
            exact_snippet, highlighted_snippet, relevance_span = self._extract_relevant_snippet(
                text=item.text,
                query=query,
            )
            sources.append(
                {
                    "url": item.url,
                    "title": item.title,
                    "chunk_index": item.chunk_index,
                    "similarity_score": item.similarity_score,
                    "snippet": exact_snippet,
                    "exact_snippet": exact_snippet,
                    "highlighted_snippet": highlighted_snippet,
                    "relevance_span": relevance_span,
                }
            )
        return sources

    def _extract_relevant_snippet(self, text: str, query: Optional[str], max_len: int = 220):
        """
        Return an exact snippet from text centered on the best query-term match,
        plus a highlighted variant and [start, end) offsets within the snippet.
        """
        if not text:
            return "", "", None

        match_start = None
        match_term = ""
        lower_text = text.lower()

        for term in self._query_terms(query):
            idx = lower_text.find(term)
            if idx != -1 and (match_start is None or idx < match_start):
                match_start = idx
                match_term = term

        if match_start is None:
            snippet = text[:max_len]
            return snippet, snippet, None

        half_window = max_len // 2
        start = max(0, match_start - half_window)
        end = min(len(text), start + max_len)

        if end - start < max_len and start > 0:
            start = max(0, end - max_len)

        snippet = text[start:end]
        local_start = max(0, match_start - start)
        local_end = min(len(snippet), local_start + len(match_term))

        highlighted = (
            snippet[:local_start]
            + "<<"
            + snippet[local_start:local_end]
            + ">>"
            + snippet[local_end:]
        )

        return snippet, highlighted, {"start": local_start, "end": local_end}

    def _query_terms(self, query: Optional[str]) -> List[str]:
        if not query:
            return []
        terms = re.findall(r"[a-zA-Z0-9]{3,}", query.lower())
        return sorted(set(terms), key=len, reverse=True)
