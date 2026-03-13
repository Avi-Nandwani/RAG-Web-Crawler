from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import chromadb
from chromadb.config import Settings

from src.rag.chunker import TextChunk
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """A single result returned by VectorStore.search()."""

    chunk_text: str
    url: str
    title: str
    chunk_index: int
    similarity_score: float          # 0–1, higher = more similar (cosine)
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """
    Persistent ChromaDB vector store.

    Each document stored is one TextChunk. The ChromaDB document ID is
    deterministic: ``{url_hash}_{chunk_index}`` so re-indexing the same
    URL automatically overwrites old chunks.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: str | None = None,
    ):
        vs_cfg = config.get("vectorstore", {})
        self.collection_name: str = collection_name or vs_cfg.get("collection_name", "web_documents")
        self.persist_directory: str = persist_directory or vs_cfg.get("persist_directory", "./data/chroma_db")

        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"VectorStore ready — collection='{self.collection_name}' "
            f"docs={self._collection.count()} persist='{self.persist_directory}'"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, chunks: List[TextChunk], embeddings: List[List[float]]) -> int:
        """
        Add a list of chunks with pre-computed embeddings.

        Args:
            chunks: TextChunk objects to store.
            embeddings: Corresponding embedding vectors (same length as chunks).

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must be same length")

        ids = [self._make_id(c) for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [self._make_metadata(c) for c in chunks]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.debug(f"Upserted {len(chunks)} chunks — collection total={self._collection.count()}")
        return len(chunks)

    def search(
        self,
        query_embedding: List[float],
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> List[SearchResult]:
        """
        Find the most similar chunks to a query embedding.

        Args:
            query_embedding: Dense vector for the query.
            top_k: Maximum number of results to return (default from config).
            similarity_threshold: Minimum similarity score 0–1 (default from config).

        Returns:
            List of SearchResult ordered by descending similarity.
        """
        ret_cfg = config.get("retrieval", {})
        k = top_k if top_k is not None else ret_cfg.get("top_k", 5)
        threshold = similarity_threshold if similarity_threshold is not None else ret_cfg.get("similarity_threshold", 0.3)

        total = self._collection.count()
        if total == 0:
            logger.warning("VectorStore is empty — nothing to search")
            return []

        # Clamp k to the number of available docs
        k = min(k, total)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        search_results: List[SearchResult] = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score 0–1
            score = 1.0 - (dist / 2.0)
            if score < threshold:
                continue
            search_results.append(SearchResult(
                chunk_text=doc,
                url=meta.get("url", ""),
                title=meta.get("title", ""),
                chunk_index=meta.get("chunk_index", 0),
                similarity_score=round(score, 4),
                metadata=meta,
            ))

        logger.debug(f"Search returned {len(search_results)} results above threshold={threshold}")
        return search_results

    def delete_by_url(self, url: str) -> int:
        """
        Remove all chunks belonging to a given source URL.
        Useful when re-crawling a page to keep the index fresh.

        Args:
            url: Source URL whose chunks should be deleted.

        Returns:
            Number of chunks deleted.
        """
        results = self._collection.get(where={"url": url})
        ids = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
            logger.debug(f"Deleted {len(ids)} chunks for URL: {url}")
        return len(ids)

    def count(self) -> int:
        """Return the total number of stored chunks."""
        return self._collection.count()

    def clear(self):
        """Delete and recreate the collection (wipes all data)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection '{self.collection_name}' cleared")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_id(self, chunk: TextChunk) -> str:
        """
        Deterministic ID = first 16 chars of URL sha1 + chunk_index.
        Guarantees re-indexing the same page overwrites old chunks.
        """
        import hashlib
        url_hash = hashlib.sha1(chunk.url.encode()).hexdigest()[:16]
        return f"{url_hash}_{chunk.chunk_index}"

    def _make_metadata(self, chunk: TextChunk) -> Dict[str, Any]:
        return {
            "url": chunk.url,
            "title": chunk.title,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
        }
