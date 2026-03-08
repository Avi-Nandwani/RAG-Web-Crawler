import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Regex matching common sentence-ending punctuation followed by whitespace
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class TextChunk:
    """A single chunk of text with its source metadata."""

    text: str
    url: str
    title: str = ""
    chunk_index: int = 0          # Position within the source page (0-based)
    total_chunks: int = 1         # Total chunks produced from this page
    char_start: int = 0           # Character offset in original cleaned text
    char_end: int = 0


class TextChunker:
    """
    Splits text into overlapping windows of approximately chunk_size characters.

    Boundary preference (in order):
      1. Sentence boundary (. ! ?) within the last 20 % of the window
      2. Word boundary (whitespace) within the last 20 % of the window
      3. Hard cut at exactly chunk_size characters
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_chunk_size: Optional[int] = None,
    ):
        chunking_cfg = config.get("chunking", {})
        self.chunk_size: int = chunk_size or chunking_cfg.get("chunk_size", 1000)
        self.chunk_overlap: int = chunk_overlap or chunking_cfg.get("chunk_overlap", 200)
        self.min_chunk_size: int = min_chunk_size or chunking_cfg.get("min_chunk_size", 100)

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str, url: str, title: str = "") -> List[TextChunk]:
        """
        Split text into overlapping TextChunk objects.

        Args:
            text: Cleaned body text of a page.
            url: Source URL (stored in each chunk for retrieval).
            title: Page title (stored in each chunk).

        Returns:
            List of TextChunk objects. Empty list if text is too short.
        """
        if not text or len(text) < self.min_chunk_size:
            logger.debug(f"Text too short to chunk ({len(text)} chars): {url}")
            return []

        raw_chunks = self._split(text)

        # Filter out chunks that are too short after splitting
        raw_chunks = [c for c in raw_chunks if len(c[0].strip()) >= self.min_chunk_size]

        total = len(raw_chunks)
        chunks: List[TextChunk] = []
        for idx, (chunk_text, char_start, char_end) in enumerate(raw_chunks):
            chunks.append(TextChunk(
                text=chunk_text.strip(),
                url=url,
                title=title,
                chunk_index=idx,
                total_chunks=total,
                char_start=char_start,
                char_end=char_end,
            ))

        logger.debug(f"Chunked '{title or url}' → {total} chunk(s)")
        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split(self, text: str) -> List[tuple]:
        """
        Return list of (chunk_text, char_start, char_end) tuples.
        Uses a sliding window advancing by (chunk_size - chunk_overlap) each step.
        """
        results = []
        step = self.chunk_size - self.chunk_overlap
        pos = 0
        text_len = len(text)

        while pos < text_len:
            end = min(pos + self.chunk_size, text_len)
            chunk_text = text[pos:end]

            # If this isn't the last chunk, try to find a better cut point
            if end < text_len:
                cut = self._find_boundary(text, pos, end)
                chunk_text = text[pos:cut]
                end = cut

            results.append((chunk_text, pos, end))

            # Advance by step; if we could not make progress, force forward
            next_pos = pos + step
            if next_pos <= pos:
                next_pos = pos + 1
            pos = next_pos

        return results

    def _find_boundary(self, text: str, start: int, end: int) -> int:
        """
        Look backwards from `end` into the last 20% of the window for a
        sentence or word boundary. Returns the best cut position.
        """
        search_from = end - max(1, int(self.chunk_size * 0.20))
        search_from = max(start, search_from)

        window = text[search_from:end]

        # Try sentence boundary first
        matches = list(_SENTENCE_END.finditer(window))
        if matches:
            last_match = matches[-1]
            return search_from + last_match.start() + 1  # include punctuation

        # Fall back to word boundary (last space in the window)
        last_space = window.rfind(" ")
        if last_space != -1:
            return search_from + last_space + 1

        # Hard cut
        return end
