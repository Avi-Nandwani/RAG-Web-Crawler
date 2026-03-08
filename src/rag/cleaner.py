import re
import unicodedata
import hashlib
from typing import Optional, Set

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum word count to consider a page worth indexing
_MIN_WORDS = 20


class TextCleaner:
    """
    Stateful cleaner that also tracks seen content fingerprints
    so duplicate pages are silently dropped across a crawl session.
    """

    def __init__(self, min_words: int = _MIN_WORDS):
        self.min_words = min_words
        self._seen_hashes: Set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, text: str) -> Optional[str]:
        """
        Clean and validate a page's text.

        Returns the cleaned string, or None if the page should be skipped
        (too short or duplicate).

        Args:
            text: Raw text extracted from HTMLParser.

        Returns:
            Cleaned text string, or None.
        """
        cleaned = self._normalize(text)

        if not self._is_long_enough(cleaned):
            logger.debug("Page skipped — too short after cleaning")
            return None

        if self._is_duplicate(cleaned):
            logger.debug("Page skipped — duplicate content detected")
            return None

        return cleaned

    def reset(self):
        """Clear seen-fingerprint cache (call between independent crawl jobs)."""
        self._seen_hashes.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalize(self, text: str) -> str:
        """Apply all normalization steps and return cleaned text."""
        # 1. Unicode NFC normalization (handles accented chars consistently)
        text = unicodedata.normalize("NFC", text)

        # 2. Remove control characters (keep newlines and tabs)
        text = re.sub(r"[^\S\n\t ]+", " ", text)          # non-space whitespace → space
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)  # control chars

        # 3. Collapse runs of whitespace on a single line to one space
        text = re.sub(r"[ \t]+", " ", text)

        # 4. Collapse 3+ consecutive blank lines to two
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 5. Strip leading/trailing whitespace per line, then overall
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines).strip()

        return text

    def _is_long_enough(self, text: str) -> bool:
        word_count = len(text.split())
        return word_count >= self.min_words

    def _is_duplicate(self, text: str) -> bool:
        """
        Use a SHA-256 fingerprint of the first 2000 chars to detect
        exact or near-exact duplicates (e.g. printer-friendly page variants).
        """
        sample = text[:2000].strip().lower()
        fingerprint = hashlib.sha256(sample.encode()).hexdigest()
        if fingerprint in self._seen_hashes:
            return True
        self._seen_hashes.add(fingerprint)
        return False
