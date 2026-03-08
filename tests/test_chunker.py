import pytest
from src.rag.cleaner import TextCleaner
from src.rag.chunker import TextChunker, TextChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_text(words: int, word: str = "hello") -> str:
    """Return a string of `words` repeated words."""
    return (word + " ") * words


# ---------------------------------------------------------------------------
# TextCleaner
# ---------------------------------------------------------------------------

class TestTextCleaner:

    def setup_method(self):
        self.cleaner = TextCleaner(min_words=20)

    def test_returns_string_for_valid_text(self):
        text = make_text(30)
        result = self.cleaner.clean(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_collapses_multiple_spaces(self):
        text = "word1   word2    word3 " * 25
        result = self.cleaner.clean(text)
        assert "  " not in result  # No double spaces

    def test_collapses_excessive_blank_lines(self):
        text = "sentence one.\n\n\n\n\nsentence two.\n" * 10
        result = self.cleaner.clean(text)
        assert "\n\n\n" not in result  # Max two consecutive newlines

    def test_removes_control_characters(self):
        text = ("good text \x00\x01\x07 more good text " * 25)
        result = self.cleaner.clean(text)
        assert "\x00" not in result
        assert "\x01" not in result

    def test_returns_none_when_too_short(self):
        text = "only five words here"  # 4 words < min_words=20
        result = self.cleaner.clean(text)
        assert result is None

    def test_returns_none_for_exact_duplicate(self):
        text = make_text(30)
        self.cleaner.clean(text)          # First time — accepted
        result = self.cleaner.clean(text) # Second time — duplicate
        assert result is None

    def test_accepts_different_content(self):
        text_a = make_text(30, "alpha")
        text_b = make_text(30, "beta")
        assert self.cleaner.clean(text_a) is not None
        assert self.cleaner.clean(text_b) is not None

    def test_reset_clears_seen_hashes(self):
        text = make_text(30)
        self.cleaner.clean(text)
        self.cleaner.reset()
        result = self.cleaner.clean(text)  # Should be accepted again after reset
        assert result is not None

    def test_strips_leading_trailing_whitespace(self):
        text = "   " + make_text(25) + "   "
        result = self.cleaner.clean(text)
        assert result == result.strip()

    def test_unicode_normalization(self):
        # Café with combining accent vs precomposed — should both work
        text = ("caf\u0065\u0301 is a nice place to visit and work " * 5)
        result = self.cleaner.clean(text)
        assert result is not None


# ---------------------------------------------------------------------------
# TextChunker
# ---------------------------------------------------------------------------

class TestTextChunker:

    URL = "https://example.com/page"
    TITLE = "Test Page"

    def _chunker(self, size=200, overlap=40, min_size=50):
        return TextChunker(chunk_size=size, chunk_overlap=overlap, min_chunk_size=min_size)

    def test_raises_if_overlap_exceeds_size(self):
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_empty_text_returns_empty_list(self):
        chunker = self._chunker()
        assert chunker.chunk("", self.URL) == []

    def test_short_text_returns_empty_list(self):
        chunker = self._chunker(min_size=200)
        assert chunker.chunk("too short", self.URL) == []

    def test_short_text_returns_single_chunk(self):
        chunker = self._chunker(size=500, overlap=50, min_size=10)
        text = make_text(30)  # ~150 chars < chunk_size
        chunks = chunker.chunk(text, self.URL, self.TITLE)
        assert len(chunks) == 1

    def test_long_text_produces_multiple_chunks(self):
        chunker = self._chunker(size=200, overlap=40)
        text = make_text(300)  # ~1500 chars >> chunk_size
        chunks = chunker.chunk(text, self.URL)
        assert len(chunks) > 1

    def test_chunks_have_correct_metadata(self):
        chunker = self._chunker(size=200, overlap=40, min_size=10)
        text = make_text(300)
        chunks = chunker.chunk(text, self.URL, self.TITLE)
        for i, chunk in enumerate(chunks):
            assert chunk.url == self.URL
            assert chunk.title == self.TITLE
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)

    def test_chunk_index_sequential(self):
        chunker = self._chunker(size=200, overlap=40)
        text = make_text(300)
        chunks = chunker.chunk(text, self.URL)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_overlap_means_shared_content(self):
        """Consecutive chunks should share some text (due to overlap)."""
        chunker = self._chunker(size=200, overlap=80, min_size=10)
        text = "word" + " word" * 300  # predictable content
        chunks = chunker.chunk(text, self.URL)
        if len(chunks) >= 2:
            end_of_first = chunks[0].text[-30:]
            start_of_second = chunks[1].text[:30]
            # At least some shared characters after trimming
            assert any(w in chunks[1].text for w in chunks[0].text.split()[-5:])

    def test_no_chunk_exceeds_chunk_size(self):
        chunker = self._chunker(size=200, overlap=40)
        text = make_text(300)
        chunks = chunker.chunk(text, self.URL)
        for chunk in chunks:
            assert len(chunk.text) <= 200

    def test_all_text_covered(self):
        """
        Every word in the original text should appear in at least one chunk.
        (Overlap guarantees full coverage.)
        """
        chunker = self._chunker(size=200, overlap=40, min_size=10)
        words = [f"word{i}" for i in range(60)]
        text = " ".join(words)
        chunks = chunker.chunk(text, self.URL)
        combined = " ".join(c.text for c in chunks)
        for word in words:
            assert word in combined

    def test_min_chunk_size_filters_tiny_tail(self):
        """A trailing fragment smaller than min_chunk_size should be dropped."""
        # 210 chars = one full chunk of 200 + 10-char tail (< min_size=50)
        chunker = self._chunker(size=200, overlap=0, min_size=50)
        text = "a" * 210
        chunks = chunker.chunk(text, self.URL)
        for chunk in chunks:
            assert len(chunk.text) >= 50

    def test_char_start_end_are_set(self):
        chunker = self._chunker(size=200, overlap=40, min_size=10)
        text = make_text(300)
        chunks = chunker.chunk(text, self.URL)
        for chunk in chunks:
            assert chunk.char_start >= 0
            assert chunk.char_end > chunk.char_start

    def test_returns_list_of_textchunk_instances(self):
        chunker = self._chunker(min_size=10)
        text = make_text(50)
        chunks = chunker.chunk(text, self.URL)
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
