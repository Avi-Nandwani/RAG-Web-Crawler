"""
HTML parser: extracts title, clean body text, and internal links.
Strips boilerplate (scripts, styles, nav, header, footer) before
returning text so downstream chunking gets clean content only.
"""

from typing import List, Optional
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag

from src.utils.helpers import normalize_url, is_same_domain
from src.utils.logger import get_logger

logger = get_logger(__name__)

# HTML tags whose entire subtree we discard before extracting text
_NOISE_TAGS = [
    "script", "style", "noscript",
    "nav", "header", "footer",
    "aside", "form", "button",
    "iframe", "svg", "img",
    "[document]", "html", "head",
]


@dataclass
class ParsedPage:
    """Structured output from parsing a single HTML page."""

    url: str
    title: str = ""
    text: str = ""                     # Clean body text
    links: List[str] = field(default_factory=list)  # Absolute, normalized internal links
    word_count: int = 0


class HTMLParser:
    """
    Parses raw HTML using BeautifulSoup4.
    - Title: <title> tag text, cleaned.
    - Text: visible text from <body> with noise tags removed.
    - Links: absolute URLs of all internal <a href> links.
    """

    def __init__(self, base_domain_url: str):
        """
        Args:
            base_domain_url: The seed/start URL used to determine domain boundaries
                             for link extraction (e.g. 'https://example.com').
        """
        self.base_domain_url = base_domain_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, url: str, html: str) -> ParsedPage:
        """
        Parse raw HTML into a ParsedPage.

        Args:
            url: Final URL of the page (used for resolving relative links).
            html: Raw HTML string.

        Returns:
            ParsedPage with title, text, and internal links.
        """
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")

        title = self._extract_title(soup)
        text = self._extract_text(soup)
        links = self._extract_links(soup, url)

        return ParsedPage(
            url=url,
            title=title,
            text=text,
            links=links,
            word_count=len(text.split()),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Return the <title> text, stripped of whitespace."""
        tag = soup.find("title")
        if tag:
            return tag.get_text(strip=True)

        # Fallback: first <h1>
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)

        return ""

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """
        Return visible body text with noise tags removed.
        Multiple whitespace characters are collapsed to single spaces.
        """
        # Work on a copy so we don't mutate the original
        body = soup.find("body")
        if not body:
            body = soup

        # Remove noisy subtrees
        for tag_name in _NOISE_TAGS:
            for tag in body.find_all(tag_name):
                tag.decompose()

        raw_text = body.get_text(separator=" ", strip=True)

        # Collapse whitespace
        import re
        text = re.sub(r"\s+", " ", raw_text).strip()
        return text

    def _extract_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """
        Return a deduplicated list of absolute URLs for internal <a href> links.
        - Relative URLs are resolved against current_url.
        - Off-domain URLs are excluded.
        - Anchors, mailto:, javascript: etc. are excluded.
        """
        seen = set()
        links: List[str] = []

        for a_tag in soup.find_all("a", href=True):
            href: str = a_tag["href"].strip()

            # Skip non-page links
            if not href or href.startswith(("mailto:", "javascript:", "tel:", "#")):
                continue

            # Resolve relative URLs
            absolute = urljoin(current_url, href)

            # Validate scheme
            parsed = urlparse(absolute)
            if parsed.scheme not in ("http", "https"):
                continue

            # Stay within the same domain
            if not is_same_domain(absolute, self.base_domain_url):
                continue

            # Normalize and deduplicate
            normalized = normalize_url(absolute)
            if normalized not in seen:
                seen.add(normalized)
                links.append(normalized)

        return links
