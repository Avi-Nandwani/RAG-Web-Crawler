"""
Unit tests for Week 2 — Web Crawler module.

All network calls are mocked so tests run offline.

Test coverage:
  - RobotsCache: allow / disallow / network failure
  - Fetcher: success, HTTP error, non-HTML content type, retry logic
  - HTMLParser: title extraction, text cleaning, link extraction, domain filtering
  - WebCrawler: domain boundary, max_pages limit, robots.txt gate, raw HTML saved
"""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

# ---------------------------------------------------------------------------
# RobotsCache tests
# ---------------------------------------------------------------------------

class TestRobotsCache:
    """Tests for src.crawler.robots.RobotsCache"""

    def _make_cache(self):
        from src.crawler.robots import RobotsCache
        return RobotsCache(user_agent="TestBot/1.0", timeout=5)

    def test_allows_when_robots_permits(self):
        """can_fetch returns True when robots.txt allows the path."""
        robots_txt = "User-agent: *\nAllow: /"
        cache = self._make_cache()

        with patch("src.crawler.robots.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, text=robots_txt)
            assert cache.can_fetch("https://example.com/page") is True

    def test_disallows_when_robots_forbids(self):
        """can_fetch returns False when robots.txt disallows the path."""
        robots_txt = "User-agent: *\nDisallow: /"
        cache = self._make_cache()

        with patch("src.crawler.robots.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, text=robots_txt)
            assert cache.can_fetch("https://example.com/page") is False

    def test_allows_on_404(self):
        """can_fetch returns True when robots.txt returns 404 (no restrictions)."""
        cache = self._make_cache()

        with patch("src.crawler.robots.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=404, text="")
            assert cache.can_fetch("https://example.com/page") is True

    def test_allows_on_network_error(self):
        """can_fetch returns True (fail open) when the request raises an exception."""
        from requests.exceptions import ConnectionError as ReqConnError
        cache = self._make_cache()

        with patch("src.crawler.robots.requests.get", side_effect=ReqConnError("timeout")):
            assert cache.can_fetch("https://example.com/page") is True

    def test_caches_per_domain(self):
        """robots.txt is fetched only once per domain regardless of how many URLs are checked."""
        robots_txt = "User-agent: *\nAllow: /"
        cache = self._make_cache()

        with patch("src.crawler.robots.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, text=robots_txt)
            cache.can_fetch("https://example.com/a")
            cache.can_fetch("https://example.com/b")
            cache.can_fetch("https://example.com/c")
            assert mock_get.call_count == 1  # Only one robots.txt download

    def test_crawl_delay_returned(self):
        """get_crawl_delay returns the value from Crawl-delay directive."""
        robots_txt = "User-agent: *\nAllow: /\nCrawl-delay: 2"
        cache = self._make_cache()

        with patch("src.crawler.robots.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, text=robots_txt)
            delay = cache.get_crawl_delay("https://example.com/page")
            assert delay == 2.0


# ---------------------------------------------------------------------------
# Fetcher tests
# ---------------------------------------------------------------------------

class TestFetcher:
    """Tests for src.crawler.fetcher.Fetcher"""

    def _make_fetcher(self):
        from src.crawler.fetcher import Fetcher
        return Fetcher(user_agent="TestBot/1.0", timeout=5, max_retries=2)

    def _mock_response(self, status_code: int, text: str, content_type: str, url: str):
        m = MagicMock()
        m.status_code = status_code
        m.text = text
        m.url = url
        m.headers = {"Content-Type": content_type}
        return m

    def test_successful_fetch(self):
        """fetch() returns ok=True with HTML content on 200 response."""
        fetcher = self._make_fetcher()
        html = "<html><body>Hello</body></html>"

        with patch.object(fetcher.session, "get") as mock_get:
            mock_get.return_value = self._mock_response(
                200, html, "text/html; charset=utf-8", "https://example.com/"
            )
            result = fetcher.fetch("https://example.com/")

        assert result.ok is True
        assert result.html == html
        assert result.status_code == 200

    def test_404_returns_failure(self):
        """fetch() returns ok=False for a 404 response."""
        fetcher = self._make_fetcher()

        with patch.object(fetcher.session, "get") as mock_get:
            mock_get.return_value = self._mock_response(
                404, "", "text/html", "https://example.com/missing"
            )
            result = fetcher.fetch("https://example.com/missing")

        assert result.ok is False
        assert "404" in result.error

    def test_non_html_returns_failure(self):
        """fetch() returns ok=False when Content-Type is not text/html."""
        fetcher = self._make_fetcher()

        with patch.object(fetcher.session, "get") as mock_get:
            mock_get.return_value = self._mock_response(
                200, b"%PDF-1.4", "application/pdf", "https://example.com/file.pdf"
            )
            result = fetcher.fetch("https://example.com/file.pdf")

        assert result.ok is False
        assert "Non-HTML" in result.error

    def test_network_exception_returns_failure(self):
        """fetch() returns ok=False when a network exception is raised."""
        from requests.exceptions import ConnectionError as ReqConnError
        fetcher = self._make_fetcher()

        with patch.object(fetcher.session, "get", side_effect=ReqConnError("no route")):
            result = fetcher.fetch("https://example.com/")

        assert result.ok is False
        assert result.error is not None

    def test_retry_on_500(self):
        """fetch() retries on 500 errors before giving up."""
        fetcher = self._make_fetcher()
        fail_resp = self._mock_response(500, "", "text/html", "https://example.com/")

        with patch.object(fetcher.session, "get", return_value=fail_resp) as mock_get:
            with patch("src.crawler.fetcher.time.sleep"):  # Skip actual sleep
                result = fetcher.fetch("https://example.com/")

        assert result.ok is False
        # Should have tried max_retries times
        assert mock_get.call_count == fetcher.max_retries


# ---------------------------------------------------------------------------
# HTMLParser tests
# ---------------------------------------------------------------------------

class TestHTMLParser:
    """Tests for src.crawler.parser.HTMLParser"""

    BASE_URL = "https://example.com"

    def _parser(self):
        from src.crawler.parser import HTMLParser
        return HTMLParser(base_domain_url=self.BASE_URL)

    def test_extracts_title(self):
        html = "<html><head><title>Hello World</title></head><body></body></html>"
        page = self._parser().parse(self.BASE_URL, html)
        assert page.title == "Hello World"

    def test_falls_back_to_h1_title(self):
        html = "<html><body><h1>My Page</h1><p>Content</p></body></html>"
        page = self._parser().parse(self.BASE_URL, html)
        assert page.title == "My Page"

    def test_strips_script_tags(self):
        html = "<html><body><p>Good text</p><script>bad code</script></body></html>"
        page = self._parser().parse(self.BASE_URL, html)
        assert "bad code" not in page.text
        assert "Good text" in page.text

    def test_strips_nav_and_footer(self):
        html = "<html><body><nav>Menu</nav><p>Article</p><footer>Footer</footer></body></html>"
        page = self._parser().parse(self.BASE_URL, html)
        assert "Menu" not in page.text
        assert "Footer" not in page.text
        assert "Article" in page.text

    def test_extracts_internal_links(self):
        html = (
            '<html><body>'
            '<a href="/about">About</a>'
            '<a href="https://example.com/contact">Contact</a>'
            '</body></html>'
        )
        page = self._parser().parse(self.BASE_URL + "/", html)
        assert "https://example.com/about" in page.links
        assert "https://example.com/contact" in page.links

    def test_excludes_external_links(self):
        html = (
            '<html><body>'
            '<a href="https://other.com/page">External</a>'
            '<a href="/internal">Internal</a>'
            '</body></html>'
        )
        page = self._parser().parse(self.BASE_URL, html)
        assert not any("other.com" in l for l in page.links)
        assert any("example.com/internal" in l for l in page.links)

    def test_excludes_mailto_and_anchors(self):
        html = (
            '<html><body>'
            '<a href="mailto:test@example.com">Email</a>'
            '<a href="#section">Anchor</a>'
            '<a href="javascript:void(0)">JS</a>'
            '</body></html>'
        )
        page = self._parser().parse(self.BASE_URL, html)
        assert len(page.links) == 0

    def test_deduplicates_links(self):
        html = (
            '<html><body>'
            '<a href="/page">Link 1</a>'
            '<a href="/page">Link 2</a>'
            '<a href="/page/">Link 3 (trailing slash)</a>'
            '</body></html>'
        )
        page = self._parser().parse(self.BASE_URL, html)
        assert len(page.links) == 1

    def test_word_count(self):
        html = "<html><body><p>one two three four five</p></body></html>"
        page = self._parser().parse(self.BASE_URL, html)
        assert page.word_count == 5


# ---------------------------------------------------------------------------
# WebCrawler integration tests (mocked network)
# ---------------------------------------------------------------------------

class TestWebCrawler:
    """Tests for src.crawler.crawler.WebCrawler (network fully mocked)."""

    SEED = "https://example.com"
    HTML_SEED = (
        '<html><head><title>Home</title></head>'
        '<body><p>Welcome</p>'
        '<a href="/about">About</a>'
        '<a href="https://evil.com/page">Evil</a>'
        '</body></html>'
    )
    HTML_ABOUT = (
        '<html><head><title>About</title></head>'
        '<body><p>About us</p></body></html>'
    )

    def _make_crawler(self, tmp_path):
        """Return a WebCrawler whose raw_dir points to tmp_path."""
        from src.crawler.crawler import WebCrawler
        crawler = WebCrawler()
        crawler.raw_dir = tmp_path
        return crawler

    def _setup_mocks(self, fetcher_mock, robots_mock):
        """Configure default mock behaviour: allow all, return HTML pages."""
        from src.crawler.fetcher import FetchResult

        robots_mock.can_fetch.return_value = True
        robots_mock.get_crawl_delay.return_value = 0

        def fake_fetch(url):
            if "about" in url:
                return FetchResult(
                    url=url, original_url=url, html=self.HTML_ABOUT,
                    status_code=200, content_type="text/html", ok=True
                )
            return FetchResult(
                url=url, original_url=url, html=self.HTML_SEED,
                status_code=200, content_type="text/html", ok=True
            )

        fetcher_mock.fetch.side_effect = fake_fetch

    @pytest.fixture
    def crawler(self, tmp_path):
        return self._make_crawler(tmp_path)

    def test_crawls_seed_page(self, crawler, tmp_path):
        """Crawler returns at least the seed page."""
        with patch.object(crawler, "fetcher") as m_fetch, \
             patch.object(crawler, "robots") as m_robots, \
             patch("src.crawler.crawler.time.sleep"):
            self._setup_mocks(m_fetch, m_robots)
            result = crawler.crawl(self.SEED)

        assert result.total_pages >= 1
        assert any("example.com" in p.url for p in result.pages)

    def test_excludes_off_domain_links(self, crawler, tmp_path):
        """Crawler never visits off-domain URLs."""
        with patch.object(crawler, "fetcher") as m_fetch, \
             patch.object(crawler, "robots") as m_robots, \
             patch("src.crawler.crawler.time.sleep"):
            self._setup_mocks(m_fetch, m_robots)
            result = crawler.crawl(self.SEED)

        visited = [p.url for p in result.pages]
        assert not any("evil.com" in u for u in visited)

    def test_respects_max_pages(self, crawler, tmp_path):
        """Crawler stops after max_pages regardless of remaining queue."""
        crawler.max_pages = 1

        with patch.object(crawler, "fetcher") as m_fetch, \
             patch.object(crawler, "robots") as m_robots, \
             patch("src.crawler.crawler.time.sleep"):
            self._setup_mocks(m_fetch, m_robots)
            result = crawler.crawl(self.SEED)

        assert result.total_pages <= 1

    def test_robots_gate_blocks_url(self, crawler, tmp_path):
        """Crawler skips URLs disallowed by robots.txt."""
        from src.crawler.fetcher import FetchResult

        with patch.object(crawler, "fetcher") as m_fetch, \
             patch.object(crawler, "robots") as m_robots, \
             patch("src.crawler.crawler.time.sleep"):

            m_robots.can_fetch.return_value = False  # Block everything
            m_robots.get_crawl_delay.return_value = 0
            m_fetch.fetch.return_value = FetchResult(
                url=self.SEED, original_url=self.SEED,
                html=self.HTML_SEED, status_code=200,
                content_type="text/html", ok=True,
            )
            result = crawler.crawl(self.SEED)

        assert result.total_pages == 0
        assert self.SEED in result.skipped_urls

    def test_saves_raw_html(self, crawler, tmp_path):
        """Crawler persists raw HTML files to raw_dir."""
        with patch.object(crawler, "fetcher") as m_fetch, \
             patch.object(crawler, "robots") as m_robots, \
             patch("src.crawler.crawler.time.sleep"):
            self._setup_mocks(m_fetch, m_robots)
            crawler.max_pages = 1
            crawler.crawl(self.SEED)

        html_files = list(tmp_path.glob("*.html"))
        assert len(html_files) >= 1

    def test_failed_fetch_recorded(self, crawler, tmp_path):
        """Failed fetches are recorded in CrawlResult.failed_urls."""
        from src.crawler.fetcher import FetchResult

        with patch.object(crawler, "fetcher") as m_fetch, \
             patch.object(crawler, "robots") as m_robots, \
             patch("src.crawler.crawler.time.sleep"):

            m_robots.can_fetch.return_value = True
            m_robots.get_crawl_delay.return_value = 0
            m_fetch.fetch.return_value = FetchResult(
                url=self.SEED, original_url=self.SEED,
                ok=False, error="HTTP 500"
            )
            result = crawler.crawl(self.SEED)

        assert self.SEED in result.failed_urls
        assert result.total_pages == 0
