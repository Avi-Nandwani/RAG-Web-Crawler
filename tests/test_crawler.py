import asyncio
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import pytest_asyncio
from aioresponses import aioresponses

# ---------------------------------------------------------------------------
# RobotsCache tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRobotsCache:
    """Tests for src.crawler.robots.RobotsCache"""

    @pytest_asyncio.fixture
    def cache(self):
        from src.crawler.robots import RobotsCache

        return RobotsCache(user_agent="TestBot/1.0", timeout=5)

    async def test_allows_when_robots_permits(self, cache):
        """can_fetch returns True when robots.txt allows the path."""
        with aioresponses() as m:
            m.get(
                "https://example.com/robots.txt",
                status=200,
                body="User-agent: *\nAllow: /",
            )
            assert await cache.can_fetch("https://example.com/page") is True

    async def test_disallows_when_robots_forbids(self, cache):
        """can_fetch returns False when robots.txt disallows the path."""
        with aioresponses() as m:
            m.get(
                "https://example.com/robots.txt",
                status=200,
                body="User-agent: *\nDisallow: /",
            )
            assert await cache.can_fetch("https://example.com/page") is False

    async def test_allows_on_404(self, cache):
        """can_fetch returns True when robots.txt returns 404 (no restrictions)."""
        with aioresponses() as m:
            m.get("https://example.com/robots.txt", status=404)
            assert await cache.can_fetch("https://example.com/page") is True

    async def test_allows_on_network_error(self, cache):
        """can_fetch returns True (fail open) when the request raises an exception."""
        with aioresponses() as m:
            m.get("https://example.com/robots.txt", exception=asyncio.TimeoutError)
            assert await cache.can_fetch("https://example.com/page") is True

    async def test_caches_per_domain(self, cache):
        """robots.txt is fetched only once per domain regardless of how many URLs are checked."""
        with aioresponses() as m:
            m.get(
                "https://example.com/robots.txt",
                status=200,
                body="User-agent: *\nAllow: /",
            )
            await cache.can_fetch("https://example.com/a")
            await cache.can_fetch("https://example.com/b")
            await cache.can_fetch("https://example.com/c")
            # The request count is checked on the mock object
            assert len(m.requests) == 1

    async def test_crawl_delay_returned(self, cache):
        """get_crawl_delay returns the value from Crawl-delay directive."""
        with aioresponses() as m:
            m.get(
                "https://example.com/robots.txt",
                status=200,
                body="User-agent: *\nAllow: /\nCrawl-delay: 2",
            )
            delay = await cache.get_crawl_delay("https://example.com/page")
            assert delay == 2.0


# ---------------------------------------------------------------------------
# Fetcher tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFetcher:
    """Tests for src.crawler.fetcher.Fetcher"""

    @pytest_asyncio.fixture
    async def fetcher(self):
        from src.crawler.fetcher import Fetcher

        fetcher = Fetcher(user_agent="TestBot/1.0", timeout=5, max_retries=2)
        yield fetcher
        await fetcher.close()

    async def test_successful_fetch(self, fetcher):
        """fetch() returns ok=True with HTML content on 200 response."""
        url = "https://example.com/"
        html = "<html><body>Hello</body></html>"
        with aioresponses() as m:
            m.get(url, status=200, body=html, headers={"Content-Type": "text/html"})
            result = await fetcher.fetch(url)
            assert result.ok is True
            assert result.url == url
            assert result.html == html
            assert result.text is None  # Should not be populated for HTML
            assert result.content_type == "text/html"

    async def test_successful_fetch_text(self, fetcher):
        """fetch() returns ok=True with text content on 200 response."""
        url = "https://example.com/doc.txt"
        text = "This is a test document."
        with aioresponses() as m:
            m.get(
                url, status=200, body=text, headers={"Content-Type": "text/plain"}
            )
            result = await fetcher.fetch(url)
            assert result.ok is True
            assert result.url == url
            assert result.html is None
            assert result.text == text
            assert result.content_type == "text/plain"

    async def test_fetch_non_html_or_text(self, fetcher):
        """fetch() returns ok=False for non-text/HTML content types."""
        url = "https://example.com/image.jpg"
        with aioresponses() as m:
            m.get(
                url, status=200, headers={"Content-Type": "image/jpeg"},
            )
            result = await fetcher.fetch(url)
            assert result.ok is False
            assert "unsupported content type" in result.error.lower()

    async def test_fetch_http_error(self, fetcher):
        """fetch() returns ok=False on HTTP error status codes (e.g., 404, 500)."""
        url = "https://example.com/404"
        with aioresponses() as m:
            m.get(url, status=404)
            result = await fetcher.fetch(url)
            assert result.ok is False
            assert "404" in result.error

    async def test_fetch_network_error(self, fetcher):
        """fetch() returns ok=False on network errors (e.g., timeout)."""
        url = "https://example.com/timeout"
        with aioresponses() as m:
            m.get(url, exception=asyncio.TimeoutError)
            result = await fetcher.fetch(url)
            assert result.ok is False
            assert "timeout" in result.error.lower()

    async def test_retry_logic(self, fetcher):
        """fetch() retries on transient errors up to max_retries."""
        url = "https://example.com/retry"
        with aioresponses() as m:
            # Fail twice, then succeed
            m.get(url, status=503)
            m.get(url, status=503)
            m.get(url, status=200, body="Success", headers={"Content-Type": "text/plain"})
            result = await fetcher.fetch(url)
            assert result.ok is True
            assert result.text == "Success"
            total_calls = sum(len(calls) for calls in m.requests.values())
            assert total_calls == 3  # 1 initial + 2 retries

    async def test_retry_limit_exceeded(self, fetcher):
        """fetch() fails after exceeding max_retries."""
        url = "https://example.com/fail"
        with aioresponses() as m:
            m.get(url, status=503, repeat=fetcher.max_retries + 1)
            result = await fetcher.fetch(url)
            assert result.ok is False
            assert "503" in result.error
            total_calls = sum(len(calls) for calls in m.requests.values())
            assert total_calls == fetcher.max_retries + 1

    async def test_redirect_handling(self, fetcher):
        """fetch() should follow redirects and return content from the final URL."""
        initial_url = "https://example.com/redirect"
        final_url = "https://example.com/final"
        with aioresponses() as m:
            m.get(initial_url, status=301, headers={"Location": final_url})
            m.get(final_url, status=200, body="Final Page", headers={"Content-Type": "text/plain"})
            result = await fetcher.fetch(initial_url)
            assert result.ok is True
            assert result.url == str(final_url)
            assert result.text == "Final Page"

    async def test_close_session(self, fetcher):
        """close() should close the aiohttp session."""
        # We can't directly check if the session is closed, but we can check that the method is called.
        # A better test might be to see if a subsequent fetch fails, but that's complex to set up.
        session = await fetcher.get_session()
        with patch.object(session, "close", new_callable=AsyncMock) as mock_close:
            await fetcher.close()
            mock_close.assert_called_once()


# ---------------------------------------------------------------------------
# WebCrawler tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWebCrawler:
    """Tests for src.crawler.crawler.WebCrawler"""

    @pytest_asyncio.fixture
    def crawler(self):
        from src.crawler.crawler import WebCrawler

        crawler = WebCrawler()
        crawler.concurrency = 1
        crawler.max_pages = 10
        crawler.max_depth = 2
        crawler.default_delay_s = 0
        return crawler

    async def test_crawl_simple_site(self, crawler):
        """Crawl a simple site with a few pages and check results."""
        start_url = "https://example.com/start"
        page1_url = "https://example.com/page1"
        page2_url = "https://example.com/page2"

        with aioresponses() as m:
            # Mock robots.txt to allow everything
            m.get("https://example.com/robots.txt", status=200, body="User-agent: *\nAllow: /")
            # Mock pages
            m.get(start_url, status=200, body=f'<html><body><a href="{page1_url}">1</a><a href="{page2_url}">2</a></body></html>', headers={"Content-Type": "text/html"})
            m.get(page1_url, status=200, body='<html><body>Page 1 Content</body></html>', headers={"Content-Type": "text/html"})
            m.get(page2_url, status=200, body='<html><body>Page 2 Content</body></html>', headers={"Content-Type": "text/html"})

            result = await crawler.crawl(start_url)

            assert result.total_pages == 3
            urls_crawled = {r.url for r in result.pages}
            assert urls_crawled == {start_url, page1_url, page2_url}

    async def test_respects_max_depth(self, crawler):
        """Crawler should not go deeper than max_depth."""
        crawler.max_depth = 1
        start_url = "https://example.com/start"
        depth1_url = "https://example.com/depth1"
        depth2_url = "https://example.com/depth2" # This should not be crawled

        with aioresponses() as m:
            m.get("https://example.com/robots.txt", status=200, body="User-agent: *\nAllow: /")
            m.get(start_url, status=200, body=f'<html><a href="{depth1_url}">1</a></html>', headers={"Content-Type": "text/html"})
            m.get(depth1_url, status=200, body=f'<html><a href="{depth2_url}">2</a></html>', headers={"Content-Type": "text/html"})
            # The crawler should not even attempt to fetch depth2_url
            m.get(depth2_url, status=200, body='<html></html>', headers={"Content-Type": "text/html"})

            result = await crawler.crawl(start_url)

            assert result.total_pages == 2
            urls_crawled = {r.url for r in result.pages}
            assert urls_crawled == {start_url, depth1_url}
            assert depth2_url not in urls_crawled

    async def test_respects_max_pages(self, crawler):
        """Crawler should stop after reaching max_pages."""
        crawler.max_pages = 2
        start_url = "https://example.com/start"
        page1_url = "https://example.com/page1"
        page2_url = "https://example.com/page2" # This might not be crawled if concurrency is > 1

        with aioresponses() as m:
            m.get("https://example.com/robots.txt", status=200, body="User-agent: *\nAllow: /")
            m.get(start_url, status=200, body=f'<html><a href="{page1_url}">1</a><a href="{page2_url}">2</a></html>', headers={"Content-Type": "text/html"})
            m.get(page1_url, status=200, body='<html>Page 1</html>', headers={"Content-Type": "text/html"})
            m.get(page2_url, status=200, body='<html>Page 2</html>', headers={"Content-Type": "text/html"})

            result = await crawler.crawl(start_url)

            assert result.total_pages <= crawler.max_pages

    async def test_respects_robots_txt(self, crawler):
        """Crawler should not fetch URLs disallowed by robots.txt."""
        start_url = "https://example.com/start"
        allowed_url = "https://example.com/allowed"
        disallowed_url = "https://example.com/disallowed"

        robots_body = "User-agent: *\nAllow: /allowed\nDisallow: /disallowed"

        with aioresponses() as m:
            m.get("https://example.com/robots.txt", status=200, body=robots_body)
            m.get(start_url, status=200, body=f'<html><a href="{allowed_url}">1</a><a href="{disallowed_url}">2</a></html>', headers={"Content-Type": "text/html"})
            m.get(allowed_url, status=200, body='<html>Allowed</html>', headers={"Content-Type": "text/html"})
            # This should not be called
            m.get(disallowed_url, status=200, body='<html>Disallowed</html>', headers={"Content-Type": "text/html"})

            result = await crawler.crawl(start_url)

            urls_crawled = {r.url for r in result.pages}
            assert urls_crawled == {start_url, allowed_url}
            assert disallowed_url not in urls_crawled

    async def test_handles_url_normalization(self, crawler):
        """Crawler should handle URL normalization and avoid re-visiting."""
        start_url = "https://example.com/"
        link1 = "https://example.com/page#fragment"
        link2 = "https://example.com/page"

        with aioresponses() as m:
            m.get("https://example.com/robots.txt", status=200, body="User-agent: *\nAllow: /")
            m.get(start_url, status=200, body=f'<html><a href="{link1}">1</a><a href="{link2}">2</a></html>', headers={"Content-Type": "text/html"})
            m.get("https://example.com/page", status=200, body='<html>Page</html>', headers={"Content-Type": "text/html"})

            result = await crawler.crawl(start_url)

            # Should have crawled the start page and the normalized page ONCE
            assert result.total_pages == 2
            urls_crawled = {r.url for r in result.pages}
            assert urls_crawled == {start_url, "https://example.com/page"}

    async def test_handles_crawl_delay(self, crawler):
        """Crawler should respect Crawl-delay from robots.txt."""
        # This is hard to test directly without making the test very slow.
        # We can test that the delay is retrieved and that asyncio.sleep is called.
        start_url = "https://example.com/start"
        robots_body = "User-agent: *\nAllow: /\nCrawl-delay: 1"

        with aioresponses() as m:
            m.get("https://example.com/robots.txt", status=200, body=robots_body)
            m.get(start_url, status=200, body='<html></html>', headers={"Content-Type": "text/html"})

            crawler.default_delay_s = 0.0
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                await crawler.crawl(start_url)

                # Check if sleep was called with the crawl delay
                mock_sleep.assert_any_call(1.0)

    async def test_close_cleans_up_resources(self, crawler):
           """close() should call close on its fetcher and robots cache."""
           with patch.object(crawler.fetcher, "close", new_callable=AsyncMock) as mock_fetcher_close, \
               patch.object(crawler.robots, "close", new_callable=AsyncMock) as mock_robots_close:
              await crawler.close()

              mock_fetcher_close.assert_called_once()
              mock_robots_close.assert_called_once()
