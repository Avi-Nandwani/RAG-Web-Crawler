import time
import hashlib
from collections import deque
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

from src.crawler.robots import RobotsCache
from src.crawler.fetcher import Fetcher, FetchResult
from src.crawler.parser import HTMLParser, ParsedPage
from src.utils.config import config
from src.utils.helpers import normalize_url, is_same_domain
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CrawlResult:
    """Summary returned after a crawl job completes."""

    start_url: str
    pages: List[ParsedPage] = field(default_factory=list)
    failed_urls: List[str] = field(default_factory=list)
    skipped_urls: List[str] = field(default_factory=list)

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def total_words(self) -> int:
        return sum(p.word_count for p in self.pages)


class WebCrawler:
    """
    BFS web crawler that:
    - Respects robots.txt (configurable via config.yaml)
    - Stays within the seed URL's domain
    - Limits crawl to max_pages
    - Throttles requests by crawl_delay_ms (or robots Crawl-delay if larger)
    - Saves raw HTML to data/raw/ for reproducibility
    - Returns structured ParsedPage objects
    """

    def __init__(self):
        crawler_cfg = config.crawler

        self.max_pages: int = crawler_cfg.get("max_pages", 30)
        self.max_depth: int = crawler_cfg.get("max_depth", 3)
        self.default_delay_s: float = crawler_cfg.get("crawl_delay_ms", 500) / 1000.0
        self.respect_robots: bool = crawler_cfg.get("respect_robots_txt", True)
        self.user_agent: str = crawler_cfg.get(
            "user_agent", "RAG-Web-Crawler/1.0 (Educational Project)"
        )
        timeout: int = crawler_cfg.get("timeout_seconds", 10)
        max_retries: int = crawler_cfg.get("max_retries", 3)

        self.fetcher = Fetcher(
            user_agent=self.user_agent,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.robots = RobotsCache(
            user_agent=self.user_agent,
            timeout=timeout,
        )

        self.raw_dir = Path(config.get("paths.raw_data", "./data/raw"))
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crawl(self, start_url: str) -> CrawlResult:
        """
        Crawl a website starting from start_url.

        Uses BFS with depth tracking. Stops when max_pages is reached
        or there are no more pages to visit.

        Args:
            start_url: The seed URL to begin crawling from.

        Returns:
            CrawlResult containing all successfully parsed pages.
        """
        start_url = normalize_url(start_url)
        parser = HTMLParser(base_domain_url=start_url)
        result = CrawlResult(start_url=start_url)

        # BFS queue: (url, depth)
        queue: deque = deque([(start_url, 0)])
        visited: set = set()

        logger.info(f"Starting crawl from: {start_url}")
        logger.info(f"Settings — max_pages={self.max_pages}, max_depth={self.max_depth}")

        while queue and len(result.pages) < self.max_pages:
            url, depth = queue.popleft()

            if url in visited:
                continue
            visited.add(url)

            if depth > self.max_depth:
                logger.debug(f"Max depth reached, skipping: {url}")
                result.skipped_urls.append(url)
                continue

            if not is_same_domain(url, start_url):
                logger.debug(f"Off-domain, skipping: {url}")
                result.skipped_urls.append(url)
                continue

            # Robots.txt check
            if self.respect_robots and not self.robots.can_fetch(url):
                logger.debug(f"Blocked by robots.txt: {url}")
                result.skipped_urls.append(url)
                continue

            # Rate limiting
            self._wait(url)

            # Fetch
            fetch_result: FetchResult = self.fetcher.fetch(url)
            if not fetch_result.ok:
                logger.warning(f"Failed ({fetch_result.error}): {url}")
                result.failed_urls.append(url)
                continue

            # Use the final URL (after redirects) as canonical
            canonical_url = normalize_url(fetch_result.url)

            # Parse
            page = parser.parse(canonical_url, fetch_result.html)
            result.pages.append(page)

            # Persist raw HTML
            self._save_raw(canonical_url, fetch_result.html)

            logger.info(
                f"[{len(result.pages)}/{self.max_pages}] "
                f"depth={depth} words={page.word_count} {canonical_url}"
            )

            # Enqueue discovered links
            for link in page.links:
                if link not in visited:
                    queue.append((link, depth + 1))

        logger.info(
            f"Crawl complete — "
            f"pages={result.total_pages}, "
            f"failed={len(result.failed_urls)}, "
            f"skipped={len(result.skipped_urls)}, "
            f"total_words={result.total_words}"
        )
        return result

    def close(self):
        """Release resources (HTTP session)."""
        self.fetcher.close()

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _wait(self, url: str):
        """
        Sleep for the configured delay, or the robots Crawl-delay if larger.
        """
        delay = self.default_delay_s
        if self.respect_robots:
            robots_delay = self.robots.get_crawl_delay(url)
            if robots_delay is not None:
                delay = max(delay, robots_delay)
        if delay > 0:
            time.sleep(delay)

    def _save_raw(self, url: str, html: str):
        """
        Persist raw HTML to data/raw/<hash>.html for reproducibility.
        Uses a SHA-1 hash of the URL as the filename.
        """
        try:
            url_hash = hashlib.sha1(url.encode()).hexdigest()[:16]
            filepath = self.raw_dir / f"{url_hash}.html"
            filepath.write_text(html, encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Could not save raw HTML for {url}: {exc}")
