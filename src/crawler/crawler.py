import asyncio
import hashlib
import time
from asyncio import Queue, Semaphore
from pathlib import Path
from typing import List, Set
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
    crawl_time_s: float = 0.0

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def total_words(self) -> int:
        return sum(p.word_count for p in self.pages)


class WebCrawler:
    """
    Async BFS web crawler that:
    - Respects robots.txt (configurable)
    - Stays within the seed URL's domain
    - Limits crawl to max_pages and max_depth
    - Uses a semaphore to control concurrency
    - Throttles requests based on crawl_delay
    - Saves raw HTML for reproducibility
    """

    def __init__(self):
        cfg = config.crawler
        self.max_pages: int = cfg.get("max_pages", 30)
        self.max_depth: int = cfg.get("max_depth", 3)
        self.default_delay_s: float = cfg.get("crawl_delay_ms", 500) / 1000.0
        self.respect_robots: bool = cfg.get("respect_robots_txt", True)
        self.user_agent: str = cfg.get(
            "user_agent", "RAG-Web-Crawler/1.0 (Educational Project)"
        )
        self.concurrency: int = cfg.get("concurrency", 5)

        self.fetcher = Fetcher(
            user_agent=self.user_agent,
            timeout=cfg.get("timeout_seconds", 10),
            max_retries=cfg.get("max_retries", 3),
        )
        self.robots = RobotsCache(
            user_agent=self.user_agent,
            timeout=cfg.get("timeout_seconds", 10),
        )

        self.raw_dir = Path(config.get("paths.raw_data", "./data/raw"))
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def crawl(self, start_url: str) -> CrawlResult:
        """
        Asynchronously crawl a website starting from start_url.
        """
        started = time.perf_counter()
        start_url = normalize_url(start_url)
        result = CrawlResult(start_url=start_url)
        semaphore = Semaphore(self.concurrency)

        # BFS queue: (url, depth)
        queue: Queue = Queue()
        queue.put_nowait((start_url, 0))
        visited: Set[str] = {start_url}

        logger.info(f"Starting async crawl from: {start_url}")
        logger.info(
            f"Settings — max_pages={self.max_pages}, max_depth={self.max_depth}, "
            f"concurrency={self.concurrency}"
        )

        while not queue.empty() and len(result.pages) < self.max_pages:
            tasks = []
            # Create a batch of tasks up to the concurrency limit
            for _ in range(min(queue.qsize(), self.concurrency)):
                if queue.empty():
                    break
                url, depth = queue.get_nowait()
                task = asyncio.create_task(
                    self._process_url(url, depth, queue, visited, result, semaphore)
                )
                tasks.append(task)

            if tasks:
                await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - started
        result.crawl_time_s = round(elapsed, 2)
        logger.info(
            f"Crawl complete in {result.crawl_time_s:.2f}s — "
            f"pages={result.total_pages}, failed={len(result.failed_urls)}, "
            f"skipped={len(result.skipped_urls)}, total_words={result.total_words}"
        )
        return result

    async def close(self):
        """Release resources (HTTP sessions)."""
        await self.fetcher.close()
        await self.robots.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _process_url(
        self,
        url: str,
        depth: int,
        queue: Queue,
        visited: Set[str],
        result: CrawlResult,
        semaphore: Semaphore,
    ):
        """Pipeline for fetching, parsing, and enqueuing links for a single URL."""
        async with semaphore:
            if len(result.pages) >= self.max_pages:
                return

            if not await self._should_fetch(url, depth, result):
                return

            # Rate limiting
            await self._wait(url)

            # Fetch
            fetch_result = await self.fetcher.fetch(url)
            if not fetch_result.ok:
                logger.warning(f"Failed ({fetch_result.error}): {url}")
                result.failed_urls.append(url)
                return

            # Use the final URL (after redirects) as canonical
            canonical_url = normalize_url(fetch_result.url)
            parser = HTMLParser(base_domain_url=result.start_url)
            page = parser.parse(canonical_url, fetch_result.html)
            result.pages.append(page)

            self._save_raw(canonical_url, fetch_result.html)

            logger.info(
                f"[{len(result.pages)}/{self.max_pages}] "
                f"depth={depth} words={page.word_count} {canonical_url}"
            )

            # Enqueue discovered links
            for link in page.links:
                if link not in visited and len(result.pages) + queue.qsize() < self.max_pages:
                    visited.add(link)
                    await queue.put((link, depth + 1))

    async def _should_fetch(self, url: str, depth: int, result: CrawlResult) -> bool:
        """Pre-flight checks to determine if a URL should be fetched."""
        if depth > self.max_depth:
            logger.debug(f"Max depth reached, skipping: {url}")
            result.skipped_urls.append(url)
            return False

        if not is_same_domain(url, result.start_url):
            logger.debug(f"Off-domain, skipping: {url}")
            result.skipped_urls.append(url)
            return False

        if self.respect_robots and not await self.robots.can_fetch(url):
            logger.debug(f"Blocked by robots.txt: {url}")
            result.skipped_urls.append(url)
            return False

        return True

    async def _wait(self, url: str):
        """Sleep for the configured delay or the robots Crawl-delay."""
        delay = self.default_delay_s
        if self.respect_robots:
            robots_delay = await self.robots.get_crawl_delay(url)
            if robots_delay is not None:
                delay = max(delay, robots_delay)
        if delay > 0:
            await asyncio.sleep(delay)

    def _save_raw(self, url: str, html: str):
        """Persist raw HTML to data/raw/<hash>.html."""
        try:
            url_hash = hashlib.sha1(url.encode()).hexdigest()[:16]
            filepath = self.raw_dir / f"{url_hash}.html"
            filepath.write_text(html, encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Could not save raw HTML for {url}: {exc}")
