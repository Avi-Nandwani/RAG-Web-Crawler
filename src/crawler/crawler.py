import asyncio
import hashlib
import json
import time
from asyncio import Queue, Semaphore
from pathlib import Path
from typing import List, Set, Optional
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

        self.checkpoint_dir = Path(
            cfg.get("checkpoint_dir", config.get("paths.checkpoints", "./data/checkpoints"))
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_every_pages: int = cfg.get("checkpoint_every_pages", 10)
        self.checkpoint_every_seconds: int = cfg.get("checkpoint_every_seconds", 30)
        self._last_checkpoint_time = 0.0
        self._last_checkpoint_pages = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def crawl(
        self,
        start_url: str,
        resume: bool = False,
        checkpoint_path: Optional[str] = None,
    ) -> CrawlResult:
        """
        Asynchronously crawl a website starting from start_url.
        Set resume=True to continue from a saved checkpoint if present.
        """
        started = time.perf_counter()
        start_url = normalize_url(start_url)
        checkpoint_file = (
            Path(checkpoint_path) if checkpoint_path else self._checkpoint_path(start_url)
        )

        if not resume and checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
            except Exception as exc:
                logger.warning(f"Could not remove stale checkpoint {checkpoint_file}: {exc}")

        result = CrawlResult(start_url=start_url)
        semaphore = Semaphore(self.concurrency)

        # BFS queue: (url, depth)
        queue: Queue = Queue()
        visited: Set[str] = set()
        elapsed_offset = 0.0

        if resume:
            state = await self._load_checkpoint(checkpoint_file)
            if state:
                result = self._checkpoint_to_result(state)
                visited = set(state.get("visited", []))
                for item in state.get("queue", []):
                    if isinstance(item, list) and len(item) == 2:
                        queue.put_nowait((item[0], int(item[1])))
                elapsed_offset = float(state.get("elapsed_s", 0.0))
                start_url = result.start_url
                logger.info(f"Resuming crawl from checkpoint: {checkpoint_file}")
            else:
                queue.put_nowait((start_url, 0))
                visited = {start_url}
        else:
            queue.put_nowait((start_url, 0))
            visited = {start_url}

        self._last_checkpoint_time = time.perf_counter()
        self._last_checkpoint_pages = len(result.pages)

        logger.info(f"Starting async crawl from: {start_url}")
        logger.info(
            f"Settings — max_pages={self.max_pages}, max_depth={self.max_depth}, "
            f"concurrency={self.concurrency}"
        )

        try:
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

                await self._maybe_checkpoint(
                    checkpoint_file,
                    result,
                    queue,
                    visited,
                    elapsed_offset,
                    started,
                )
        except (asyncio.CancelledError, KeyboardInterrupt):
            await self._save_checkpoint(
                checkpoint_file,
                result,
                queue,
                visited,
                elapsed_offset,
                started,
            )
            logger.warning(f"Crawl interrupted; checkpoint saved to {checkpoint_file}")
            raise

        elapsed = elapsed_offset + (time.perf_counter() - started)
        result.crawl_time_s = round(elapsed, 2)
        logger.info(
            f"Crawl complete in {result.crawl_time_s:.2f}s — "
            f"pages={result.total_pages}, failed={len(result.failed_urls)}, "
            f"skipped={len(result.skipped_urls)}, total_words={result.total_words}"
        )
        await self._clear_checkpoint(checkpoint_file)
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
            if fetch_result.html is None and fetch_result.text is not None:
                text = fetch_result.text
                page = ParsedPage(
                    url=canonical_url,
                    title="",
                    text=text,
                    links=[],
                    word_count=len(text.split()),
                )
            else:
                parser = HTMLParser(base_domain_url=result.start_url)
                page = parser.parse(canonical_url, fetch_result.html)

            result.pages.append(page)

            raw_content = fetch_result.html if fetch_result.html is not None else fetch_result.text
            if raw_content is not None:
                self._save_raw(canonical_url, raw_content)

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

    def _checkpoint_path(self, start_url: str) -> Path:
        url_hash = hashlib.sha1(start_url.encode()).hexdigest()[:16]
        return self.checkpoint_dir / f"crawl_{url_hash}.json"

    async def _maybe_checkpoint(
        self,
        checkpoint_file: Path,
        result: CrawlResult,
        queue: Queue,
        visited: Set[str],
        elapsed_offset: float,
        started: float,
    ):
        if self.checkpoint_every_pages <= 0 and self.checkpoint_every_seconds <= 0:
            return

        now = time.perf_counter()
        pages_since = len(result.pages) - self._last_checkpoint_pages
        time_since = now - self._last_checkpoint_time

        should_save = False
        if self.checkpoint_every_pages > 0 and pages_since >= self.checkpoint_every_pages:
            should_save = True
        if self.checkpoint_every_seconds > 0 and time_since >= self.checkpoint_every_seconds:
            should_save = True

        if not should_save:
            return

        await self._save_checkpoint(
            checkpoint_file,
            result,
            queue,
            visited,
            elapsed_offset,
            started,
        )
        self._last_checkpoint_time = now
        self._last_checkpoint_pages = len(result.pages)

    async def _save_checkpoint(
        self,
        checkpoint_file: Path,
        result: CrawlResult,
        queue: Queue,
        visited: Set[str],
        elapsed_offset: float,
        started: float,
    ):
        if checkpoint_file is None:
            return

        elapsed_s = elapsed_offset + (time.perf_counter() - started)
        pending = list(getattr(queue, "_queue", []))

        state = {
            "version": 1,
            "start_url": result.start_url,
            "elapsed_s": round(elapsed_s, 2),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "visited": sorted(visited),
            "queue": [[item[0], item[1]] for item in pending],
            "pages": [self._page_to_dict(page) for page in result.pages],
            "failed_urls": list(result.failed_urls),
            "skipped_urls": list(result.skipped_urls),
        }

        await asyncio.to_thread(self._write_checkpoint, checkpoint_file, state)

    def _write_checkpoint(self, checkpoint_file: Path, state: dict):
        checkpoint_file.write_text(
            json.dumps(state, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    async def _load_checkpoint(self, checkpoint_file: Path) -> Optional[dict]:
        if not checkpoint_file.exists():
            return None

        try:
            raw = await asyncio.to_thread(checkpoint_file.read_text, encoding="utf-8")
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except Exception as exc:
            logger.warning(f"Failed to load checkpoint {checkpoint_file}: {exc}")
            return None

    async def _clear_checkpoint(self, checkpoint_file: Path):
        if not checkpoint_file.exists():
            return

        try:
            await asyncio.to_thread(checkpoint_file.unlink)
        except Exception as exc:
            logger.warning(f"Failed to remove checkpoint {checkpoint_file}: {exc}")

    def _checkpoint_to_result(self, state: dict) -> CrawlResult:
        start_url = normalize_url(state.get("start_url", ""))
        pages = [self._page_from_dict(page) for page in state.get("pages", [])]
        return CrawlResult(
            start_url=start_url,
            pages=pages,
            failed_urls=state.get("failed_urls", []),
            skipped_urls=state.get("skipped_urls", []),
        )

    def _page_to_dict(self, page: ParsedPage) -> dict:
        return {
            "url": page.url,
            "title": page.title,
            "text": page.text,
            "links": list(page.links),
            "word_count": page.word_count,
        }

    def _page_from_dict(self, data: dict) -> ParsedPage:
        return ParsedPage(
            url=data.get("url", ""),
            title=data.get("title", ""),
            text=data.get("text", ""),
            links=data.get("links", []),
            word_count=int(data.get("word_count", 0) or 0),
        )
