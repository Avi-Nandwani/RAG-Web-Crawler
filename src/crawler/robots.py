import asyncio
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from typing import Optional

import aiohttp
from src.utils.logger import get_logger
from src.utils.helpers import get_domain

logger = get_logger(__name__)


class RobotsCache:
    """
    Async-aware cache for robots.txt files.

    - Fetches and parses robots.txt for a domain only once.
    - Caches the parsed RobotFileParser object in memory.
    - Handles network errors gracefully (fail-open: assumes allowed if fetch fails).
    - Uses aiohttp for non-blocking network requests.
    """

    def __init__(self, user_agent: str, timeout: int = 10):
        self.user_agent = user_agent
        self.timeout = timeout
        self._cache: dict[tuple[str, str], RobotFileParser] = {}
        self._session: Optional[aiohttp.ClientSession] = None

    async def get_session(self) -> aiohttp.ClientSession:
        """Lazy-initialize the aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": self.user_agent}
            )
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def can_fetch(self, url: str) -> bool:
        """
        Check if the user agent is allowed to fetch the given URL.

        Args:
            url: The URL to check.

        Returns:
            True if fetching is allowed, False otherwise.
        """
        parser = await self._get_parser(url)
        return parser.can_fetch(self.user_agent, url)

    async def get_crawl_delay(self, url: str) -> Optional[float]:
        """
        Get the Crawl-delay for the user agent from robots.txt.

        Args:
            url: The URL to check.

        Returns:
            The delay in seconds, or None if not specified.
        """
        parser = await self._get_parser(url)
        delay = parser.crawl_delay(self.user_agent)
        if delay is None:
            delay = parser.crawl_delay("*")
        return float(delay) if delay is not None else None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _get_parser(self, url: str) -> RobotFileParser:
        """
        Retrieve the parsed robots.txt for a URL's domain from cache or by fetching.
        """
        parsed = urlparse(url)
        scheme = parsed.scheme or "http"
        domain = get_domain(url)
        key = (scheme, domain)
        if key not in self._cache:
            self._cache[key] = await self._fetch_and_parse(domain, scheme)
        return self._cache[key]

    async def _fetch_and_parse(self, domain: str, scheme: str) -> RobotFileParser:
        """
        Fetch and parse the robots.txt file for a given domain.
        Returns a parser that allows all access if fetching fails.
        """
        robots_url = f"{scheme}://{domain}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)

        try:
            session = await self.get_session()
            async with session.get(robots_url, timeout=self.timeout) as response:
                if response.status == 200:
                    text = await response.text()
                    parser.parse(text.splitlines())
                    logger.debug(f"Fetched and parsed robots.txt for {domain}")
                else:
                    logger.debug(
                        f"Failed to fetch robots.txt for {domain} (status: {response.status}), "
                        "assuming allowed."
                    )
                    parser.allow_all = True
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning(
                f"Network error fetching robots.txt for {domain}: {exc}. Assuming allowed."
            )
            parser.allow_all = True
        except Exception as exc:
            logger.error(f"Unexpected error parsing robots.txt for {domain}: {exc}")
            parser.allow_all = True

        return parser
