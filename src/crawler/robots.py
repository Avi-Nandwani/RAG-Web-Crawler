"""
Robots.txt parser with per-domain caching.
Uses Python stdlib urllib.robotparser — no extra dependencies.
"""

from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from typing import Dict, Optional
import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RobotsCache:
    """
    Fetches and caches robots.txt rules per domain.
    One RobotFileParser instance is kept per domain so we don't
    re-download the same file for every URL.
    """

    def __init__(self, user_agent: str, timeout: int = 10):
        self.user_agent = user_agent
        self.timeout = timeout
        self._cache: Dict[str, RobotFileParser] = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_robots_url(self, url: str) -> str:
        """Return the robots.txt URL for the given page URL."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    def _domain_key(self, url: str) -> str:
        return urlparse(url).netloc.lower()

    def _fetch_parser(self, domain_key: str, robots_url: str) -> RobotFileParser:
        """Download robots.txt and return a populated RobotFileParser."""
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            resp = requests.get(
                robots_url,
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
            )
            if resp.status_code == 200:
                parser.parse(resp.text.splitlines())
                logger.debug(f"Loaded robots.txt for {domain_key}")
            elif resp.status_code == 404:
                # No robots.txt → allow everything
                parser.parse([])
                logger.debug(f"No robots.txt found for {domain_key} — allowing all")
            else:
                # Unexpected status → allow everything (fail open)
                parser.parse([])
                logger.warning(
                    f"robots.txt returned {resp.status_code} for {domain_key} — allowing all"
                )
        except Exception as exc:
            # Network / timeout error → allow everything
            parser.parse([])
            logger.warning(f"Could not fetch robots.txt for {domain_key}: {exc}")
        return parser

    def _get_parser(self, url: str) -> RobotFileParser:
        """Return cached (or freshly fetched) parser for the URL's domain."""
        key = self._domain_key(url)
        if key not in self._cache:
            robots_url = self._get_robots_url(url)
            self._cache[key] = self._fetch_parser(key, robots_url)
        return self._cache[key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_fetch(self, url: str) -> bool:
        """
        Return True if our user-agent is allowed to fetch this URL.

        Args:
            url: The page URL to check.

        Returns:
            True if crawling is allowed, False if disallowed.
        """
        try:
            parser = self._get_parser(url)
            allowed = parser.can_fetch(self.user_agent, url)
            if not allowed:
                logger.debug(f"robots.txt disallows: {url}")
            return allowed
        except Exception as exc:
            logger.warning(f"robots.txt check failed for {url}: {exc}")
            return True  # fail open

    def get_crawl_delay(self, url: str) -> Optional[float]:
        """
        Return the Crawl-delay directive for our user-agent, or None if not set.

        Args:
            url: Any URL on the target domain.

        Returns:
            Delay in seconds, or None.
        """
        try:
            parser = self._get_parser(url)
            return parser.crawl_delay(self.user_agent)
        except Exception:
            return None

    def clear_cache(self):
        """Clear all cached robots.txt entries."""
        self._cache.clear()
