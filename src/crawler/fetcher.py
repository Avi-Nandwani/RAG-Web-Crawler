import asyncio
import time
from typing import Optional
from dataclasses import dataclass

import aiohttp
from aiohttp import ClientResponse, ClientSession
from aiohttp.client_exceptions import ClientError

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FetchResult:
    """Structured result returned by Fetcher.fetch()."""

    url: str
    original_url: str
    html: Optional[str] = None
    text: Optional[str] = None
    status_code: Optional[int] = None
    content_type: str = ""
    ok: bool = False
    error: Optional[str] = None


class Fetcher:
    """
    Async wrapper around aiohttp that adds:
    - Consistent User-Agent header
    - Configurable timeout
    - Automatic retry with backoff on transient errors
    - Only accepts text/html responses
    """

    RETRYABLE_CODES = {429, 500, 502, 503, 504}
    BACKOFF_FACTOR = 1.5

    def __init__(
        self,
        user_agent: str,
        timeout: int = 10,
        max_retries: int = 3,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self._session: Optional[ClientSession] = None

    async def get_session(self) -> ClientSession:
        """Lazy-initialize the aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": self.user_agent}
            )
        return self._session

    async def close(self):
        """Close the underlying aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch(self, url: str) -> FetchResult:
        """
        Asynchronously fetch a URL and return a FetchResult.

        - Retries on transient server errors (5xx, 429).
        - Returns ok=False for non-HTML content types, 4xx errors, or exceptions.
        """
        last_error: Optional[str] = None
        wait = self.BACKOFF_FACTOR
        session = await self.get_session()

        total_attempts = self.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            try:
                async with session.get(
                    url, timeout=self.timeout, allow_redirects=True
                ) as response:
                    result = await self._process_response(url, response)

                    if result.ok:
                        return result

                    if (
                        response.status in self.RETRYABLE_CODES
                        and attempt < total_attempts
                    ):
                        logger.debug(
                            f"Retrying {url} (attempt {attempt}/{self.max_retries}) "
                            f"— status {response.status}"
                        )
                        await asyncio.sleep(wait)
                        wait *= self.BACKOFF_FACTOR
                        continue

                    return result  # Final failure (non-retryable)

            except (ClientError, asyncio.TimeoutError) as exc:
                last_error = str(exc)
                if attempt < total_attempts:
                    logger.debug(
                        f"Network error fetching {url} (attempt {attempt}/{self.max_retries}): {exc}"
                    )
                    await asyncio.sleep(wait)
                    wait *= self.BACKOFF_FACTOR

        logger.warning(
            f"Failed to fetch {url} after {self.max_retries} attempts: {last_error}"
        )
        return FetchResult(
            url=url,
            original_url=url,
            ok=False,
            error=last_error or "Max retries exceeded",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _process_response(
        self, original_url: str, response: ClientResponse
    ) -> FetchResult:
        """Turn an aiohttp ClientResponse into a FetchResult."""
        content_type = response.headers.get("Content-Type", "")
        content_type_lower = content_type.lower()
        final_url = str(response.url)  # After redirects

        if response.status == 200:
            if "text/html" in content_type_lower:
                html = await response.text()
                return FetchResult(
                    url=final_url,
                    original_url=original_url,
                    html=html,
                    status_code=response.status,
                    content_type=content_type,
                    ok=True,
                )

            if content_type_lower.startswith("text/"):
                text = await response.text()
                return FetchResult(
                    url=final_url,
                    original_url=original_url,
                    text=text,
                    status_code=response.status,
                    content_type=content_type,
                    ok=True,
                )

            return FetchResult(
                url=final_url,
                original_url=original_url,
                status_code=response.status,
                content_type=content_type,
                ok=False,
                error=f"Unsupported content type: {content_type}",
            )

        # Non-200 response
        return FetchResult(
            url=final_url,
            original_url=original_url,
            status_code=response.status,
            content_type=content_type,
            ok=False,
            error=f"HTTP {response.status}",
        )
