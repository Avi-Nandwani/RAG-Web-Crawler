import time
from typing import Optional
from dataclasses import dataclass, field

import requests
from requests import Response
from requests.exceptions import RequestException

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FetchResult:
    """Structured result returned by Fetcher.fetch()."""

    url: str                          # Final URL after any redirects
    original_url: str                 # URL that was requested
    html: Optional[str] = None        # Raw HTML content (None on failure)
    status_code: Optional[int] = None
    content_type: str = ""
    ok: bool = False                  # True only when HTML was successfully fetched
    error: Optional[str] = None       # Human-readable error message


class Fetcher:
    """
    Thin wrapper around requests that adds:
    - Consistent User-Agent header
    - Configurable timeout
    - Automatic retry with backoff on transient errors
    - Only accepts text/html responses (skips PDFs, images, etc.)
    """

    RETRYABLE_CODES = {429, 500, 502, 503, 504}
    BACKOFF_FACTOR = 1.5  # seconds between retry attempts (multiplied each time)

    def __init__(
        self,
        user_agent: str,
        timeout: int = 10,
        max_retries: int = 3,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, url: str) -> FetchResult:
        """
        Fetch a URL and return a FetchResult.

        - Retries on transient server errors (5xx, 429).
        - Returns ok=False for non-HTML content types, 4xx errors, or exceptions.

        Args:
            url: The URL to fetch.

        Returns:
            FetchResult with populated fields.
        """
        last_error: Optional[str] = None
        wait = self.BACKOFF_FACTOR

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
                result = self._process_response(url, response)

                if result.ok:
                    return result

                # Retry on transient server errors
                if response.status_code in self.RETRYABLE_CODES and attempt < self.max_retries:
                    logger.debug(
                        f"Retrying {url} (attempt {attempt}/{self.max_retries}) "
                        f"— status {response.status_code}"
                    )
                    time.sleep(wait)
                    wait *= self.BACKOFF_FACTOR
                    continue

                return result  # Final failure (non-retryable)

            except RequestException as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    logger.debug(
                        f"Network error fetching {url} (attempt {attempt}/{self.max_retries}): {exc}"
                    )
                    time.sleep(wait)
                    wait *= self.BACKOFF_FACTOR

        logger.warning(f"Failed to fetch {url} after {self.max_retries} attempts: {last_error}")
        return FetchResult(
            url=url,
            original_url=url,
            ok=False,
            error=last_error or "Max retries exceeded",
        )

    def close(self):
        """Close the underlying requests session."""
        self.session.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_response(self, original_url: str, response: Response) -> FetchResult:
        """Turn a requests Response into a FetchResult."""
        content_type = response.headers.get("Content-Type", "")
        final_url = response.url  # After redirects

        if response.status_code == 200:
            if "text/html" not in content_type:
                return FetchResult(
                    url=final_url,
                    original_url=original_url,
                    status_code=response.status_code,
                    content_type=content_type,
                    ok=False,
                    error=f"Non-HTML content type: {content_type}",
                )
            return FetchResult(
                url=final_url,
                original_url=original_url,
                html=response.text,
                status_code=response.status_code,
                content_type=content_type,
                ok=True,
            )

        # Non-200 response
        return FetchResult(
            url=final_url,
            original_url=original_url,
            status_code=response.status_code,
            content_type=content_type,
            ok=False,
            error=f"HTTP {response.status_code}",
        )
