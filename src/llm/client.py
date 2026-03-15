from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Normalized response returned by OllamaClient.generate()."""

    text: str
    model: str
    raw: Dict[str, Any] = field(default_factory=dict)


class OllamaClient:
    """
    Thin wrapper around Ollama chat API.

    Supports dependency injection of an existing client for tests.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        client: Any = None,
    ):
        llm_cfg = config.get("llm", {})
        self.model_name = model_name or llm_cfg.get("model_name", "llama3.2:3b")
        self.base_url = base_url or llm_cfg.get("base_url", "http://localhost:11434")
        self.temperature = temperature if temperature is not None else llm_cfg.get("temperature", 0.1)
        self.max_tokens = max_tokens if max_tokens is not None else llm_cfg.get("max_tokens", 512)
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else llm_cfg.get("timeout_seconds", 30)
        )
        self._client = client

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """
        Generate a completion using Ollama chat endpoint.

        Raises:
            RuntimeError: if Ollama call fails.
        """
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
        except Exception as exc:
            logger.error(f"Ollama generation failed: {exc}")
            raise RuntimeError(f"Ollama generation failed: {exc}") from exc

        text = (
            response.get("message", {}).get("content", "")
            if isinstance(response, dict)
            else ""
        )

        return LLMResponse(
            text=text.strip(),
            model=self.model_name,
            raw=response if isinstance(response, dict) else {},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is not None:
            return self._client

        import ollama

        # ollama.Client(host=...) is supported in the installed SDK.
        self._client = ollama.Client(host=self.base_url)
        return self._client
