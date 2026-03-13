from typing import List

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """
    Generates embeddings using a sentence-transformers model.

    The underlying model is loaded lazily on the first call to embed()
    so that importing this module doesn't trigger a slow model download.
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        device: str | None = None,
    ):
        emb_cfg = config.get("embeddings", {})
        self.model_name: str = model_name or emb_cfg.get("model_name", "all-MiniLM-L6-v2")
        self.batch_size: int = batch_size if batch_size is not None else emb_cfg.get("batch_size", 32)
        self.device: str = device or emb_cfg.get("device", "cpu")
        self._model = None  # Loaded lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        if not texts:
            return []

        model = self._get_model()

        total = len(texts)
        all_embeddings: List[List[float]] = []

        for batch_start in range(0, total, self.batch_size):
            batch = texts[batch_start: batch_start + self.batch_size]
            batch_end = min(batch_start + self.batch_size, total)

            logger.debug(f"Embedding batch {batch_start + 1}–{batch_end} of {total}")
            vectors = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            all_embeddings.extend(vectors.tolist())

        logger.debug(f"Generated {len(all_embeddings)} embeddings with {self.model_name}")
        return all_embeddings

    def embed_one(self, text: str) -> List[float]:
        """
        Embed a single string. Convenience wrapper around embed().

        Args:
            text: String to embed.

        Returns:
            Single embedding vector as a list of floats.
        """
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (requires model to be loaded)."""
        return self._get_model().get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Load model on first access (lazy initialisation)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name} (device={self.device})")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Embedding model loaded — dimension={self._model.get_sentence_embedding_dimension()}")
        return self._model
