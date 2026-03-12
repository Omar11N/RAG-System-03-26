"""
retrieval/embedder.py — Dense embedding model wrapper.

Swap strategy: change cfg.retrieval.embedding_model in config.py.
Supports any sentence-transformers compatible model or OpenAI embeddings.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Union

from config import cfg
from utils.logger import get_logger

log = get_logger(__name__)


class EmbeddingModel:
    """
    Wraps sentence-transformers (BGE-M3, all-mpnet, etc.) or
    OpenAI's embedding API behind a single `.encode()` interface.
    """

    def __init__(self):
        self.model_name = cfg.retrieval.embedding_model
        self.device = cfg.retrieval.embedding_device
        self._model = None
        self._openai_client = None

    def _load(self):
        if self._model is not None or self._openai_client is not None:
            return

        if self.model_name.startswith("text-embedding"):
            # OpenAI embeddings
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=cfg.retrieval.embedding_model)
            log.info(f"Loaded OpenAI embedding model: {self.model_name}")
        else:
            # HuggingFace / sentence-transformers
            from sentence_transformers import SentenceTransformer
            log.info(f"Loading sentence-transformer: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            log.info("Embedding model loaded")

    def encode(
        self,
        texts: Union[str, list[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode text(s) → numpy float32 array of shape (N, dim)."""
        self._load()
        if isinstance(texts, str):
            texts = [texts]

        if self._openai_client:
            response = self._openai_client.embeddings.create(
                input=texts,
                model=self.model_name,
            )
            return np.array([d.embedding for d in response.data], dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # BGE-M3 needs normalized vectors
        )
        return embeddings.astype(np.float32)

    @property
    def dimension(self) -> int:
        self._load()
        if self._model:
            return self._model.get_sentence_embedding_dimension()
        return 3072  # text-embedding-3-large default


# Module-level singleton
_embedder: EmbeddingModel | None = None


def get_embedder() -> EmbeddingModel:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingModel()
    return _embedder
