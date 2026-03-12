"""
retrieval/vector_store.py — Qdrant vector store with dense & sparse indexing.

Swap strategy: implement the `VectorStore` Protocol for any backend
(Chroma, Pinecone, FAISS, Weaviate) without changing retrieval code.
"""

from __future__ import annotations
import uuid
from typing import Protocol, runtime_checkable

import numpy as np

from config import cfg
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocol — every backend must satisfy this
# ---------------------------------------------------------------------------
@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunks: list[dict], embeddings: np.ndarray) -> None: ...
    def dense_search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]: ...
    def count(self) -> int: ...


# ---------------------------------------------------------------------------
# Qdrant implementation
# ---------------------------------------------------------------------------
class QdrantStore:
    """
    Stores dense vectors in Qdrant.
    Connects to a local Qdrant instance by default (docker or in-memory).
    Set QDRANT_API_KEY env var + a cloud URL for Qdrant Cloud.
    """

    def __init__(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.collection = cfg.retrieval.collection_name
        self._dim: int | None = None

        # Local (no auth) vs Cloud (API key)
        if cfg.retrieval.qdrant_api_key:
            self.client = QdrantClient(
                url=cfg.retrieval.qdrant_url,
                api_key=cfg.retrieval.qdrant_api_key,
            )
        else:
            try:
                self.client = QdrantClient(url=cfg.retrieval.qdrant_url)
                self.client.get_collections()  # ping
                log.info(f"Connected to Qdrant at {cfg.retrieval.qdrant_url}")
            except Exception:
                log.warning("Qdrant server unavailable — using in-memory mode")
                self.client = QdrantClient(":memory:")

        self._Distance = Distance
        self._VectorParams = VectorParams

    def _ensure_collection(self, dim: int):
        from qdrant_client.models import Distance, VectorParams
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            log.info(f"Created Qdrant collection '{self.collection}' (dim={dim})")

    def upsert(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        from qdrant_client.models import PointStruct

        dim = embeddings.shape[1]
        self._ensure_collection(dim)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={
                    "sentence_chunk": chunk["sentence_chunk"],
                    "page_number": chunk["page_number"],
                    "chunk_idx": idx,
                },
            )
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]

        batch_size = 256
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=points[i : i + batch_size],
            )
        log.info(f"Upserted {len(points)} vectors into '{self.collection}'")

    def dense_search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            with_payload=True,
        )
        return [
            {**hit.payload, "score": hit.score, "id": hit.id}
            for hit in hits
        ]

    def count(self) -> int:
        try:
            return self.client.count(self.collection).count
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_BACKENDS: dict[str, type] = {
    "qdrant": QdrantStore,
}


def get_vector_store() -> VectorStore:
    backend = cfg.retrieval.vector_store_backend
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown vector store '{backend}'. Add it to _BACKENDS in vector_store.py")
    return _BACKENDS[backend]()
