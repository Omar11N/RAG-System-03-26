"""
retrieval/hybrid_search.py — Hybrid retrieval: BM25 (sparse) + dense (BGE-M3).

Uses Reciprocal Rank Fusion (RRF) to merge result lists without needing
score normalization. Set cfg.retrieval.enable_hybrid_search = False to
fall back to dense-only retrieval.
"""

from __future__ import annotations
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from config import cfg
from retrieval.embedder import get_embedder
from retrieval.vector_store import VectorStore
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# BM25 index (in-memory, rebuilt from chunks each session)
# ---------------------------------------------------------------------------
class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized = [c["sentence_chunk"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        log.info(f"BM25 index built over {len(chunks)} chunks")

    def search(self, query: str, top_k: int) -> list[dict]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {**self.chunks[i], "bm25_score": float(scores[i]), "chunk_idx": i}
            for i in top_indices
            if scores[i] > 0
        ]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    id_key: str = "chunk_idx",
    k: int = 60,
) -> list[dict]:
    """
    Merge multiple ranked lists via RRF.
    Each doc's score = Σ 1/(k + rank_in_list).
    """
    rrf_scores: dict[int, float] = defaultdict(float)
    doc_lookup: dict[int, dict] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            doc_id = doc[id_key]
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            doc_lookup[doc_id] = doc

    sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
    return [
        {**doc_lookup[doc_id], "rrf_score": rrf_scores[doc_id]}
        for doc_id in sorted_ids
    ]


# ---------------------------------------------------------------------------
# Main retriever
# ---------------------------------------------------------------------------
class HybridRetriever:
    def __init__(self, chunks: list[dict], vector_store: VectorStore):
        self.chunks = chunks
        self.vector_store = vector_store
        self.embedder = get_embedder()
        self._bm25: BM25Index | None = None

        if cfg.retrieval.enable_hybrid_search:
            self._bm25 = BM25Index(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or cfg.retrieval.top_k_final
        k_dense = cfg.retrieval.top_k_dense
        k_sparse = cfg.retrieval.top_k_sparse

        query_emb = self.embedder.encode(query)[0]

        # Dense retrieval
        dense_results = self.vector_store.dense_search(query_emb, top_k=k_dense)
        for i, r in enumerate(dense_results):
            r.setdefault("chunk_idx", r.get("chunk_idx", i))

        if not cfg.retrieval.enable_hybrid_search or self._bm25 is None:
            log.debug(f"Dense-only retrieval → {len(dense_results[:top_k])} docs")
            return dense_results[:top_k]

        # Sparse retrieval (BM25)
        sparse_results = self._bm25.search(query, top_k=k_sparse)

        # RRF fusion
        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            id_key="chunk_idx",
            k=cfg.retrieval.rrf_k,
        )
        log.debug(
            f"Hybrid retrieval: {len(dense_results)} dense + "
            f"{len(sparse_results)} sparse → {len(fused[:top_k])} fused"
        )
        return fused[:top_k]
