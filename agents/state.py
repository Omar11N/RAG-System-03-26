"""
agents/state.py — LangGraph state schema for the agentic RAG loop.

Every node reads from and writes to this shared state dict.
Adding a new field here is all that's needed to pass new data between nodes.
"""

from __future__ import annotations
from typing import TypedDict, Optional


class RAGState(TypedDict):
    # Input
    query: str
    original_query: str          # preserved across rewrites

    # Retrieval
    retrieved_chunks: list[dict]
    context_quality_score: float
    context_quality_reason: str

    # Generation
    answer: str

    # Control flow
    retry_count: int
    should_rewrite: bool

    # Evaluation (post-generation)
    ragas_scores: Optional[dict]
    final: bool
