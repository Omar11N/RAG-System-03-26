"""
config.py — Central configuration for the Modern RAG system.
Swap any component here without touching the rest of the codebase.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import os
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# 🗂️  INGESTION
# ---------------------------------------------------------------------------
@dataclass
class IngestionConfig:
    pdf_path: str = "document.pdf"
    # Chunking
    sentences_per_chunk: int = 10
    chunk_overlap: int = 2          # sentences shared between adjacent chunks
    min_token_length: int = 30
    # GraphRAG — entity extraction
    enable_graph_index: bool = True
    max_entities_per_chunk: int = 10


# ---------------------------------------------------------------------------
# 🔍  RETRIEVAL
# ---------------------------------------------------------------------------
@dataclass
class RetrievalConfig:
    # ── Embedding model (swap freely) ──────────────────────────────────────
    # Options: "BAAI/bge-m3" | "all-mpnet-base-v2" | "text-embedding-3-large"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"   # "cpu" | "cuda" | "mps"

    # ── Vector store ───────────────────────────────────────────────────────
    # Options: "qdrant" | "chroma" | "faiss"
    vector_store_backend: Literal["qdrant", "chroma", "faiss"] = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    collection_name: str = "rag_documents"

    # ── Hybrid search ──────────────────────────────────────────────────────
    enable_hybrid_search: bool = True  # dense + BM25 sparse fusion
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_final: int = 5             # after Reciprocal Rank Fusion
    rrf_k: int = 60                  # RRF constant


# ---------------------------------------------------------------------------
# 🤖  GENERATION
# ---------------------------------------------------------------------------
@dataclass
class GenerationConfig:
    # ── LLM provider (swap freely) ─────────────────────────────────────────
    # Options: "openai" | "anthropic" | "gemini" | "local_hf" | "ollama"
    provider: Literal["openai", "anthropic", "gemini", "local_hf", "ollama"] = "anthropic"

    # OpenAI
    openai_model: str = "gpt-4o"
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # Anthropic
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # Google Gemini
    gemini_model: str = "gemini-2.0-flash"
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    # Local HuggingFace
    local_hf_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    local_hf_load_in_4bit: bool = True

    # Ollama (local server)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # Shared generation params
    max_new_tokens: int = 1536
    temperature: float = 0.3


# ---------------------------------------------------------------------------
# 🔄  AGENTIC LOOP
# ---------------------------------------------------------------------------
@dataclass
class AgentConfig:
    max_retries: int = 3              # re-retrieval attempts before giving up
    quality_score_threshold: float = 0.6  # below this → re-query


# ---------------------------------------------------------------------------
# 📊  EVALUATION
# ---------------------------------------------------------------------------
@dataclass
class EvaluationConfig:
    enable_ragas: bool = True
    # RAGAs uses an LLM judge; point it at any supported provider
    ragas_llm_provider: Literal["openai", "anthropic", "gemini"] = "anthropic"
    ragas_llm_model: str = "claude-sonnet-4-20250514"
    # Metrics to run
    metrics: list[str] = field(default_factory=lambda: [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ])


# ---------------------------------------------------------------------------
# 🏗️  ROOT CONFIG
# ---------------------------------------------------------------------------
@dataclass
class AppConfig:
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    log_level: str = "INFO"
    cache_dir: str = ".cache"


# Singleton — import this everywhere
cfg = AppConfig()
