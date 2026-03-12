"""
main.py — Entry point for the Modern Agentic RAG system.

Usage:
    python main.py --pdf path/to/document.pdf
    python main.py --pdf path/to/document.pdf --provider anthropic
    python main.py --pdf path/to/document.pdf --skip-ingestion  # reuse cached index
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from config import cfg
from utils.logger import get_logger

log = get_logger("main")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Modern Agentic RAG System")
    parser.add_argument("--pdf", type=str, default=cfg.ingestion.pdf_path, help="Path to PDF file")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini", "local_hf", "ollama"],
        default=cfg.generation.provider,
    )
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip PDF parsing; use cached index")
    parser.add_argument("--no-eval", action="store_true", help="Disable RAGAs evaluation")
    parser.add_argument("--top-k", type=int, default=cfg.retrieval.top_k_final)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------
def run_ingestion(pdf_path: str) -> tuple:
    """Parse → chunk → embed → upsert. Returns (chunks, vector_store)."""
    from ingestion.pdf_parser import get_parser
    from ingestion.chunker import split_sentences, create_chunks
    from retrieval.embedder import get_embedder
    from retrieval.vector_store import get_vector_store

    store = get_vector_store()

    # Skip if already indexed
    if store.count() > 0:
        log.info(f"Vector store already has {store.count()} vectors — skipping re-ingestion")
        # We still need the chunks for BM25; reload from a quick re-parse without embedding
        parser = get_parser("pymupdf")  # fast fallback for chunk reload
        pages = parser.parse(pdf_path)
        pages = split_sentences(pages)
        chunks = create_chunks(pages)
        return chunks, store

    log.info("=== INGESTION PIPELINE ===")
    parser = get_parser("unstructured")
    pages = parser.parse(pdf_path)
    pages = split_sentences(pages)
    chunks = create_chunks(pages)

    embedder = get_embedder()
    log.info(f"Embedding {len(chunks)} chunks with {cfg.retrieval.embedding_model}...")
    texts = [c["sentence_chunk"] for c in chunks]
    embeddings = embedder.encode(texts, batch_size=32, show_progress=True)

    store.upsert(chunks, embeddings)
    return chunks, store


# ---------------------------------------------------------------------------
# Optional GraphRAG build
# ---------------------------------------------------------------------------
def build_graph_index(chunks: list[dict], llm):
    from ingestion.graph_indexer import KnowledgeGraphIndex
    index = KnowledgeGraphIndex()
    index.build(chunks, llm_client=llm)
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Override config from CLI
    cfg.ingestion.pdf_path = args.pdf
    cfg.generation.provider = args.provider
    cfg.retrieval.top_k_final = args.top_k
    if args.no_eval:
        cfg.evaluation.enable_ragas = False

    if not Path(args.pdf).exists() and not args.skip_ingestion:
        log.error(f"PDF not found: {args.pdf}")
        sys.exit(1)

    os.makedirs(cfg.cache_dir, exist_ok=True)

    # ── 1. LLM ────────────────────────────────────────────────────────────
    from generation.llm_factory import get_llm
    llm = get_llm()

    # ── 2. Ingestion ──────────────────────────────────────────────────────
    if not args.skip_ingestion:
        chunks, vector_store = run_ingestion(args.pdf)
    else:
        from retrieval.vector_store import get_vector_store
        vector_store = get_vector_store()
        chunks = []  # BM25 disabled when skipping
        log.info("Skipping ingestion — using existing vector index")

    # ── 3. GraphRAG (optional) ────────────────────────────────────────────
    if cfg.ingestion.enable_graph_index and chunks:
        build_graph_index(chunks, llm)

    # ── 4. Hybrid retriever ───────────────────────────────────────────────
    from retrieval.hybrid_search import HybridRetriever
    retriever = HybridRetriever(chunks, vector_store)

    # ── 5. Compile LangGraph pipeline ─────────────────────────────────────
    from agents.graph import build_rag_graph, run_query
    graph = build_rag_graph(retriever=retriever, llm=llm)

    # ── 6. Interactive query loop ──────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  Modern Agentic RAG — ready for queries")
    print("  Type 'exit' or 'quit' to stop")
    print("═" * 70 + "\n")

    while True:
        try:
            query = input("❓ Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        if not query:
            continue

        result = run_query(graph, query)

        print(f"\n{'─' * 70}")
        print(f"📄 Answer (retry #{result['retry_count']}):\n")
        print(result["answer"])

        if result.get("context_quality_score") is not None:
            print(f"\n📊 Context quality: {result['context_quality_score']:.2f} — {result['context_quality_reason']}")

        if result.get("ragas_scores"):
            print(f"\n🔬 RAGAs scores: {result['ragas_scores']}")

        print(f"{'─' * 70}\n")


if __name__ == "__main__":
    main()
