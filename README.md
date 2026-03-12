# 🧠 Modern Agentic RAG System

> A production-grade, fully modular Retrieval-Augmented Generation pipeline with hybrid search, LangGraph agentic loops, GraphRAG context, and RAGAs evaluation.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C3C3C?style=flat-square)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC244C?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## Architecture

```
PDF → Unstructured.io (parse)
    → spaCy sentence splitting + chunking with overlap
    → GraphRAG knowledge graph (entity/relationship extraction)
         ↓
  BAAI/bge-m3 embeddings → Qdrant vector store
         ↓
  LangGraph Agentic Loop:
    retrieve (hybrid: BM25 + dense, RRF fusion)
      → evaluate context quality
        → [re-query with rewritten query if quality < threshold]
      → generate answer
      → RAGAs evaluation (faithfulness, relevancy, precision, recall)
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/your-username/modern-rag
cd modern-rag
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Start Qdrant (Docker)

```bash
docker run -p 6333:6333 qdrant/qdrant
# Or use in-memory mode (no setup needed — auto-fallback)
```

### 4. Run

```bash
python main.py --pdf your_document.pdf --provider anthropic
```

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--pdf` | `document.pdf` | Path to input PDF |
| `--provider` | `anthropic` | LLM: `openai`, `anthropic`, `gemini`, `local_hf`, `ollama` |
| `--skip-ingestion` | off | Reuse existing Qdrant index |
| `--no-eval` | off | Disable RAGAs scoring |
| `--top-k` | `5` | Number of chunks to retrieve |

---

## 🔧 Swapping Components

Every major component is behind a config flag or factory. Change one line — nothing else breaks.

### Switch LLM provider

```python
# config.py
cfg.generation.provider = "gemini"       # or "openai" | "local_hf" | "ollama"
cfg.generation.gemini_model = "gemini-2.0-flash"
```

Or via CLI: `python main.py --pdf doc.pdf --provider ollama`

### Switch embedding model

```python
# config.py
cfg.retrieval.embedding_model = "BAAI/bge-m3"        # best open-source
# cfg.retrieval.embedding_model = "all-mpnet-base-v2" # lighter weight
# cfg.retrieval.embedding_model = "text-embedding-3-large"  # OpenAI
```

**Note:** Changing the embedding model requires re-ingesting (delete `.cache/` and restart).

### Switch vector store

```python
# config.py
cfg.retrieval.vector_store_backend = "qdrant"   # currently supported
```

To add a new backend (e.g. Chroma), implement the `VectorStore` protocol in `retrieval/vector_store.py` and add it to `_BACKENDS`.

### Switch PDF parser

```python
# ingestion/pdf_parser.py — get_parser()
parser = get_parser("unstructured")   # high-quality, handles tables/columns
parser = get_parser("pymupdf")        # fast, lightweight fallback
```

### Disable hybrid search (dense-only)

```python
cfg.retrieval.enable_hybrid_search = False
```

### Disable GraphRAG

```python
cfg.ingestion.enable_graph_index = False
```

### Tune the agentic loop

```python
cfg.agent.max_retries = 3              # max re-retrieval attempts
cfg.agent.quality_score_threshold = 0.6  # re-query if score below this
```

### Change RAGAs metrics

```python
cfg.evaluation.metrics = ["faithfulness", "answer_relevancy"]
cfg.evaluation.ragas_llm_provider = "openai"   # LLM used as judge
```

---

## Project Structure

```
modern-rag/
├── config.py                   # 🎛️  All configuration — start here
├── main.py                     # 🚀  Entry point & CLI
├── requirements.txt
├── .env.example
│
├── ingestion/
│   ├── pdf_parser.py           # PDF → pages (unstructured / PyMuPDF)
│   ├── chunker.py              # pages → overlapping sentence chunks
│   └── graph_indexer.py        # chunks → knowledge graph (GraphRAG)
│
├── retrieval/
│   ├── embedder.py             # text → dense vectors (BGE-M3 / OpenAI)
│   ├── vector_store.py         # Qdrant upsert & search
│   └── hybrid_search.py        # BM25 + dense, RRF fusion
│
├── agents/
│   ├── state.py                # LangGraph state schema
│   ├── nodes.py                # retrieve / evaluate / rewrite / generate
│   └── graph.py                # graph wiring & conditional edges
│
├── generation/
│   ├── llm_factory.py          # OpenAI / Anthropic / Gemini / HF / Ollama
│   └── prompt_templates.py     # all prompts in one place
│
├── evaluation/
│   └── ragas_eval.py           # faithfulness, relevancy, precision, recall
│
└── utils/
    └── logger.py               # structured logging
```

---

## Scaling Guide

### Scaling ingestion (large document collections)

- **Batch embedding:** `embedder.encode()` already batches at 32. Increase `batch_size` for large GPUs.
- **Parallel parsing:** Wrap `get_parser().parse()` in a `ProcessPoolExecutor` for multi-PDF ingestion.
- **Persistent index:** Qdrant persists to disk automatically. For cloud scale, use [Qdrant Cloud](https://cloud.qdrant.io).

### Scaling retrieval

- **Qdrant Cloud:** Set `QDRANT_URL` and `QDRANT_API_KEY` in `.env` — no code change needed.
- **Increase `top_k_dense` / `top_k_sparse`** for higher recall at the cost of more LLM context.
- **Add reranking:** Insert a cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) after RRF fusion in `hybrid_search.py`.

### Scaling generation

- **Switch to API providers** (`anthropic`, `openai`, `gemini`) for unlimited concurrency.
- **Ollama** for fully local, no-cost inference on a local GPU server.
- For **high-throughput batch Q&A**, wrap `run_query()` with `asyncio` — LangGraph supports async graphs.

### Adding a new LLM provider

1. Add a class with a `generate(prompt: str) -> str` method in `generation/llm_factory.py`
2. Add it to `_PROVIDERS` dict
3. Set `cfg.generation.provider = "your_provider_name"`

---

## Requirements

- Python 3.10+
- Docker (for Qdrant local server, optional — in-memory fallback available)
- GPU recommended for local embedding / local HF models; CPU works for API-based setups

---

## License

MIT
