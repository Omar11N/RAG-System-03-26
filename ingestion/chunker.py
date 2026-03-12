"""
ingestion/chunker.py — Sentence-aware chunking with configurable overlap.

Each chunk carries its source page number for provenance tracking.
"""

from __future__ import annotations
import re

from tqdm.auto import tqdm

from config import cfg
from utils.logger import get_logger

log = get_logger(__name__)


def _load_sentencizer():
    """Lazy-load spaCy sentencizer (no full model needed)."""
    try:
        from spacy.lang.en import English
        nlp = English()
        nlp.add_pipe("sentencizer")
        return nlp
    except ImportError:
        return None


def _sentence_split_spacy(text: str, nlp) -> list[str]:
    doc = nlp(text)
    return [str(s).strip() for s in doc.sents if str(s).strip()]


def _sentence_split_simple(text: str) -> list[str]:
    """Regex fallback if spaCy is unavailable."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def split_sentences(pages_and_texts: list[dict]) -> list[dict]:
    """Add 'sentences' list to every page dict."""
    nlp = _load_sentencizer()
    method = "spaCy" if nlp else "regex"
    log.info(f"Splitting sentences with {method}")

    for item in tqdm(pages_and_texts, desc="Splitting sentences"):
        if nlp:
            item["sentences"] = _sentence_split_spacy(item["text"], nlp)
        else:
            item["sentences"] = _sentence_split_simple(item["text"])
    return pages_and_texts


def create_chunks(pages_and_texts: list[dict]) -> list[dict]:
    """
    Slide a window of `sentences_per_chunk` sentences with `chunk_overlap`
    overlap across each page, producing a flat list of chunk dicts.
    """
    n = cfg.ingestion.sentences_per_chunk
    overlap = cfg.ingestion.chunk_overlap
    min_tokens = cfg.ingestion.min_token_length
    stride = max(1, n - overlap)

    pages_and_chunks: list[dict] = []

    for item in tqdm(pages_and_texts, desc="Creating chunks"):
        sentences = item.get("sentences", [])
        for i in range(0, len(sentences), stride):
            chunk_sentences = sentences[i : i + n]
            chunk_text = re.sub(r'\.([A-Z])', r'. \1', " ".join(chunk_sentences))
            chunk_text = chunk_text.replace("  ", " ").strip()

            # Filter very short chunks — likely headers / noise
            if len(chunk_text) / 4 > min_tokens:
                pages_and_chunks.append({
                    "page_number": item["page_number"],
                    "sentence_chunk": chunk_text,
                    "chunk_char_count": len(chunk_text),
                    "chunk_word_count": len(chunk_text.split()),
                    "chunk_token_estimate": len(chunk_text) / 4,
                })

    log.info(f"Created {len(pages_and_chunks)} chunks from {len(pages_and_texts)} pages")
    return pages_and_chunks
