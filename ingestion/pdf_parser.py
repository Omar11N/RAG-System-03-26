"""
ingestion/pdf_parser.py — PDF parsing via unstructured.io.

Swap strategy: replace `_parse_with_unstructured` with any other parser
(PyMuPDF, pdfplumber, Docling) without changing downstream code.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Protocol

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocol — any parser must satisfy this interface
# ---------------------------------------------------------------------------
class PDFParser(Protocol):
    def parse(self, pdf_path: str) -> list[dict]:
        """Returns list of {page_number, text} dicts."""
        ...


# ---------------------------------------------------------------------------
# Unstructured.io parser (default)
# ---------------------------------------------------------------------------
class UnstructuredParser:
    """
    Uses `unstructured` library for high-fidelity PDF parsing.
    Handles multi-column layouts, tables, and embedded images better than
    naive text extraction.
    """

    def parse(self, pdf_path: str) -> list[dict]:
        log.info(f"Parsing PDF with unstructured.io: {pdf_path}")
        try:
            from unstructured.partition.pdf import partition_pdf
            elements = partition_pdf(
                filename=pdf_path,
                strategy="hi_res",          # best quality; use "fast" for speed
                infer_table_structure=True,  # extract tables as structured text
                include_page_breaks=True,
            )
        except ImportError:
            log.warning("unstructured not installed — falling back to PyMuPDF")
            return PyMuPDFParser().parse(pdf_path)

        pages: dict[int, list[str]] = {}
        for el in elements:
            page_num = el.metadata.page_number or 0
            pages.setdefault(page_num, []).append(str(el).strip())

        result = [
            {"page_number": page, "text": " ".join(texts)}
            for page, texts in sorted(pages.items())
            if " ".join(texts).strip()
        ]
        log.info(f"Parsed {len(result)} pages from {Path(pdf_path).name}")
        return result


# ---------------------------------------------------------------------------
# PyMuPDF fallback parser
# ---------------------------------------------------------------------------
class PyMuPDFParser:
    """Lightweight fallback using PyMuPDF (fitz)."""

    def parse(self, pdf_path: str) -> list[dict]:
        import fitz
        from tqdm.auto import tqdm

        log.info(f"Parsing PDF with PyMuPDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        result = []
        for page_number, page in tqdm(enumerate(doc), desc="Reading pages", total=len(doc)):
            text = page.get_text().replace("\n", " ").strip()
            if text:
                result.append({"page_number": page_number, "text": text})
        log.info(f"Parsed {len(result)} pages")
        return result


# ---------------------------------------------------------------------------
# Factory — resolves parser by name
# ---------------------------------------------------------------------------
PARSERS: dict[str, type] = {
    "unstructured": UnstructuredParser,
    "pymupdf": PyMuPDFParser,
}


def get_parser(name: str = "unstructured") -> PDFParser:
    if name not in PARSERS:
        raise ValueError(f"Unknown parser '{name}'. Choose from: {list(PARSERS)}")
    return PARSERS[name]()
