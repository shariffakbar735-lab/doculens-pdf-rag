"""
pdf_processor.py – PDF text extraction with per-page metadata.

Responsibilities:
  - Open and validate a PDF from raw bytes.
  - Extract text page-by-page, keeping track of page numbers (1-based).
  - Split the full text stream into overlapping chunks.
  - Attach filename and page number metadata to every chunk.

This module has no network calls and no Ollama dependency, so it is easy to
test in isolation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List

import pymupdf as fitz  # PyMuPDF (1.24+ preferred import)

from config import CHUNK_SIZE, CHUNK_OVERLAP


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PageText:
    """Raw text extracted from a single PDF page."""

    page_number: int  # 1-based
    text: str


@dataclass
class Chunk:
    """A single text chunk ready to be embedded and stored."""

    chunk_id: str          # Stable, deterministic identifier
    filename: str
    page_number: int       # Page where the chunk starts
    text: str


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


def extract_pages(pdf_bytes: bytes, filename: str) -> List[PageText]:
    """
    Open *pdf_bytes* as a PDF and return one PageText per page.

    Raises:
        ValueError: if the bytes are not a valid PDF.
        ValueError: if the PDF contains no extractable text.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{filename}': {exc}") from exc

    pages: List[PageText] = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")  # plain text, preserves newlines
        if text.strip():
            pages.append(PageText(page_number=page_index + 1, text=text))

    doc.close()

    if not pages:
        raise ValueError(
            f"'{filename}' contains no extractable text. "
            "It may be a scanned image PDF. OCR is not supported in this version."
        )

    return pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _make_chunk_id(filename: str, start_char: int) -> str:
    """Return a short, stable hex ID for a chunk."""
    key = f"{filename}::{start_char}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def chunk_pages(
    pages: List[PageText],
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Split page text into overlapping chunks while tracking page numbers.

    Strategy:
      1. Build a single concatenated text with page-boundary markers so we
         can walk back to find which page a character position belongs to.
      2. Slide a window of *chunk_size* characters with *chunk_overlap* step.
      3. Record the page number for each chunk's start position.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    # Build a list of (start_char, page_number) boundaries.
    segments: list[tuple[int, int, str]] = []  # (start, page_number, text)
    cursor = 0
    for page in pages:
        segments.append((cursor, page.page_number, page.text))
        cursor += len(page.text)

    full_text = "".join(p.text for p in pages)
    total = len(full_text)

    if total == 0:
        return []

    def page_at(char_pos: int) -> int:
        """Return the 1-based page number for a character position."""
        result = segments[0][1]
        for start, pnum, _ in segments:
            if start <= char_pos:
                result = pnum
            else:
                break
        return result

    chunks: List[Chunk] = []
    step = chunk_size - chunk_overlap
    pos = 0

    while pos < total:
        end = min(pos + chunk_size, total)
        text_slice = full_text[pos:end].strip()

        if text_slice:
            chunk_id = _make_chunk_id(filename, pos)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    filename=filename,
                    page_number=page_at(pos),
                    text=text_slice,
                )
            )

        pos += step

    return chunks


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def process_pdf(pdf_bytes: bytes, filename: str) -> List[Chunk]:
    """
    Full pipeline: bytes → extracted pages → chunks.

    This is the only function the rest of the application needs to call.
    """
    pages = extract_pages(pdf_bytes, filename)
    return chunk_pages(pages, filename)
