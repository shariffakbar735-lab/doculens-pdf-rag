"""
tests/test_pdf_processor.py

Unit tests for pdf_processor.py.  No network calls, no Ollama dependency.
A synthetic PDF is created in-memory using PyMuPDF so tests work without
fixture files.
"""

from __future__ import annotations

import io
import pytest
import pymupdf as fitz  # PyMuPDF (1.24+ preferred import)


# ---------------------------------------------------------------------------
# Helpers – create synthetic PDFs in memory
# ---------------------------------------------------------------------------


def _make_pdf(pages: list[str]) -> bytes:
    """Return bytes of a valid PDF where each item in *pages* is one page."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((50, 100), text, fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_empty_pdf() -> bytes:
    """Return bytes of a valid PDF with one blank page (no text)."""
    doc = fitz.open()
    doc.new_page()
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from pdf_processor import (
    extract_pages,
    chunk_pages,
    process_pdf,
    Chunk,
    PageText,
)


# ---------------------------------------------------------------------------
# Tests – extract_pages
# ---------------------------------------------------------------------------


class TestExtractPages:
    def test_single_page_text_is_extracted(self):
        pdf = _make_pdf(["Hello world, this is page one."])
        pages = extract_pages(pdf, "test.pdf")
        assert len(pages) == 1
        assert "Hello" in pages[0].text
        assert pages[0].page_number == 1

    def test_multi_page_numbers_are_correct(self):
        pdf = _make_pdf(["Page one content", "Page two content", "Page three"])
        pages = extract_pages(pdf, "multi.pdf")
        assert len(pages) == 3
        assert pages[0].page_number == 1
        assert pages[1].page_number == 2
        assert pages[2].page_number == 3

    def test_empty_pdf_raises_value_error(self):
        pdf = _make_empty_pdf()
        with pytest.raises(ValueError, match="no extractable text"):
            extract_pages(pdf, "empty.pdf")

    def test_invalid_bytes_raises_value_error(self):
        with pytest.raises(ValueError, match="Cannot open PDF"):
            extract_pages(b"this is not a pdf", "bad.pdf")

    def test_page_numbers_are_one_based(self):
        pdf = _make_pdf(["First page"])
        pages = extract_pages(pdf, "test.pdf")
        assert pages[0].page_number == 1  # never 0


# ---------------------------------------------------------------------------
# Tests – chunk_pages
# ---------------------------------------------------------------------------


class TestChunkPages:
    def _pages(self, *texts: str) -> list[PageText]:
        return [PageText(page_number=i + 1, text=t) for i, t in enumerate(texts)]

    def test_short_text_produces_one_chunk(self):
        pages = self._pages("Short text.")
        chunks = chunk_pages(pages, "doc.pdf", chunk_size=200, chunk_overlap=20)
        assert len(chunks) == 1
        assert chunks[0].filename == "doc.pdf"
        assert chunks[0].page_number == 1

    def test_chunk_size_controls_number_of_chunks(self):
        # 400 chars of 'A', chunk_size=100, overlap=10 → step=90 → ~5 chunks
        pages = self._pages("A" * 400)
        chunks = chunk_pages(pages, "doc.pdf", chunk_size=100, chunk_overlap=10)
        assert len(chunks) >= 4

    def test_overlap_means_adjacent_chunks_share_content(self):
        text = "abcdefghijklmnopqrstuvwxyz" * 4  # 104 chars
        pages = self._pages(text)
        chunks = chunk_pages(pages, "doc.pdf", chunk_size=20, chunk_overlap=5)
        # The end of chunk 0 should appear at the start of chunk 1
        end_of_first = chunks[0].text[-5:]
        start_of_second = chunks[1].text[:5]
        assert end_of_first == start_of_second

    def test_page_number_is_preserved_across_pages(self):
        pages = self._pages("Page one " * 50, "Page two " * 50)
        chunks = chunk_pages(pages, "doc.pdf", chunk_size=100, chunk_overlap=10)
        page_numbers = {c.page_number for c in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers

    def test_chunk_ids_are_unique(self):
        pages = self._pages("Some text " * 200)
        chunks = chunk_pages(pages, "doc.pdf", chunk_size=80, chunk_overlap=10)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_invalid_chunk_size_raises(self):
        pages = self._pages("text")
        with pytest.raises(ValueError):
            chunk_pages(pages, "doc.pdf", chunk_size=0)

    def test_invalid_overlap_raises(self):
        pages = self._pages("text")
        with pytest.raises(ValueError):
            chunk_pages(pages, "doc.pdf", chunk_size=100, chunk_overlap=100)

    def test_empty_pages_returns_empty_list(self):
        chunks = chunk_pages([], "doc.pdf")
        assert chunks == []


# ---------------------------------------------------------------------------
# Tests – process_pdf (integration of extract + chunk)
# ---------------------------------------------------------------------------


class TestProcessPdf:
    def test_valid_pdf_returns_chunks_with_metadata(self):
        pdf = _make_pdf(["This is the first page.", "This is the second page."])
        chunks = process_pdf(pdf, "sample.pdf")
        assert len(chunks) >= 1
        for c in chunks:
            assert c.filename == "sample.pdf"
            assert c.page_number >= 1
            assert c.text.strip() != ""

    def test_invalid_pdf_raises_value_error(self):
        with pytest.raises(ValueError):
            process_pdf(b"garbage bytes", "bad.pdf")

    def test_chunk_ids_are_deterministic(self):
        pdf = _make_pdf(["Deterministic chunk content here."])
        chunks1 = process_pdf(pdf, "det.pdf")
        chunks2 = process_pdf(pdf, "det.pdf")
        ids1 = [c.chunk_id for c in chunks1]
        ids2 = [c.chunk_id for c in chunks2]
        assert ids1 == ids2
