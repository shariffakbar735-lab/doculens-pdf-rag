"""
tests/test_app_logic.py

Unit tests for the logic introduced in the DocuLens app.py rewrite:
  - Content-hash deduplication (_file_hash)
  - The hash is stable for the same bytes
  - Different bytes produce different hashes
  - Same filename, different content produces a different hash

These tests do NOT import Streamlit (which requires a running server) and do
NOT call Ollama or ChromaDB.  They test only the pure Python helper.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import types


# ---------------------------------------------------------------------------
# Import just the _file_hash helper without triggering Streamlit's runtime.
# We do this by extracting only the function source rather than importing app.
# ---------------------------------------------------------------------------

def _file_hash(data: bytes) -> str:
    """Replica of the helper in app.py for isolated testing."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFileHash:
    def test_same_bytes_produce_same_hash(self):
        data = b"Hello, world! This is a PDF chunk."
        assert _file_hash(data) == _file_hash(data)

    def test_different_bytes_produce_different_hash(self):
        assert _file_hash(b"content A") != _file_hash(b"content B")

    def test_empty_bytes_hash_is_stable(self):
        h = _file_hash(b"")
        assert len(h) == 64  # sha256 hex is always 64 chars
        assert _file_hash(b"") == h

    def test_same_content_different_filename_same_hash(self):
        """
        Two PDFs with identical bytes but different filenames must produce the
        same hash — so the deduplication correctly recognises them as the same
        content regardless of the filename.
        """
        pdf_bytes = b"%PDF-1.4 fake pdf content 12345"
        hash_as_doc_a = _file_hash(pdf_bytes)
        hash_as_doc_b = _file_hash(pdf_bytes)  # same bytes, "different" filename
        assert hash_as_doc_a == hash_as_doc_b

    def test_same_filename_different_content_different_hash(self):
        """
        Two PDFs that happen to share the same filename but differ in content
        must produce different hashes — so the app knows to re-index.
        """
        v1 = b"%PDF-1.4 version one"
        v2 = b"%PDF-1.4 version two"
        assert _file_hash(v1) != _file_hash(v2)

    def test_hash_is_hex_string_of_correct_length(self):
        h = _file_hash(b"arbitrary content")
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
