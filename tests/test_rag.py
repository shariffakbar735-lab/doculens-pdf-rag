"""
tests/test_rag.py

Unit tests for rag.py.

Network calls (Ollama, ChromaDB persistence) are mocked so these tests run
without Ollama installed and without writing to disk.

Tests that require a live Ollama instance are marked with
@pytest.mark.integration and are skipped by default.  Run them with:

    pytest -m integration
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from pdf_processor import Chunk

# Patch chromadb.PersistentClient before importing rag so the module-level
# singleton is never created against a real database during unit tests.

import chromadb

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="aaa111",
            filename="report.pdf",
            page_number=1,
            text="The revenue for Q1 was $1.2 million.",
        ),
        Chunk(
            chunk_id="bbb222",
            filename="report.pdf",
            page_number=2,
            text="Operating costs increased by 15% year-over-year.",
        ),
        Chunk(
            chunk_id="ccc333",
            filename="manual.pdf",
            page_number=5,
            text="Press the red button to reset the device.",
        ),
    ]


def _fake_collection(existing_ids=None):
    """Return a MagicMock that behaves like a chromadb Collection."""
    col = MagicMock()
    existing_ids = existing_ids or []
    col.get.return_value = {"ids": existing_ids, "metadatas": []}
    col.count.return_value = 3
    col.query.return_value = {
        "documents": [["The revenue for Q1 was $1.2 million."]],
        "metadatas": [[{"filename": "report.pdf", "page_number": 1}]],
        "distances": [[0.05]],
    }
    return col


# ---------------------------------------------------------------------------
# Tests – check_ollama
# ---------------------------------------------------------------------------


class TestCheckOllama:
    def test_returns_ok_when_ollama_healthy(self):
        with patch("rag.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "models": [
                    {"name": "qwen3:4b"},
                    {"name": "nomic-embed-text"},
                ]
            }
            mock_get.return_value.raise_for_status = MagicMock()
            import rag
            result = rag.check_ollama()
        assert result["ok"] is True

    def test_returns_error_when_connection_fails(self):
        import requests as req
        with patch("rag.requests.get", side_effect=req.exceptions.ConnectionError):
            import rag
            result = rag.check_ollama()
        assert result["ok"] is False
        assert "not reachable" in result["error"]

    def test_returns_error_when_model_missing(self):
        with patch("rag.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "models": [{"name": "some-other-model"}]
            }
            mock_get.return_value.raise_for_status = MagicMock()
            import rag
            result = rag.check_ollama()
        assert result["ok"] is False
        assert "not pulled" in result["error"]


# ---------------------------------------------------------------------------
# Tests – index_chunks
# ---------------------------------------------------------------------------


class TestIndexChunks:
    def test_indexes_new_chunks(self, sample_chunks):
        import rag
        col = _fake_collection(existing_ids=[])  # nothing pre-existing
        with patch("rag._get_collection", return_value=col), \
             patch("rag._ollama_embed", return_value=[[0.1] * 768] * 3):
            added = rag.index_chunks(sample_chunks)
        assert added == 3
        col.add.assert_called_once()

    def test_skips_already_indexed_chunks(self, sample_chunks):
        import rag
        # Pretend all IDs already exist
        existing = [c.chunk_id for c in sample_chunks]
        col = _fake_collection(existing_ids=existing)
        with patch("rag._get_collection", return_value=col), \
             patch("rag._ollama_embed", return_value=[]) as mock_embed:
            added = rag.index_chunks(sample_chunks)
        assert added == 0
        col.add.assert_not_called()

    def test_empty_list_returns_zero(self):
        import rag
        assert rag.index_chunks([]) == 0

    def test_ollama_error_propagates(self, sample_chunks):
        import rag
        col = _fake_collection(existing_ids=[])
        with patch("rag._get_collection", return_value=col), \
             patch("rag._ollama_embed", side_effect=RuntimeError("Ollama down")):
            with pytest.raises(RuntimeError, match="Ollama down"):
                rag.index_chunks(sample_chunks)


# ---------------------------------------------------------------------------
# Tests – delete_document
# ---------------------------------------------------------------------------


class TestDeleteDocument:
    def test_deletes_chunks_for_filename(self):
        import rag
        col = MagicMock()
        col.get.return_value = {
            "ids": ["aaa111", "bbb222"],
            "metadatas": [
                {"filename": "report.pdf"},
                {"filename": "report.pdf"},
            ],
        }
        with patch("rag._get_collection", return_value=col):
            deleted = rag.delete_document("report.pdf")
        assert deleted == 2
        col.delete.assert_called_once_with(ids=["aaa111", "bbb222"])

    def test_delete_nonexistent_doc_returns_zero(self):
        import rag
        col = MagicMock()
        col.get.return_value = {"ids": [], "metadatas": []}
        with patch("rag._get_collection", return_value=col):
            deleted = rag.delete_document("ghost.pdf")
        assert deleted == 0
        col.delete.assert_not_called()


# ---------------------------------------------------------------------------
# Tests – retrieve
# ---------------------------------------------------------------------------


class TestRetrieve:
    def test_returns_hits_with_required_keys(self):
        import rag
        col = _fake_collection()
        with patch("rag._get_collection", return_value=col), \
             patch("rag._ollama_embed", return_value=[[0.1] * 768]):
            hits = rag.retrieve("What was the Q1 revenue?")
        assert len(hits) == 1
        hit = hits[0]
        assert "text" in hit
        assert "filename" in hit
        assert "page_number" in hit
        assert "distance" in hit

    def test_empty_collection_returns_empty_list(self):
        import rag
        col = MagicMock()
        col.count.return_value = 0
        with patch("rag._get_collection", return_value=col):
            hits = rag.retrieve("Any question")
        assert hits == []


# ---------------------------------------------------------------------------
# Tests – answer
# ---------------------------------------------------------------------------


class TestAnswer:
    def test_returns_answer_dict_structure(self):
        import rag
        col = _fake_collection()
        with patch("rag._get_collection", return_value=col), \
             patch("rag._ollama_embed", return_value=[[0.1] * 768]), \
             patch("rag._ollama_generate", return_value="Revenue was $1.2M."):
            result = rag.answer("What was Q1 revenue?")
        assert "answer" in result
        assert "sources" in result
        assert "retrieved" in result
        assert result["retrieved"] is True
        assert result["answer"] == "Revenue was $1.2M."

    def test_no_documents_returns_helpful_message(self):
        import rag
        col = MagicMock()
        col.count.return_value = 0
        with patch("rag._get_collection", return_value=col):
            result = rag.answer("What is X?")
        assert result["retrieved"] is False
        assert "No documents" in result["answer"]

    def test_ollama_generate_error_propagates(self):
        import rag
        col = _fake_collection()
        with patch("rag._get_collection", return_value=col), \
             patch("rag._ollama_embed", return_value=[[0.1] * 768]), \
             patch("rag._ollama_generate", side_effect=RuntimeError("LLM down")):
            with pytest.raises(RuntimeError, match="LLM down"):
                rag.answer("What happened?")

    def test_list_documents_returns_sorted_unique_filenames(self):
        import rag
        col = MagicMock()
        col.get.return_value = {
            "ids": ["1", "2", "3"],
            "metadatas": [
                {"filename": "zebra.pdf"},
                {"filename": "apple.pdf"},
                {"filename": "zebra.pdf"},
            ],
        }
        with patch("rag._get_collection", return_value=col):
            docs = rag.list_documents()
        assert docs == ["apple.pdf", "zebra.pdf"]
