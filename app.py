"""
app.py – DocuLens web interface.

Layout:
  Sidebar  → Ollama status, indexed documents, PDF upload (auto-indexes).
  Main     → Logo/header, chat input, answer display, clean source citations.

Design principles:
  - All state is kept in st.session_state.
  - Auto-indexing: uploading a PDF triggers indexing immediately, no extra
    button click required.
  - Duplicate detection uses sha256(pdf_bytes) so same-content re-uploads are
    skipped, while same-filename-different-content files are handled correctly.
  - Errors are shown as st.error() messages, never raised to the user.
  - rag.py, pdf_processor.py and config.py are unchanged.
"""

from __future__ import annotations

import hashlib
import logging

import streamlit as st

import rag
from config import (
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_MB,
    OLLAMA_HOST,
)
from pdf_processor import process_pdf

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuLens – PDF Q&A",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Inline SVG logo  (document + magnifying-glass, no external dependency)
# ---------------------------------------------------------------------------
_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="52" height="52">
  <!-- Document body -->
  <rect x="8" y="6" width="34" height="44" rx="3" ry="3"
        fill="#EEF2FF" stroke="#4F46E5" stroke-width="2.5"/>
  <!-- Folded corner -->
  <polyline points="32,6 42,6 42,16 32,16 32,6"
            fill="#C7D2FE" stroke="#4F46E5" stroke-width="2"/>
  <!-- Text lines -->
  <line x1="14" y1="24" x2="36" y2="24" stroke="#6366F1" stroke-width="2" stroke-linecap="round"/>
  <line x1="14" y1="30" x2="36" y2="30" stroke="#6366F1" stroke-width="2" stroke-linecap="round"/>
  <line x1="14" y1="36" x2="28" y2="36" stroke="#6366F1" stroke-width="2" stroke-linecap="round"/>
  <!-- Magnifying glass circle -->
  <circle cx="44" cy="46" r="11" fill="none" stroke="#4F46E5" stroke-width="3"/>
  <circle cx="44" cy="46" r="6" fill="#EEF2FF"/>
  <!-- Magnifying glass handle -->
  <line x1="52" y1="54" x2="59" y2="61" stroke="#4F46E5" stroke-width="3.5"
        stroke-linecap="round"/>
</svg>
"""

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []

# Maps sha256 hex → filename, so we know which content hashes are already indexed.
if "indexed_hashes" not in st.session_state:
    st.session_state.indexed_hashes: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Helper: content hash for duplicate detection
# ---------------------------------------------------------------------------

def _file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Helper: Ollama status indicator (compact, sidebar)
# ---------------------------------------------------------------------------

def _render_ollama_status() -> bool:
    """Return True when Ollama is healthy; show a compact indicator."""
    status = rag.check_ollama()
    if status["ok"]:
        st.sidebar.success("✅ Ollama connected", icon=None)
        return True
    else:
        st.sidebar.error(f"⚠️ Ollama unavailable\n\n{status['error']}")
        return False


# ---------------------------------------------------------------------------
# Helper: document list with delete buttons
# ---------------------------------------------------------------------------

def _render_document_list():
    docs = rag.list_documents()
    if not docs:
        st.sidebar.caption("No documents indexed yet.")
        return

    for doc in docs:
        col1, col2 = st.sidebar.columns([5, 1])
        col1.markdown(f"📄 `{doc}`")
        if col2.button("🗑", key=f"del_{doc}", help=f"Remove {doc}"):
            deleted = rag.delete_document(doc)
            # Also remove from our hash cache
            st.session_state.indexed_hashes = {
                h: fn for h, fn in st.session_state.indexed_hashes.items()
                if fn != doc
            }
            st.sidebar.success(f"Removed '{doc}' ({deleted} chunks).")
            st.rerun()


# ---------------------------------------------------------------------------
# Helper: auto-indexing upload section
# ---------------------------------------------------------------------------

def _render_upload_section(ollama_ok: bool):
    """
    Upload widget that triggers indexing automatically on file selection.

    Duplicate detection:
      - Same content (any filename): sha256 already in indexed_hashes → skip.
      - Same filename, different content: new hash → delete old, index new.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Upload a PDF**")

    uploaded = st.sidebar.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help=f"Max {MAX_UPLOAD_MB} MB · Text-based PDFs only",
        label_visibility="collapsed",
    )

    if uploaded is None:
        return

    # Size guard
    if uploaded.size > MAX_UPLOAD_BYTES:
        st.sidebar.error(
            f"File too large ({uploaded.size // (1024 * 1024)} MB). "
            f"Max allowed: {MAX_UPLOAD_MB} MB."
        )
        return

    if not ollama_ok:
        st.sidebar.warning("Ollama is not available. Cannot index until it is running.")
        return

    pdf_bytes = uploaded.read()
    content_hash = _file_hash(pdf_bytes)
    filename = uploaded.name

    # ── Already indexed (same content) ──────────────────────────────────────
    if content_hash in st.session_state.indexed_hashes:
        existing_name = st.session_state.indexed_hashes[content_hash]
        if existing_name == filename:
            st.sidebar.info(f"✅ '{filename}' is already indexed.")
        else:
            st.sidebar.info(
                f"✅ This file's content is already indexed as '{existing_name}'."
            )
        return

    # ── Same filename, different content: remove stale chunks first ─────────
    existing_docs = rag.list_documents()
    if filename in existing_docs:
        rag.delete_document(filename)
        # Remove stale hash entries for this filename
        st.session_state.indexed_hashes = {
            h: fn for h, fn in st.session_state.indexed_hashes.items()
            if fn != filename
        }

    # ── Index the new file ───────────────────────────────────────────────────
    with st.sidebar.status(f"Indexing '{filename}'…", expanded=True) as sw:
        try:
            sw.write("Extracting text and page numbers…")
            chunks = process_pdf(pdf_bytes, filename)

            sw.write(f"Embedding {len(chunks)} chunks via Ollama…")
            added = rag.index_chunks(chunks)

            st.session_state.indexed_hashes[content_hash] = filename

            if added == 0:
                sw.update(label="Already up to date.", state="complete")
                st.sidebar.info(f"'{filename}' — no new chunks (all already indexed).")
            else:
                sw.update(label=f"✅ Indexed {added} chunks.", state="complete")

            st.rerun()

        except ValueError as exc:
            sw.update(label="PDF error", state="error")
            st.sidebar.error(f"Could not read PDF: {exc}")
        except RuntimeError as exc:
            sw.update(label="Ollama error", state="error")
            st.sidebar.error(f"Ollama error: {exc}")
        except Exception as exc:
            sw.update(label="Unexpected error", state="error")
            st.sidebar.error(f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Helper: render sources block (clean, no distance values, no "Excerpt N")
# ---------------------------------------------------------------------------

def _render_sources(sources: list[dict], expanded: bool = False):
    """Render source passages in a compact, user-friendly expandable block."""
    if not sources:
        return

    with st.expander(f"Sources [{len(sources)}]", expanded=expanded):
        for i, src in enumerate(sources, 1):
            st.markdown(
                f"**[{i}]** `{src['filename']}` — page {src['page_number']}"
            )
            passage = src["text"][:500].strip()
            if len(src["text"]) > 500:
                passage += "…"
            st.markdown(
                f'<div style="'
                f'background:#F8FAFF;border-left:3px solid #6366F1;'
                f'padding:8px 12px;border-radius:4px;'
                f'font-size:0.9em;color:#374151;margin:4px 0 12px 0">'
                f'{passage}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Helper: render a single chat turn from history
# ---------------------------------------------------------------------------

def _render_chat_turn(turn: dict):
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        _render_sources(turn.get("sources", []), expanded=False)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main():
    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;padding:4px 0 8px 0">'
            f'{_LOGO_SVG}'
            f'<div><span style="font-size:1.4em;font-weight:700;color:#4F46E5">DocuLens</span>'
            f'<br><span style="font-size:0.78em;color:#6B7280">PDF question answering</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        ollama_ok = _render_ollama_status()

        st.markdown("---")
        st.markdown("**Indexed documents**")
        _render_document_list()

        _render_upload_section(ollama_ok)

    # ── Main area ─────────────────────────────────────────────────────────────
    # Header
    col_logo, col_title = st.columns([1, 9])
    with col_logo:
        st.markdown(_LOGO_SVG, unsafe_allow_html=True)
    with col_title:
        st.markdown(
            '<h1 style="margin:0;padding:0;color:#1F2937">DocuLens</h1>'
            '<p style="margin:2px 0 0 0;color:#6B7280;font-size:1.05em">'
            'Ask questions. Find answers in your documents.</p>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Replay chat history
    for turn in st.session_state.chat_history:
        _render_chat_turn(turn)

    # Chat input
    question = st.chat_input(
        "Ask a question about your uploaded document…",
        disabled=not ollama_ok,
    )

    if question:
        if not question.strip():
            return

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer…"):
                try:
                    result = rag.answer(question)
                    reply = result["answer"]
                    sources = result["sources"]
                except RuntimeError as exc:
                    reply = f"⚠️ Ollama error: {exc}"
                    sources = []
                except Exception as exc:
                    reply = f"⚠️ Unexpected error: {exc}"
                    sources = []

            st.markdown(reply)
            _render_sources(sources, expanded=True)

        st.session_state.chat_history.append(
            {"question": question, "answer": reply, "sources": sources}
        )


if __name__ == "__main__":
    main()
