"""
rag.py – ChromaDB indexing, similarity retrieval, and Ollama answer generation.

Responsibilities:
  - Maintain a ChromaDB collection.
  - Generate embeddings via Ollama (nomic-embed-text).
  - Index chunks, skip already-indexed ones (idempotent).
  - Retrieve the top-K most relevant chunks for a question.
  - Build a context string and call Ollama for an answer.
  - Delete all chunks belonging to a document.

Design notes:
  - No LangChain, no agent frameworks.
  - Each public function is a plain Python function: easy to call and test.
  - Ollama errors are surfaced as descriptive exceptions rather than swallowed.
"""

from __future__ import annotations

import logging
from typing import List

import chromadb
import requests

from config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    EMBED_MODEL,
    LLM_MODEL,
    OLLAMA_HOST,
    RETRIEVAL_TOP_K,
)
from pdf_processor import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ChromaDB client – module-level singleton (one client per process)
# ---------------------------------------------------------------------------

_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------


def _ollama_embed(texts: List[str]) -> List[List[float]]:
    """
    Call the Ollama embedding endpoint and return one vector per input text.

    Raises:
        RuntimeError: if Ollama is unreachable or returns an error.
    """
    url = f"{OLLAMA_HOST}/api/embed"
    try:
        resp = requests.post(
            url,
            json={"model": EMBED_MODEL, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. "
            "Is Ollama running? Check the OLLAMA_HOST environment variable."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"Ollama embedding request failed ({resp.status_code}): {resp.text}"
        ) from exc

    data = resp.json()

    # Ollama /api/embed returns {"embeddings": [[...], ...]}
    if "embeddings" not in data:
        raise RuntimeError(f"Unexpected Ollama embed response: {data}")

    return data["embeddings"]


def _ollama_generate(prompt: str) -> str:
    """
    Call the Ollama chat/generate endpoint and return the assistant's reply.

    Raises:
        RuntimeError: if Ollama is unreachable or returns an error.
    """
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": LLM_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. "
            "Is Ollama running? Check the OLLAMA_HOST environment variable."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"Ollama generate request failed ({resp.status_code}): {resp.text}"
        ) from exc

    data = resp.json()
    try:
        return data["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Ollama chat response: {data}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_ollama() -> dict:
    """
    Return {"ok": True} if Ollama is reachable and both required models are
    available, otherwise {"ok": False, "error": "<message>"}.
    """
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        resp.raise_for_status()
        models = {m["name"] for m in resp.json().get("models", [])}

        missing = []
        for needed in [LLM_MODEL, EMBED_MODEL]:
            # Ollama model names may carry a ":latest" suffix
            if needed not in models and f"{needed}:latest" not in models:
                missing.append(needed)

        if missing:
            return {
                "ok": False,
                "error": (
                    f"The following models are not pulled: {', '.join(missing)}. "
                    f"Run: ollama pull {' && ollama pull '.join(missing)}"
                ),
            }
        return {"ok": True}

    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "error": (
                f"Ollama is not reachable at {OLLAMA_HOST}. "
                "Start Ollama or check OLLAMA_HOST."
            ),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def index_chunks(chunks: List[Chunk]) -> int:
    """
    Embed and store *chunks* in ChromaDB.  Already-indexed chunk IDs are
    skipped so that re-uploading the same document does not re-embed.

    Returns:
        The number of *new* chunks actually added.
    """
    if not chunks:
        return 0

    collection = _get_collection()

    # Find which IDs are already present
    existing_ids = set(
        collection.get(ids=[c.chunk_id for c in chunks])["ids"]
    )
    new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

    if not new_chunks:
        logger.info("All %d chunks already indexed; skipping.", len(chunks))
        return 0

    texts = [c.text for c in new_chunks]
    embeddings = _ollama_embed(texts)

    collection.add(
        ids=[c.chunk_id for c in new_chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"filename": c.filename, "page_number": c.page_number}
            for c in new_chunks
        ],
    )

    logger.info("Indexed %d new chunks.", len(new_chunks))
    return len(new_chunks)


def list_documents() -> List[str]:
    """Return the unique filenames currently stored in ChromaDB."""
    collection = _get_collection()
    result = collection.get(include=["metadatas"])
    filenames = {m["filename"] for m in result["metadatas"] if "filename" in m}
    return sorted(filenames)


def delete_document(filename: str) -> int:
    """
    Remove all chunks associated with *filename* from ChromaDB.

    Returns:
        The number of chunks deleted.
    """
    collection = _get_collection()
    result = collection.get(
        where={"filename": filename}, include=["metadatas"]
    )
    ids_to_delete = result["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info("Deleted %d chunks for '%s'.", len(ids_to_delete), filename)
    return len(ids_to_delete)


def retrieve(question: str, top_k: int = RETRIEVAL_TOP_K) -> List[dict]:
    """
    Embed *question* and return the top-k most similar chunks.

    Each returned dict has keys: text, filename, page_number, distance.
    """
    collection = _get_collection()

    if collection.count() == 0:
        return []

    [q_embedding] = _ollama_embed([question])

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "text": doc,
                "filename": meta.get("filename", "unknown"),
                "page_number": meta.get("page_number", "?"),
                "distance": dist,
            }
        )
    return hits


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """\
You are a helpful assistant that answers questions strictly based on the \
provided document excerpts. If the excerpts do not contain enough information \
to answer the question, say "I don't have enough information in the provided \
document to answer this question." Do not make up facts.\
"""

_PROMPT_TEMPLATE = """\
{system}

Document excerpts:
{context}

Question: {question}

Answer:"""


def answer(question: str, top_k: int = RETRIEVAL_TOP_K) -> dict:
    """
    Full RAG pipeline: retrieve → build context → generate answer.

    Returns a dict with:
      - answer (str)
      - sources (list of dicts with text, filename, page_number, distance)
      - retrieved (bool) – whether any chunks were found
    """
    sources = retrieve(question, top_k=top_k)

    if not sources:
        return {
            "answer": (
                "No documents are indexed yet. "
                "Please upload a PDF first."
            ),
            "sources": [],
            "retrieved": False,
        }

    context_parts = []
    for i, src in enumerate(sources, 1):
        context_parts.append(
            f"[Excerpt {i} – {src['filename']}, page {src['page_number']}]\n"
            f"{src['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = _PROMPT_TEMPLATE.format(
        system=_SYSTEM_INSTRUCTION,
        context=context,
        question=question,
    )

    reply = _ollama_generate(prompt)

    return {
        "answer": reply,
        "sources": sources,
        "retrieved": True,
    }
