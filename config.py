"""
config.py – Central configuration loaded from environment variables.

All tuneable parameters live here. The application reads these values once at
import time so there is a single, explicit place to look when something needs
to change.
"""

import os

# ---------------------------------------------------------------------------
# Ollama settings
# ---------------------------------------------------------------------------
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3:4b")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")

# ---------------------------------------------------------------------------
# Chunking settings
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))

# ---------------------------------------------------------------------------
# ChromaDB settings
# ---------------------------------------------------------------------------
CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "ragkit")

# ---------------------------------------------------------------------------
# Upload settings
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024
