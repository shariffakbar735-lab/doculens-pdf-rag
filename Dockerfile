# ─────────────────────────────────────────────────────────────────────────────
# RagKit Simple – Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build: only the runtime stage is shipped.
# Model weights are NOT baked in; they live in a Docker volume.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim AS base

# Keep Python output unbuffered so logs appear immediately
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install OS-level dependencies needed by PyMuPDF and chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Dependency layer (cached unless requirements.txt changes) ─────────────
COPY requirements.txt .
RUN pip install -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────
COPY app.py rag.py pdf_processor.py config.py ./

# ── Streamlit configuration ───────────────────────────────────────────────
RUN mkdir -p /app/.streamlit
COPY .streamlit/config.toml /app/.streamlit/config.toml 2>/dev/null || true

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
