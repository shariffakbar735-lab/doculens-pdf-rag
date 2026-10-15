# DocuLens — PDF Question Answering

A small, self-contained Retrieval-Augmented Generation (RAG) application.
Upload a PDF, ask questions about it, and receive answers grounded in the
document's actual content — with source citations showing the filename and
page number.

## Architecture (in one paragraph)

PDF upload → PyMuPDF text extraction (page-by-page) → overlapping text chunks →
`nomic-embed-text` embeddings via Ollama → ChromaDB vector store → cosine
similarity retrieval → context prompt → `qwen3:4b` answer generation via Ollama
→ Streamlit answer display with expandable source passages.

---

## Prerequisites

| Tool | Required for |
|---|---|
| [Ollama](https://ollama.com) | Local LLM inference (both paths) |
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Docker path |
| Python 3.11+ | Local path |

---

## Option A — Run with Docker (recommended)

This is the simplest path. Docker Desktop must be running.

### 1. Clone / copy the project

```
cd ragkit-simple
```

### 2. Copy the example environment file (optional)

```
copy .env.example .env
```

Edit `.env` if you want to change models or chunk sizes.

### 3. Start everything

```
docker compose up --build
```

This will:
1. Start an Ollama server container.
2. Run a one-shot `model-init` container that pulls `qwen3:4b` and
   `nomic-embed-text` into a persistent Docker volume.
3. Build and start the Streamlit application.

> **First run note**: Pulling `qwen3:4b` (~2.6 GB) and `nomic-embed-text`
> (~270 MB) can take several minutes depending on your connection.  The
> application waits for Ollama to be healthy before starting.

### 4. Open the application

<http://localhost:8501>

### 5. Stop

```
docker compose down
```

Data (model weights and ChromaDB) is stored in Docker named volumes and
survives restarts.  To wipe everything:

```
docker compose down -v
```

---

## Option B — Run locally (Ollama already installed)

Use this if you already have Ollama running on your machine.

### 1. Pull the required models

```
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

### 2. Create a virtual environment

```
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Start the application

```
streamlit run app.py
```

Open <http://localhost:8501>.

To point the app at a non-default Ollama URL:

```
set OLLAMA_HOST=http://localhost:11434   # Windows
# or
export OLLAMA_HOST=http://localhost:11434  # macOS / Linux
streamlit run app.py
```

---

## Usage

1. Open the app in your browser.
2. Check the **Ollama status** in the sidebar — it must show ✅.
3. Use the **Upload a PDF** section to choose a file and click **Index this PDF**.
4. Wait for indexing to complete (the status indicator updates in real time).
5. Type a question in the chat input at the bottom of the page.
6. The answer appears with expandable **Source passages** showing the filename
   and page number of each supporting excerpt.
7. To remove a document, click the 🗑️ button next to its name in the sidebar.

---

## Configuration

All settings are read from environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `qwen3:4b` | Answer-generation model |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlapping characters between chunks |
| `RETRIEVAL_TOP_K` | `4` | Chunks retrieved per question |
| `MAX_UPLOAD_MB` | `50` | Maximum PDF upload size |
| `CHROMA_PATH` | `./chroma_db` | Local ChromaDB storage path |

---

## Running Tests

```
cd ragkit-simple
pip install -r requirements.txt
pytest tests/ -v
```

Tests use mocks for Ollama and ChromaDB, so no live services are required.

---

## Project Structure

```
ragkit-simple/
├── app.py              # Streamlit interface
├── rag.py              # ChromaDB + Ollama RAG pipeline
├── pdf_processor.py    # PDF extraction and chunking
├── config.py           # Environment variable configuration
├── requirements.txt
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .env.example
├── README.md
└── tests/
    ├── test_pdf_processor.py
    └── test_rag.py
```

---

## Troubleshooting

**Ollama shows ⚠️ in the sidebar**
- Ensure Ollama is running: `ollama serve`
- If using Docker, wait for `model-init` to finish pulling models.
- Check `OLLAMA_HOST` points to the correct address.

**"Models not pulled" error**
- Run `ollama pull qwen3:4b` and `ollama pull nomic-embed-text` manually.

**PDF upload fails with "no extractable text"**
- The PDF is probably a scanned image. OCR is not supported in this version.
  Use a PDF with embedded text layers.

**Answers seem unrelated to the document**
- Try reducing `CHUNK_SIZE` or increasing `RETRIEVAL_TOP_K`.
- Check that indexing completed without errors.

**Docker: port 8501 or 11434 already in use**
- Stop any existing Ollama or Streamlit processes, or change the host port in
  `compose.yaml`.

---

## Known Limitations

- Scanned / image-only PDFs are not supported (no OCR).
- The application does not support authentication or multi-user sessions.
- Docker GPU pass-through for Ollama is not configured in this version;
  inference runs on CPU (slower for large models).
- The `model-init` container exits after pulling; it does not re-pull models
  that were already downloaded unless the `ollama_data` volume is deleted.

---

## Shutdown

```
# Docker
docker compose down

# Local
Ctrl+C  (in the Streamlit terminal)
```
