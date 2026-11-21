# CiteRAG

Agentic RAG service that answers from your documents with citations, and
scores its own answers with an evaluation harness.

> Status: day 6 plus a web UI (see below). `POST /v1/ask` answers from the indexed documents with
> numbered citations, and says so when the documents don't cover the question.
> Under it sit hybrid retrieval (BM25 plus vectors, merged with Reciprocal Rank
> Fusion), a Gemini embedding client, ingestion and settings. The document
> upload endpoint and the Chroma-backed store are next.
> Document ingestion (PDF, Markdown, text), settings, logging and the health
> endpoints came before it. The Chroma-backed store is next; tests use an
> in-memory store. See [REQUIREMENTS.md](REQUIREMENTS.md) for the scope and
> milestones.

## Structure

```
src/citerag/
├── api/            # FastAPI app and routes
├── core/           # settings, logging
├── ingestion/      # loading and chunking documents
├── retrieval/      # embeddings, vector store, hybrid search
├── agent/          # LangGraph agent and nodes
├── schemas/        # Pydantic models
└── observability/  # tracing and cost tracking
evals/              # golden dataset and evaluation runner
tests/              # unit and integration tests
docs/               # architecture notes and failure analysis
deploy/             # Dockerfile and compose files
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,ui]"
pytest -q
```

## Web UI

Upload documents, index them, then ask questions and see the answer with its
sources.

```bash
copy .env.example .env          # Linux/macOS: cp .env.example .env
# set GEMINI_API_KEY in .env
streamlit run src/citerag/ui/app.py
```

1. In the sidebar, upload PDF, Markdown or text files and press **Index documents**.
2. Type a question in the chat box.
3. Open **Source** entries under an answer to see the cited passages. If the
   documents don't cover the question, the app says so instead of guessing.

Each document is stored with a content hash. Uploading the same file again does
nothing, and uploading a changed file with the same name replaces its old
chunks, so outdated text is never retrieved or cited. Use the bin button next to
a document to remove it.

The index is kept in memory, so documents must be uploaded again after a restart.
