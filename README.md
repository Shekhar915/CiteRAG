# CiteRAG

Agentic RAG service that answers from your documents with citations, and
scores its own answers with an evaluation harness.

> Status: day 10. Upload documents, ask questions and get answers with numbered
> citations, through the API or a web UI. Retrieval is hybrid (BM25 plus
> vectors, merged with Reciprocal Rank Fusion), and an optional LangGraph agent
> grades the retrieved passages and rewrites weak queries. When the documents
> don't cover a question, the answer says so. Next: the evaluation harness and
> a Chroma-backed store (tests and the UI currently use an in-memory index).
> See [REQUIREMENTS.md](REQUIREMENTS.md) for the scope and milestones.

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

## API

Run `uvicorn citerag.api.main:app --reload` and open http://localhost:8000/docs.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health`, `/ready` | Liveness and configuration check |
| POST | `/v1/documents` | Upload a PDF, Markdown or text file (re-uploads replace old chunks) |
| GET | `/v1/documents` | List indexed documents |
| DELETE | `/v1/documents/{name}` | Remove a document |
| POST | `/v1/ask` | Ask a question; returns an answer with citations |

Uploads over `MAX_UPLOAD_MB` return 413, and unsupported file types return 422.

Add `"agentic": true` to a `/v1/ask` request to grade the retrieved passages and
rewrite weak queries before answering (see [docs/architecture.md](docs/architecture.md)).
It makes extra model calls, so it is off by default.

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
