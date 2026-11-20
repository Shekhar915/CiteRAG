# CiteRAG

Agentic RAG service that answers from your documents with citations, and
scores its own answers with an evaluation harness.

> Status: day 6. `POST /v1/ask` answers from the indexed documents with
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
pip install -e ".[dev]"
pytest -q
```
