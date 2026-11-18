# CiteRAG

Agentic RAG service that answers from your documents with citations, and
scores its own answers with an evaluation harness.

> Status: day 5. Hybrid retrieval is in place: BM25 keyword search combined
> with vector search through Reciprocal Rank Fusion. Before it came a Gemini
> embedding client, a vector store interface, batched indexing and source
> filtering.
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
