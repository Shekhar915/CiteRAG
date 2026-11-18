# CiteRAG: Project Requirements

An agentic RAG service that answers questions from a document collection with
verifiable citations, and measures its own answer quality with an automated
evaluation harness.

## 1. Why this project
- Evaluation and failure analysis are what separate production RAG from demos.
- An agent that decides *how* to retrieve (documents, web, or ask the user)
  shows control-flow design, not just a fixed pipeline.
- Tracing, cost tracking and Docker show operational maturity.

## 2. Functional requirements
| ID  | Requirement | Priority |
|-----|-------------|----------|
| F1  | `POST /v1/documents` ingests PDF, Markdown and text files | Must |
| F2  | Documents are split into overlapping chunks with source metadata | Must |
| F3  | Chunks are embedded and stored in a vector store | Must |
| F4  | Hybrid retrieval: vector search combined with keyword (BM25) search | Must |
| F5  | `POST /v1/ask` returns an answer with citations (document, page, chunk) | Must |
| F6  | An agent chooses between document search, web search, or a clarifying question | Should |
| F7  | The agent grades retrieved context and rewrites the query when it is weak | Should |
| F8  | Answers that are not supported by context say so instead of guessing | Must |
| F9  | `POST /v1/ask/stream` streams the answer tokens | Should |
| F10 | `GET /health` and `GET /ready` | Must |

## 3. Evaluation requirements
- A versioned golden dataset of question, expected answer and source passages (`evals/`).
- Metrics: retrieval hit rate and MRR, answer faithfulness, answer relevance.
- One command runs the suite and writes a report; CI fails on regressions beyond a set threshold.
- A failure-analysis note in the docs showing real failures and the fixes tried.

## 4. Non-functional requirements
- **Observability:** every LLM and retrieval call traced (LangSmith or Langfuse), with latency and token cost per request.
- **Reliability:** timeouts, bounded retries, graceful errors.
- **Security:** secrets from environment only, upload size and type limits, API key auth.
- **Quality:** unit tests that run without a real LLM, integration tests for the API, Ruff and mypy, CI on every push.
- **Deployability:** `docker compose up` starts the API and the vector store.

## 5. Tech stack
Python 3.11+, FastAPI, LangGraph, Google Gemini (LLM and embeddings), Chroma
(vector store), an in-house BM25 index, Pydantic v2, pytest, Ruff, mypy, Docker, GitHub Actions.

## 6. Milestones (one real step per day)
1. Day 1: scaffold, requirements, tooling, document chunker with tests.
2. Settings, logging, Gemini client.
3. Ingestion: loaders for PDF, Markdown and text.
4. Embeddings and vector store.
5. Retrieval: vector search, then BM25 and hybrid fusion.
6. `/v1/ask` with citations.
7. Agent graph: route, grade context, rewrite query.
8. Golden dataset and evaluation harness.
9. Metrics report and CI regression gate.
10. Tracing and cost tracking.
11. Streaming endpoint and auth.
12. Docker compose, failure-analysis write-up, demo.

## 7. Out of scope
Multi-tenant access control, a custom frontend, model fine-tuning.
