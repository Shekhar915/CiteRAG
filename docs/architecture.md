# Architecture

## Agentic ask (implemented)

`POST /v1/ask` with `"agentic": true` runs a LangGraph loop in
`src/citerag/agent/graph.py`:

```
START -> retrieve -> grade --yes--> answer -> END
            ^          |
            |          +--no, attempts left--> rewrite
            +----------------------------------+
                       |
                       +--no, out of attempts--> give_up -> END
```

- `grade` asks the model whether the passages can answer the question. With no
  passages it answers "no" without a model call.
- `rewrite` produces a more specific search query.
- The loop is bounded by `max_attempts` (default 2 retrievals). When the context
  is still weak, the agent says so instead of guessing.
- Without the flag, `/v1/ask` runs plain retrieve then answer, with one model call.

## Original plan

Planned request flow for `POST /v1/ask`:

```
question -> route -> retrieve (hybrid) -> grade context
                         ^                    |
                         |  weak: rewrite     | good
                         +--------------------+--> generate answer with citations
```

Ingestion flow: file -> loader -> chunker -> embeddings -> vector store.
