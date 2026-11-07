# Architecture

Planned request flow for `POST /v1/ask`:

```
question -> route -> retrieve (hybrid) -> grade context
                         ^                    |
                         |  weak: rewrite     | good
                         +--------------------+--> generate answer with citations
```

Ingestion flow: file -> loader -> chunker -> embeddings -> vector store.
