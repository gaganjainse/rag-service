# RAG Service

Production-style **RAG API** with hybrid retrieval: dense embeddings + BM25-style keyword
scoring, fused with **Reciprocal Rank Fusion (RRF)** over a ChromaDB vector store. Grounded
answers with source citations via any OpenAI-compatible chat endpoint (or offline fallback).

## Features

- `POST /ingest/text` & `POST /ingest/file` — chunk (word-based, with overlap) and index documents
- `POST /search` — hybrid retrieval (vector + keyword), RRF merge
- `POST /ask` — RAG answer generation with `[source]` citations
- Embedding backends: `sentence-transformers` (real) or a deterministic hashing embedder (offline/dev)
- LLM: OpenAI-compatible `chat/completions`; offline echo fallback when no `OPENAI_API_KEY`
- Docker Compose with persistent ChromaDB volume
- CI: `pytest` on every push (28 tests)

## Quick start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload

curl -X POST localhost:8000/ingest/text \
  -H 'Content-Type: application/json' \
  -d '{"text": "VIT Vellore focuses on AI and systems...", "source": "vit"}'

curl -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What does the CS program focus on?", "k": 3}'
```

### Production embeddings (optional)

```bash
pip install sentence-transformers
export EMBEDDING_BACKEND=st        # or leave "auto"
export EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Use a real LLM

```bash
export OPENAI_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini       # any OpenAI-compatible model
```

### Docker

```bash
docker compose up --build
```

## Architecture

```
ingest ──► chunk_text ──► embeddings ──► ChromaDB (persistent)
                                        ▲
ask ──► hybrid_search (dense + BM25, RRF) ──► context ──► LLM ──► grounded answer + sources
```

Evaluated with [`llm-eval-harness`](https://github.com/gaganjainse/llm-eval-harness):
faithfulness, answer relevance, and correctness against a golden set.

## License

MIT
