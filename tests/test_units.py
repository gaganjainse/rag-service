"""Additional unit tests: chunking, embeddings, store, and pipeline behavior."""

from __future__ import annotations

import pytest

from app.embeddings import DIM, HashingEmbedder
from app.rag import RAGPipeline, PROMPT_TEMPLATE
from app.store import Document, VectorStore, chunk_text


# --- chunking ---

def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_short_text_single_chunk():
    chunks = chunk_text("just a few words")
    assert len(chunks) == 1
    assert chunks[0] == "just a few words"


def test_chunk_text_respects_max_size():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    for chunk in chunks:
        assert len(chunk.split()) <= 30


def test_chunk_text_never_empty_chunks():
    text = " ".join(f"token{i}" for i in range(50))
    chunks = chunk_text(text, chunk_size=10, overlap=3)
    assert chunks
    assert all(c.strip() for c in chunks)


# --- embeddings ---

def test_hashing_embedder_deterministic():
    e = HashingEmbedder()
    assert e.encode(["hello world"]) == e.encode(["hello world"])


def test_hashing_embedder_dim():
    e = HashingEmbedder()
    vec = e.encode(["test"])[0]
    assert len(vec) == DIM


def test_hashing_embedder_normalized():
    e = HashingEmbedder()
    vec = e.encode(["test"])[0]
    norm = sum(v * v for v in vec) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_hashing_embedder_similar_texts_are_close():
    e = HashingEmbedder()
    a = e.encode(["cats like fish"])[0]
    b = e.encode(["cats like fish and milk"])[0]
    dot = sum(x * y for x, y in zip(a, b))
    c = e.encode(["quantum chromodynamics"])[0]
    dot_bad = sum(x * y for x, y in zip(a, c))
    assert dot > dot_bad


# --- store ---

def test_add_documents_returns_chunk_count():
    store = VectorStore(HashingEmbedder(), persist_dir="/tmp/test-chroma-1")
    n = store.add_documents([Document("word " * 1200, source="a")])
    assert n > 1
    assert store.count() == n


def test_store_metadata_preserved():
    store = VectorStore(HashingEmbedder(), persist_dir="/tmp/test-chroma-2")
    store.add_documents([Document("alpha beta gamma delta epsilon", source="doc1", metadata={"kind": "manual"})])
    results = store.search("alpha beta", k=1)
    assert results[0]["source"] == "doc1"


def test_search_respects_k():
    store = VectorStore(HashingEmbedder(), persist_dir="/tmp/test-chroma-3")
    for i in range(5):
        store.add_documents([Document(f"topic{i} unique terms keywords indexing", source=f"s{i}")])
    results = store.search("topic unique terms", k=3)
    assert len(results) == 3


def test_search_k_larger_than_corpus():
    store = VectorStore(HashingEmbedder(), persist_dir="/tmp/test-chroma-4")
    store.add_documents([Document("only one document here", source="only")])
    results = store.search("one document", k=10)
    assert len(results) == 1


# --- pipeline ---

def test_prompt_template_contains_placeholders():
    filled = PROMPT_TEMPLATE.format(context="ctx", question="q")
    assert "ctx" in filled and "q" in filled


def test_pipeline_uses_injected_llm():
    calls = []
    def fake_llm(messages, temperature=0.0):
        calls.append(messages)
        return "grounded answer"

    store = VectorStore(HashingEmbedder(), persist_dir="/tmp/test-chroma-5")
    store.add_documents([Document("RAG grounds answers in documents.", source="docs")])
    pipe = RAGPipeline(store, llm_call=fake_llm)
    out = pipe.answer("What grounds answers?", k=2)
    assert out["answer"] == "grounded answer"
    assert calls and calls[0][0]["role"] == "user"


def test_pipeline_retrieve_returns_sources():
    store = VectorStore(HashingEmbedder(), persist_dir="/tmp/test-chroma-6")
    store.add_documents([Document("unique topic about vectors and retrieval", source="v")])
    pipe = RAGPipeline(store)
    hits = pipe.retrieve("unique topic retrieval", k=2)
    assert hits and all("source" in h and "text" in h for h in hits)
