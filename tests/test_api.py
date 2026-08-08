"""End-to-end API tests using a fast hashing embedder + offline LLM fallback."""

from __future__ import annotations

import os

os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["OPENAI_API_KEY"] = ""  # force offline fallback

import pytest
from fastapi.testclient import TestClient

from app.embeddings import HashingEmbedder
from app.main import _pipeline, _store, app
from app.rag import RAGPipeline
from app.store import Document

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_store():
    ids = _store._col.get()["ids"]
    if ids:
        _store._col.delete(ids=ids)
    yield


SAMPLE = (
    "VIT Vellore is a university in India. The Computer Science program focuses on "
    "algorithms, systems, and artificial intelligence. Students build compilers and "
    "operating systems as part of the curriculum. The campus hosts research in "
    "large language models and retrieval-augmented generation."
)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ingest_text_and_stats():
    r = client.post("/ingest/text", json={"text": SAMPLE, "source": "vit"})
    assert r.status_code == 200
    assert r.json()["ingested"] > 0
    assert client.get("/stats").json()["documents"] == r.json()["total"]


def test_ingest_text_rejects_empty():
    r = client.post("/ingest/text", json={"text": ""})
    assert r.status_code == 422


def test_ingest_file(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text(SAMPLE, encoding="utf-8")
    with f.open("rb") as fh:
        r = client.post("/ingest/file", files={"file": ("notes.txt", fh, "text/plain")})
    assert r.status_code == 200
    assert r.json()["file"] == "notes.txt"
    assert r.json()["ingested"] > 0


def test_ingest_file_rejects_binary(tmp_path):
    f = tmp_path / "bad.bin"
    f.write_bytes(b"\xff\xfe\x00\x01binary")
    with f.open("rb") as fh:
        r = client.post("/ingest/file", files={"file": ("bad.bin", fh, "application/octet-stream")})
    assert r.status_code == 400


def test_search_finds_relevant_chunk():
    client.post("/ingest/text", json={"text": SAMPLE, "source": "vit"})
    r = client.post("/search", json={"query": "compiler curriculum", "k": 3})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) > 0
    assert all("text" in res and "source" in res for res in results)


def test_search_empty_store():
    r = client.post("/search", json={"query": "anything", "k": 3})
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_ask_returns_answer_and_sources():
    client.post("/ingest/text", json={"text": SAMPLE, "source": "vit"})
    r = client.post("/ask", json={"question": "What does the CS program focus on?", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["question"]
    assert body["answer"]
    assert body["sources"]
    assert all(s["source"] for s in body["sources"])


def test_ask_empty_store_still_returns_answer():
    r = client.post("/ask", json={"question": "hello"})
    assert r.status_code == 200
    assert r.json()["retrieved"] == 0


def test_ask_validation():
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "q", "k": 0}).status_code == 422


def test_rag_pipeline_offline():
    store = _store
    pipeline = RAGPipeline(store)
    store.add_documents([Document(SAMPLE, source="vit")])
    out = pipeline.answer("What is VIT?", k=2)
    assert out["answer"] and out["retrieved"] >= 1


def test_chunk_text_overlap():
    from app.store import chunk_text

    text = " ".join(f"word{i}" for i in range(60))
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) >= 3
    # overlap means consecutive chunks share words
    c1, c2 = set(chunks[0].split()), set(chunks[1].split())
    assert len(c1 & c2) > 0


def test_hybrid_search_rrf_ranks(tmp_path):
    from app.store import VectorStore

    embedder = HashingEmbedder()
    store = VectorStore(embedder, persist_dir=str(tmp_path / "chroma"))
    store.add_documents(
        [
            Document("The cat sat on the mat. Cats like fish and milk.", source="cat"),
            Document("Rust is a systems programming language with ownership.", source="rust"),
            Document("Retrieval augmented generation grounds answers in documents.", source="rag"),
        ]
    )
    results = store.search("cat on the mat", k=2)
    assert results[0]["source"] == "cat"
