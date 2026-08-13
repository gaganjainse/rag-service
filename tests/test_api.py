"""End-to-end API tests using a fast hashing embedder + offline LLM fallback."""

from __future__ import annotations

import os

os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["OPENAI_API_KEY"] = ""  # force offline fallback
os.environ["CHROMA_DIR"] = "/tmp/test-chroma-api"  # isolate from repo working dir

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    # `with` triggers the lifespan, which builds store + pipeline on app.state.
    with TestClient(app) as c:
        ids = c.app.state.store._col.get()["ids"]
        if ids:
            c.app.state.store._col.delete(ids=ids)
        yield c


SAMPLE = (
    "VIT Vellore is a university in India. The Computer Science program focuses on "
    "algorithms, systems, and artificial intelligence. Students build compilers and "
    "operating systems as part of the curriculum. The campus hosts research in "
    "large language models and retrieval-augmented generation."
)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ingest_text_and_stats(client):
    r = client.post("/ingest/text", json={"text": SAMPLE, "source": "vit"})
    assert r.status_code == 200
    assert r.json()["ingested"] > 0
    assert client.get("/stats").json()["documents"] == r.json()["total"]


def test_ingest_text_rejects_empty(client):
    r = client.post("/ingest/text", json={"text": ""})
    assert r.status_code == 422


def test_ingest_file(client, tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text(SAMPLE, encoding="utf-8")
    with f.open("rb") as fh:
        r = client.post("/ingest/file", files={"file": ("notes.txt", fh, "text/plain")})
    assert r.status_code == 200
    assert r.json()["file"] == "notes.txt"
    assert r.json()["ingested"] > 0


def test_ingest_file_rejects_binary(client, tmp_path):
    f = tmp_path / "bad.bin"
    f.write_bytes(b"\xff\xfe\x00\x01binary")
    with f.open("rb") as fh:
        r = client.post("/ingest/file", files={"file": ("bad.bin", fh, "application/octet-stream")})
    assert r.status_code == 400


def test_search_finds_relevant_chunk(client):
    client.post("/ingest/text", json={"text": SAMPLE, "source": "vit"})
    r = client.post("/search", json={"query": "compiler curriculum", "k": 3})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) > 0


def test_ask_uses_offline_fallback(client):
    client.post("/ingest/text", json={"text": SAMPLE, "source": "vit"})
    r = client.post("/ask", json={"question": "What do students build?", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body and "sources" in body
