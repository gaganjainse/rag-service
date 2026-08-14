"""Regression guard: ChromaDB must stay embedded (PersistentClient), never the
HTTP server.

CVE-2026-45829 (PYSEC-2026-311, GHSA-f4j7-r4q5-qw2c) is a pre-authentication
code-injection in chromadb's HTTP /api/v2 collection endpoint, EPSS 0.124
(actively exploited). It is only reachable when chromadb runs its HTTP server
(HttpClient / `chroma run`). This test pins the invariant that makes us safe:
VectorStore never instantiates chromadb.HttpClient.
"""
import unittest.mock

import chromadb

from app.store import VectorStore


def _fake_embed(texts):
    return [[0.0, 0.0] for _ in texts]


def test_store_never_uses_http_client():
    with unittest.mock.patch.object(
        chromadb, "HttpClient",
        side_effect=AssertionError("VectorStore must never use chromadb.HttpClient"),
    ):
        store = VectorStore(_fake_embed, persist_dir=None)
    assert store._client is not None
