"""Vector store with hybrid retrieval (dense + keyword), merged with RRF."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

import chromadb

from .embeddings import HashingEmbedder  # for typing convenience


class Document:
    def __init__(
        self, text: str, source: str = "manual", metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.text = text
        self.source = source
        self.metadata = metadata or {}


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """Word-based chunking with overlap."""
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class VectorStore:
    """ChromaDB-backed store with hybrid search (vector + BM25-ish, RRF merge)."""

    def __init__(
        self,
        embedder,
        persist_dir: str = ".chroma",
        collection: str = "docs",
    ) -> None:
        self._embedder = embedder
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self, docs: List[Document], chunk_size: int = 500, overlap: int = 80
    ) -> int:
        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        n = 0
        for doc in docs:
            for chunk in chunk_text(doc.text, chunk_size, overlap):
                ids.append(f"{doc.source}:{n}")
                texts.append(chunk)
                metadatas.append({"source": doc.source, **doc.metadata})
                n += 1
        if texts:
            embeddings = self._embedder.encode(texts)
            self._col.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        return n

    def count(self) -> int:
        return self._col.count()

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Hybrid search: cosine (dense) + keyword (BM25-ish), RRF-fused."""
        if self.count() == 0:
            return []
        q_emb = self._embedder.encode([query])
        dense = self._col.query(
            query_embeddings=q_emb, n_results=max(k, 1), include=["documents", "metadatas", "distances"]
        )
        dense_docs = self._col.get(include=["documents", "metadatas"])
        all_docs = list(zip(dense_docs["ids"], dense_docs["documents"], dense_docs["metadatas"]))

        dense_scores: Dict[str, float] = {}
        for i, doc_id in enumerate(dense["ids"][0]):
            dense_scores[doc_id] = 1.0 - float(dense["distances"][0][i])

        # BM25-ish keyword scoring
        q_terms = _tokens(query)
        kw_scores: Dict[str, float] = {}
        for doc_id, text, meta in all_docs:
            tf = Counter(_tokens(text))
            score = sum(1.0 / (1.0 + abs(tf.get(t, 0) - 1)) for t in q_terms)
            kw_scores[doc_id] = score

        # Reciprocal Rank Fusion
        rrf: Dict[str, float] = {}
        for ranking in (dense_scores, kw_scores):
            for rank, doc_id in enumerate(sorted(ranking, key=ranking.get, reverse=True)[:k]):
                rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (60 + rank + 1)

        ranked = sorted(rrf, key=rrf.get, reverse=True)[:k]
        by_id = {doc_id: (text, meta) for doc_id, text, meta in all_docs}
        return [
            {
                "id": doc_id,
                "text": by_id[doc_id][0],
                "source": by_id[doc_id][1].get("source", "unknown"),
                "score": round(rrf[doc_id], 4),
            }
            for doc_id in ranked
            if doc_id in by_id
        ]
