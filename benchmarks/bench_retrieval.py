"""Real hot-path benchmarks for rag-service (stdlib only, CI-safe).

Measures the retrieval pipeline: chunking + embedding + hybrid search
over an in-memory corpus. Median of N runs, loose bounds.
Run:  python benchmarks/bench_retrieval.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.embeddings import HashingEmbedder  # noqa: E402
from app.store import Document, VectorStore, chunk_text  # noqa: E402


def bench(label: str, fn, n: int = 200) -> float:
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    print(f"  {label:44s} median {med * 1e6:9.2f} µs  (n={n})")
    return med


def main() -> int:
    # Corpus: 20 documents × ~1.5KB → ~60 chunks.
    corpus = []
    for d in range(20):
        corpus.append(
            f"Document {d} about hybrid retrieval. " * 40
            + f"RAG and dense embeddings and BM25 keyword search. " * 20
        )

    import tempfile
    store = VectorStore(HashingEmbedder(), persist_dir=tempfile.mkdtemp(prefix="rag-bench-"))
    chunks = 0
    for doc in corpus:
        store.add_documents([Document(text=doc)])
        chunks += len(chunk_text(doc))
    print(f"  corpus: {len(corpus)} docs, {chunks} chunks")

    bench("chunk_text (500-char, 80 overlap)", lambda: chunk_text(corpus[0]), n=300)
    bench(f"hybrid search ({chunks} chunks)", lambda: store.search("hybrid retrieval rag", k=5), n=300)
    bench("embed (hashing, 256-dim)", lambda: HashingEmbedder().encode(["the cat sat"]), n=300)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
