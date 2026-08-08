"""FastAPI application for the RAG service."""

from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .embeddings import make_embedder
from .rag import RAGPipeline
from .store import Document, VectorStore

app = FastAPI(title="RAG Service", version="1.0.0", description="Hybrid-retrieval RAG API")

_store = VectorStore(make_embedder())
_pipeline = RAGPipeline(_store)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)


class IngestTextRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str = "manual"
    metadata: dict = {}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents": _store.count()}


@app.post("/ingest/text")
def ingest_text(req: IngestTextRequest) -> dict:
    n = _store.add_documents([Document(req.text, source=req.source, metadata=req.metadata)])
    return {"ingested": n, "total": _store.count()}


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="File must be UTF-8 text") from exc
    n = _store.add_documents([Document(text, source=file.filename or "upload")])
    return {"file": file.filename, "ingested": n, "total": _store.count()}


@app.post("/search")
def search(req: SearchRequest) -> dict:
    return {"results": _store.search(req.query, k=req.k)}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return _pipeline.answer(req.question, k=req.k)


@app.get("/stats")
def stats() -> dict:
    return {"documents": _store.count()}
