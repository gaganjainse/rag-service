"""FastAPI application for the RAG service.

Dependencies (embedder + vector store + pipeline) are created in the lifespan
and attached to app.state — no module-level singletons, so import no longer
loads a model or opens a ChromaDB handle, and tests can override state cleanly.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from .embeddings import make_embedder
from .rag import RAGPipeline
from .store import Document, VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    persist_dir = os.getenv("CHROMA_DIR", ".chroma")
    store = VectorStore(make_embedder(), persist_dir=persist_dir)
    app.state.store = store
    app.state.pipeline = RAGPipeline(store)
    yield


app = FastAPI(
    title="RAG Service",
    version="1.0.0",
    description="Hybrid-retrieval RAG API",
    lifespan=lifespan,
)


def _store(request: Request) -> VectorStore:
    return request.app.state.store


def _pipeline(request: Request) -> RAGPipeline:
    return request.app.state.pipeline


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
def health(request: Request) -> dict:
    return {"status": "ok", "documents": _store(request).count()}


@app.post("/ingest/text")
def ingest_text(req: IngestTextRequest, request: Request) -> dict:
    n = _store(request).add_documents([Document(req.text, source=req.source, metadata=req.metadata)])
    return {"ingested": n, "total": _store(request).count()}


@app.post("/ingest/file")
async def ingest_file(request: Request, file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="File must be UTF-8 text") from exc
    n = _store(request).add_documents([Document(text, source=file.filename or "upload")])
    return {"file": file.filename, "ingested": n, "total": _store(request).count()}


@app.post("/search")
def search(req: SearchRequest, request: Request) -> dict:
    return {"results": _store(request).search(req.query, k=req.k)}


@app.post("/ask")
def ask(req: AskRequest, request: Request) -> dict:
    return _pipeline(request).answer(req.question, k=req.k)


@app.get("/stats")
def stats(request: Request) -> dict:
    return {"documents": _store(request).count()}
