"""RAG pipeline: retrieval + grounded answer generation with citation."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

import httpx

from .store import VectorStore

PROMPT_TEMPLATE = """You are a precise assistant. Answer the question using ONLY the
provided context. If the context does not contain the answer, say you don't know.
Cite the source of each claim in brackets, e.g. [source].

Context:
{context}

Question: {question}
Answer:"""


def default_llm_call(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    """Call an OpenAI-compatible chat endpoint. Falls back to a local echo."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        # Offline fallback: return the user message wrapped (for local demo/tests)
        return messages[-1]["content"] if messages else ""

    resp = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


class RAGPipeline:
    def __init__(
        self,
        store: VectorStore,
        llm_call: Callable[[List[Dict[str, str]], float], str] | None = None,
    ) -> None:
        self.store = store
        self.llm_call = llm_call or default_llm_call

    def retrieve(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        return self.store.search(question, k=k)

    def answer(self, question: str, k: int = 5) -> Dict[str, Any]:
        hits = self.retrieve(question, k=k)
        context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
        prompt = PROMPT_TEMPLATE.format(context=context or "(no context retrieved)", question=question)
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_call(messages, temperature=0.2)
        return {
            "question": question,
            "answer": response,
            "sources": [{"source": h["source"], "text": h["text"][:200]} for h in hits],
            "retrieved": len(hits),
        }
