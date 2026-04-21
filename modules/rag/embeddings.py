"""Ollama embeddings client.

Uses POST /api/embed (batch endpoint, Ollama >= 0.3). Returns plain lists of
floats so callers can hand them straight to Qdrant without numpy.
"""

from __future__ import annotations

import os

import httpx


def _ollama_url() -> str:
    return os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")


def _embed_model() -> str:
    return os.environ.get("EMBED_MODEL", "bge-m3")


async def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed a batch of strings. One HTTP call regardless of batch size."""
    if not texts:
        return []
    payload = {"model": model or _embed_model(), "input": texts}
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{_ollama_url()}/api/embed", json=payload)
        r.raise_for_status()
        data = r.json()
    return data["embeddings"]


async def embed_one(text: str, model: str | None = None) -> list[float]:
    vectors = await embed([text], model=model)
    return vectors[0]
