"""Smoke tests for Ollama clients. Skipped when Ollama isn't reachable."""
from __future__ import annotations

import os

import httpx
import pytest

from modules.rag import embeddings, llm


def _ollama_reachable() -> bool:
    try:
        r = httpx.get(f"{os.environ.get('OLLAMA_URL', 'http://host.docker.internal:11434')}/api/tags", timeout=1)
        return r.status_code == 200
    except Exception:
        return False


needs_ollama = pytest.mark.skipif(
    not _ollama_reachable(), reason="Ollama not reachable — run `invoke mtia.pull-models` on the host first")


@needs_ollama
@pytest.mark.asyncio
async def test_embed_returns_expected_dim():
    vec = await embeddings.embed_one("hola mundo")
    assert isinstance(vec, list)
    assert len(vec) == 1024
    assert all(isinstance(x, float) for x in vec[:5])


@needs_ollama
@pytest.mark.asyncio
async def test_embed_batch_parallel_shape():
    vectors = await embeddings.embed(["a", "b", "c"])
    assert len(vectors) == 3
    assert len(vectors[0]) == len(vectors[1]) == len(vectors[2])


@needs_ollama
@pytest.mark.asyncio
async def test_chat_stream_yields_tokens():
    tokens: list[str] = []
    async for t in llm.chat_stream(
        [{"role": "user", "content": "di hola en una palabra"}],
        options={"num_predict": 16},
    ):
        tokens.append(t)
        if len("".join(tokens)) > 20:
            break
    assert tokens, "expected at least one token"
