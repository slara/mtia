"""Unit tests for the chunker — no Ollama / Qdrant needed."""
from __future__ import annotations

from modules.rag.chunker import _pack, _split_paragraphs


def test_split_paragraphs_strips_whitespace_and_drops_empties():
    text = "  alpha  \n\n  beta  \n\n\n\ngamma\n\n"
    assert _split_paragraphs(text) == ["alpha", "beta", "gamma"]


def test_pack_groups_paragraphs_without_splitting_when_they_fit():
    paragraphs = ["a" * 100, "b" * 100, "c" * 100]
    chunks = _pack(paragraphs, chunk_chars=500, overlap_chars=0)
    assert len(chunks) == 1
    assert "a" * 100 in chunks[0]
    assert "c" * 100 in chunks[0]


def test_pack_flushes_when_budget_exceeded_and_seeds_overlap():
    paragraphs = ["a" * 300, "b" * 300, "c" * 300]
    chunks = _pack(paragraphs, chunk_chars=400, overlap_chars=50)
    # First chunk holds "a" para; second starts with overlap from first then "b", etc.
    assert len(chunks) >= 3
    assert chunks[0].startswith("a")
    # Overlap seeding: the start of chunk 2 carries tail of chunk 1.
    assert chunks[1][:50].endswith("a" * 50) or chunks[1].startswith("a")


def test_pack_hard_wraps_oversized_single_paragraph():
    big = "x" * 1000
    chunks = _pack([big], chunk_chars=300, overlap_chars=50)
    # 1000 chars, step = 250 → 4 chunks
    assert len(chunks) == 4
    assert all(len(c) <= 300 for c in chunks)
    # Consecutive chunks overlap by ~50 chars.
    assert chunks[0][-50:] == chunks[1][:50]
