"""PDF → text → paragraph-aware chunks.

Uses a character-based budget (not token-exact) because the Ollama HTTP API
does not expose a tokenizer and precision isn't load-bearing for retrieval
quality on manual-length text. 1 token ≈ 4 chars is a decent heuristic.

Default chunk ≈ 800 tokens ≈ 3200 chars, with 400-char overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


CHUNK_CHARS = 3200
OVERLAP_CHARS = 400


@dataclass
class Chunk:
    text: str
    page: int
    chunk_idx: int


def load_pdf_pages(path: str | Path) -> list[tuple[int, str]]:
    """Return [(page_number_1based, text), ...] for a PDF."""
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank-line paragraph boundaries, preserving order, trimming whitespace."""
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def _pack(paragraphs: list[str], chunk_chars: int, overlap_chars: int) -> list[str]:
    """Greedy pack paragraphs into chunks. If a single paragraph exceeds chunk_chars,
    hard-wrap it by character window. Overlap is applied by re-including the tail of
    the previous chunk when starting the next."""
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n\n".join(buf))
            buf = []
            buf_len = 0

    for para in paragraphs:
        if len(para) > chunk_chars:
            flush()
            # Hard-wrap oversized paragraph.
            step = chunk_chars - overlap_chars
            for start in range(0, len(para), step):
                chunks.append(para[start : start + chunk_chars])
            continue

        if buf_len + len(para) + 2 > chunk_chars and buf:
            flush()
            # Seed the next buffer with the overlap tail of the last chunk.
            if chunks and overlap_chars > 0:
                tail = chunks[-1][-overlap_chars:]
                buf = [tail]
                buf_len = len(tail)

        buf.append(para)
        buf_len += len(para) + 2

    flush()
    return chunks


def chunk_pdf(
    path: str | Path,
    chunk_chars: int = CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[Chunk]:
    """Read a PDF and return chunks annotated with their source page.

    A chunk is attributed to the page of its first paragraph. Chunks never
    span pages — we chunk each page independently, keeping page provenance
    clean for citations.
    """
    out: list[Chunk] = []
    chunk_idx = 0
    for page_num, page_text in load_pdf_pages(path):
        paragraphs = _split_paragraphs(page_text)
        if not paragraphs:
            continue
        for body in _pack(paragraphs, chunk_chars, overlap_chars):
            out.append(Chunk(text=body, page=page_num, chunk_idx=chunk_idx))
            chunk_idx += 1
    return out
