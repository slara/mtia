"""End-to-end ingestion pipeline test with Ollama stubbed.

Runs without Ollama by monkeypatching `embeddings.embed`. Qdrant must be up
(started by the compose stack).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from modules.rag import ingest, retrieve


def _write_pdf_with_reportlab(path: Path, pages_text: list[str]) -> None:
    from reportlab.pdfgen import canvas  # type: ignore
    c = canvas.Canvas(str(path))
    for text in pages_text:
        # Split each page into several paragraphs so we get multiple chunks.
        y = 800
        for para_idx, para in enumerate(text.split("\n\n")):
            c.drawString(72, y - para_idx * 20, para[:80])
        c.showPage()
    c.save()


def _fake_vec(seed: int, dim: int = 1024) -> list[float]:
    v = [0.0] * dim
    v[seed % dim] = 1.0
    return v


@pytest.fixture
def simple_pdf(tmp_path, monkeypatch):
    """Create a PDF on disk at the expected documents/<client>/<blueprint>/ path."""
    pytest.importorskip("reportlab")
    client = "testing_" + os.urandom(3).hex()
    blueprint = "empaque"
    docs_dir = tmp_path / "documents" / client / blueprint
    docs_dir.mkdir(parents=True)
    pdf_path = docs_dir / "manual.pdf"
    _write_pdf_with_reportlab(pdf_path, [
        "Primera pagina del manual de empaque.\n\nSegunda oracion con otra idea.",
        "Instrucciones detalladas para llenar las cajas correctamente.",
    ])
    monkeypatch.setenv("DOCUMENTS_DIR", str(tmp_path / "documents"))
    yield client, blueprint, pdf_path
    retrieve.drop_collection(client)


@pytest.fixture
def stubbed_embeddings(monkeypatch):
    """Replace Ollama embedding calls with a deterministic stub."""
    call_counter = {"n": 0}

    async def fake_embed(texts, model=None):
        out = []
        for t in texts:
            call_counter["n"] += 1
            out.append(_fake_vec(hash(t) & 0xFFFF))
        return out

    monkeypatch.setattr(ingest.embeddings, "embed", fake_embed)
    return call_counter


@pytest.mark.asyncio
async def test_ingest_file_chunks_embeds_and_upserts(simple_pdf, stubbed_embeddings):
    client, blueprint, pdf_path = simple_pdf

    stats = await ingest.ingest_file(client, blueprint, pdf_path)

    assert stats["source"] == "manual.pdf"
    assert stats["chunks"] >= 1
    assert stats["pages"] >= 1
    assert stubbed_embeddings["n"] == stats["chunks"]
    assert retrieve.count_by_source(client, blueprint, "manual.pdf") == stats["chunks"]


@pytest.mark.asyncio
async def test_ingest_is_idempotent_across_size_changes(simple_pdf, stubbed_embeddings):
    client, blueprint, pdf_path = simple_pdf

    # First ingest
    stats1 = await ingest.ingest_file(client, blueprint, pdf_path)
    n1 = stats1["chunks"]
    assert n1 >= 1

    # Rewrite the PDF shorter and re-ingest; old chunks must not linger.
    _write_pdf_with_reportlab(pdf_path, ["Pagina unica y corta."])
    stats2 = await ingest.ingest_file(client, blueprint, pdf_path)
    assert stats2["chunks"] >= 1

    total = retrieve.count_by_source(client, blueprint, "manual.pdf")
    assert total == stats2["chunks"], "stale chunks from first ingest should be gone"


@pytest.mark.asyncio
async def test_reindex_drops_and_rebuilds(simple_pdf, stubbed_embeddings):
    client, blueprint, pdf_path = simple_pdf

    await ingest.ingest_file(client, blueprint, pdf_path)

    # Drop the file on disk AFTER ingest; reindex should erase it from Qdrant.
    pdf_path.unlink()
    (pdf_path.parent / "other.pdf").write_bytes(b"")  # present but empty — will be skipped with warning
    _write_pdf_with_reportlab(pdf_path.parent / "other.pdf", ["Otro contenido"])

    stats = await ingest.reindex_client(client)
    sources = {s["source"] for s in stats}
    assert sources == {"other.pdf"}
    assert retrieve.count_by_source(client, blueprint, "manual.pdf") == 0
