"""Ingestion pipeline: PDF → chunks → embeddings → Qdrant upsert.

CLI (runs inside the mtia container):
    python -m modules.rag.ingest --client cyt --blueprint empaque \\
        --path /app/documents/cyt/empaque/manual.pdf

    python -m modules.rag.ingest --reindex --client cyt

Idempotency: before upserting, we delete any existing chunks for
(client, blueprint_code, source). Re-ingesting the same file produces the
same end state whether the PDF got longer or shorter.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from modules.rag import chunker, embeddings, retrieve, storage
from modules.rag.retrieve import ChunkRecord


EMBED_BATCH = 32


async def ingest_file(client: str, blueprint_code: str, path: Path) -> dict:
    """Index one file. Returns stats dict."""
    if not path.is_file():
        raise FileNotFoundError(str(path))

    t0 = time.monotonic()
    source = path.name

    chunks = chunker.chunk_pdf(path)
    if not chunks:
        return {"source": source, "chunks": 0, "pages": 0, "seconds": 0.0,
                "warning": "no extractable text"}

    # Embed in batches to keep request payloads reasonable.
    vectors: list[list[float]] = []
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        vectors.extend(await embeddings.embed([c.text for c in batch]))

    records = [
        ChunkRecord(
            blueprint_code=blueprint_code,
            source=source,
            page=c.page,
            chunk_idx=c.chunk_idx,
            text=c.text,
            embedding=vec,
        )
        for c, vec in zip(chunks, vectors)
    ]

    # Replace-by-source semantics: drop stale chunks first.
    retrieve.delete_by_source(client, blueprint_code, source)
    retrieve.upsert_chunks(client, records)

    return {
        "source": source,
        "chunks": len(records),
        "pages": max(c.page for c in chunks),
        "seconds": round(time.monotonic() - t0, 2),
    }


async def reindex_client(client: str) -> list[dict]:
    """Drop the client's collection and re-ingest every file on disk."""
    retrieve.drop_collection(client)
    stats: list[dict] = []
    for blueprint_code in storage.list_blueprints(client):
        for doc in storage.list_documents(client, blueprint_code):
            stats.append(await ingest_file(client, blueprint_code, doc.path))
    return stats


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mtia-rag-ingest")
    p.add_argument("--client", required=True)
    p.add_argument("--blueprint", help="blueprint_code (required unless --reindex)")
    p.add_argument("--path", help="absolute path to PDF inside container")
    p.add_argument("--reindex", action="store_true",
                   help="drop collection and re-ingest all on-disk documents for client")
    args = p.parse_args(argv)

    if args.reindex:
        stats = asyncio.run(reindex_client(args.client))
        if not stats:
            print(f"No documents on disk for client={args.client}. "
                  f"DOCUMENTS_DIR={os.environ.get('DOCUMENTS_DIR')}")
            return 0
        total_chunks = sum(s["chunks"] for s in stats)
        print(f"Reindexed client={args.client}: {len(stats)} files, "
              f"{total_chunks} chunks")
        for s in stats:
            print(f"  {s['source']:40s} {s['chunks']:4d} chunks  "
                  f"{s['pages']:3d} pages  {s['seconds']:.2f}s"
                  + (f"  [{s['warning']}]" if s.get('warning') else ""))
        return 0

    if not args.blueprint or not args.path:
        p.error("--blueprint and --path are required unless --reindex")

    stats = asyncio.run(ingest_file(args.client, args.blueprint, Path(args.path)))
    print(f"Indexed {stats['source']}: {stats['chunks']} chunks, "
          f"{stats['pages']} pages, {stats['seconds']}s")
    if stats.get("warning"):
        print(f"  warning: {stats['warning']}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
