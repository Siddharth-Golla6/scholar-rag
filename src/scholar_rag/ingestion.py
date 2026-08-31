"""Turn PDFs / text into embedded, retrievable chunks."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .chunking import chunk_text
from .config import get_settings
from .logging_utils import get_logger
from .schemas import Chunk

log = get_logger("ingestion")


def _chunk_id(source: str, page: int, idx: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}|{page}|{idx}|{text[:64]}".encode()).hexdigest()[:16]
    return f"{Path(source).stem}-p{page}-{idx}-{digest}"


def load_pdf_pages(path) -> tuple[str | None, list[tuple[int, str]]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    title = None
    try:
        if reader.metadata and reader.metadata.title:
            title = str(reader.metadata.title)
    except Exception:
        pass
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i + 1, text))
    return title, pages


def chunks_from_pdf(path, chunk_size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    s = get_settings()
    chunk_size = chunk_size or s.chunk_size
    overlap = s.chunk_overlap if overlap is None else overlap
    source = Path(path).name
    title, pages = load_pdf_pages(path)
    chunks: list[Chunk] = []
    for page_no, text in pages:
        for idx, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(
                Chunk(id=_chunk_id(source, page_no, idx, piece), text=piece,
                      source=source, page=page_no, title=title)
            )
    log.info("PDF '%s' -> %d chunks", source, len(chunks))
    return chunks


def chunks_from_text(text: str, source: str, title: str | None = None,
                     chunk_size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    s = get_settings()
    chunk_size = chunk_size or s.chunk_size
    overlap = s.chunk_overlap if overlap is None else overlap
    return [
        Chunk(id=_chunk_id(source, 0, idx, piece), text=piece, source=source, page=None, title=title)
        for idx, piece in enumerate(chunk_text(text, chunk_size, overlap))
    ]


def ingest_paths(paths, store=None, embedder=None) -> int:
    from .embeddings import get_embedder
    from .vector_store import get_vector_store

    store = store or get_vector_store()
    embedder = embedder or get_embedder()
    all_chunks: list[Chunk] = []
    for p in paths:
        p = Path(p)
        if p.suffix.lower() == ".pdf":
            all_chunks.extend(chunks_from_pdf(p))
        elif p.suffix.lower() in (".txt", ".md"):
            all_chunks.extend(
                chunks_from_text(p.read_text(encoding="utf-8", errors="ignore"), p.name)
            )
    if not all_chunks:
        log.warning("No chunks produced from %s", paths)
        return 0
    embeddings = embedder.encode([c.text for c in all_chunks])
    store.add(all_chunks, embeddings)
    log.info("Ingested %d chunks (store now holds %d)", len(all_chunks), store.count())
    return len(all_chunks)


def ingest_directory(directory=None) -> int:
    s = get_settings()
    directory = Path(directory) if directory else s.papers_dir
    paths = [p for p in directory.glob("**/*") if p.suffix.lower() in (".pdf", ".txt", ".md")]
    return ingest_paths(paths)
