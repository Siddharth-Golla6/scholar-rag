"""Recursive character text splitter with overlap.

Splits on progressively finer separators (paragraph -> line -> sentence -> word)
so chunks stay close to ``chunk_size`` without cutting mid-sentence where it can
be avoided. A small character overlap preserves context across boundaries.
"""
from __future__ import annotations

import re

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_recursive(text: str, chunk_size: int, seps: list[str]) -> list[str]:
    if len(text) <= chunk_size or not seps:
        return [text]
    sep = seps[0]
    parts = text.split(sep)
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else current + sep + part
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(part) > chunk_size:
            # part itself too big -> recurse with a finer separator
            chunks.extend(_split_recursive(part, chunk_size, seps[1:]))
            current = ""
        else:
            current = part
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Return a list of overlapping text chunks."""
    text = re.sub(r"[ \t]+", " ", text or "").strip()
    if not text:
        return []
    raw = _split_recursive(text, chunk_size, _SEPARATORS)
    result: list[str] = []
    for piece in raw:
        piece = piece.strip()
        if not piece:
            continue
        if overlap > 0 and result:
            tail = result[-1][-overlap:]
            piece = f"{tail} {piece}"
        result.append(piece)
    return result
