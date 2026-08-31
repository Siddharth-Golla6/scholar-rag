"""Prompt templates for grounded answering."""
from __future__ import annotations

from .schemas import RetrievedContext

RAG_SYSTEM = """You are ScholarRAG, a precise research assistant.
Answer the user's question using ONLY the numbered context passages provided.

Rules:
- Support every factual claim with a citation to the passage number in square
  brackets, e.g. [1] or [2][3].
- If the passages do not contain the answer, reply exactly:
  "I don't have enough information in the provided sources to answer that."
- Do NOT add methods, model names, datasets, metrics, or numbers that are not
  explicitly stated in the passages.
- Be concise and technical. Never invent citations, sources, or facts.
"""


def build_rag_prompt(question: str, contexts: list[RetrievedContext]) -> str:
    blocks = []
    for i, rc in enumerate(contexts, 1):
        loc = rc.chunk.source + (f" p.{rc.chunk.page}" if rc.chunk.page else "")
        blocks.append(f"[{i}] (source: {loc})\n{rc.chunk.text}")
    context_block = "\n\n".join(blocks) if blocks else "(no context retrieved)"
    return (
        f"Context passages:\n\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer (cite passages as [n]):"
    )
