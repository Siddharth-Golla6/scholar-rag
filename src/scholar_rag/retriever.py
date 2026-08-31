"""Retrieval with optional Maximal Marginal Relevance (MMR) re-ranking.

MMR trades off relevance to the query against diversity among selected passages,
which reduces redundant near-duplicate chunks in the context window.
"""
from __future__ import annotations

import numpy as np

from .config import get_settings
from .embeddings import get_embedder
from .logging_utils import get_logger
from .schemas import RetrievedContext
from .vector_store import get_vector_store

log = get_logger("retriever")


def _mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, k: int,
         lambda_mult: float = 0.6) -> list[int]:
    sim_to_query = cand_vecs @ query_vec
    selected: list[int] = []
    remaining = list(range(len(cand_vecs)))
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda i: float(sim_to_query[i]))
        else:
            def mmr_score(i: int) -> float:
                redundancy = max(float(cand_vecs[i] @ cand_vecs[j]) for j in selected)
                return lambda_mult * float(sim_to_query[i]) - (1 - lambda_mult) * redundancy

            best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)
    return selected


class Retriever:
    def __init__(self, store=None, embedder=None, settings=None):
        self.settings = settings or get_settings()
        self.store = store or get_vector_store(self.settings)
        self.embedder = embedder or get_embedder()

    def retrieve(self, query: str, k: int | None = None, use_mmr: bool = True,
                 fetch_k: int | None = None) -> list[RetrievedContext]:
        k = k or self.settings.top_k
        fetch_k = fetch_k or max(k * 3, k)
        query_vec = self.embedder.encode(query, is_query=True)[0]
        candidates = self.store.query(query_vec, k=fetch_k)
        if not candidates:
            return []
        if not use_mmr or len(candidates) <= k:
            return candidates[:k]
        cand_vecs = self.embedder.encode([c.chunk.text for c in candidates])
        chosen = _mmr(np.asarray(query_vec, dtype="float32"), cand_vecs, k)
        return [candidates[i] for i in chosen]
