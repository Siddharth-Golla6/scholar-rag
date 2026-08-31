"""Embedding backends.

``SentenceTransformerEmbedder`` is the real, CPU-friendly encoder
(all-MiniLM-L6-v2, 384-dim). ``HashingEmbedder`` is a deterministic, no-download
fallback used for offline/CI tests and if the model can't be loaded.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

from .config import get_settings
from .logging_utils import get_logger

log = get_logger("embeddings")


class BaseEmbedder:
    dim: int

    def encode(self, texts, is_query: bool = False) -> np.ndarray:  # noqa: D401
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model '%s' (first run downloads ~90MB)", model_name)
        self.model = SentenceTransformer(model_name)
        try:  # method renamed in newer sentence-transformers
            self.dim = self.model.get_embedding_dimension()
        except AttributeError:
            self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts, is_query: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        emb = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return emb.astype("float32")


class HashingEmbedder(BaseEmbedder):
    """Deterministic bag-of-words hashing embedder — no network, for tests."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts, is_query: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vecs = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for token in str(text).lower().split():
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                vecs[i, bucket] += 1.0
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] /= norm
        return vecs


@lru_cache(maxsize=2)
def get_embedder(offline: bool = False) -> BaseEmbedder:
    if offline:
        return HashingEmbedder()
    settings = get_settings()
    try:
        return SentenceTransformerEmbedder(settings.embedding_model)
    except Exception as exc:  # pragma: no cover - depends on environment
        log.warning("Embedding model unavailable (%s); using HashingEmbedder", exc)
        return HashingEmbedder()
