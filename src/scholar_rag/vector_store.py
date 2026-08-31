"""Pluggable vector store.

Two interchangeable backends behind one interface:
  * ``ChromaVectorStore`` — persistent ChromaDB (production-style, resume keyword).
  * ``NumpyVectorStore``  — zero-dependency cosine search, always available.

``get_vector_store`` picks Chroma when installed and falls back to NumPy, so the
app runs everywhere. Cosine similarity is used throughout (embeddings are
L2-normalised, so a dot product is the cosine).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .logging_utils import get_logger
from .schemas import Chunk, RetrievedContext

log = get_logger("vector_store")


class BaseVectorStore:
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...
    def query(self, query_embedding: np.ndarray, k: int = 5) -> list[RetrievedContext]: ...
    def count(self) -> int: ...
    def reset(self) -> None: ...


class NumpyVectorStore(BaseVectorStore):
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._emb_file = self.path / "embeddings.npy"
        self._meta_file = self.path / "chunks.pkl"
        self.embeddings: np.ndarray | None = None
        self.chunks: list[Chunk] = []
        self._load()

    def _load(self) -> None:
        if self._emb_file.exists() and self._meta_file.exists():
            self.embeddings = np.load(self._emb_file)
            with open(self._meta_file, "rb") as fh:
                self.chunks = pickle.load(fh)

    def _save(self) -> None:
        if self.embeddings is not None:
            np.save(self._emb_file, self.embeddings)
        with open(self._meta_file, "wb") as fh:
            pickle.dump(self.chunks, fh)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype="float32")
        self.embeddings = (
            embeddings if self.embeddings is None else np.vstack([self.embeddings, embeddings])
        )
        self.chunks.extend(chunks)
        self._save()

    def query(self, query_embedding: np.ndarray, k: int = 5) -> list[RetrievedContext]:
        if self.embeddings is None or not self.chunks:
            return []
        q = np.asarray(query_embedding, dtype="float32").reshape(-1)
        sims = self.embeddings @ q
        top = np.argsort(-sims)[:k]
        return [RetrievedContext(chunk=self.chunks[i], score=float(sims[i])) for i in top]

    def count(self) -> int:
        return len(self.chunks)

    def reset(self) -> None:
        self.embeddings = None
        self.chunks = []
        for f in (self._emb_file, self._meta_file):
            if f.exists():
                f.unlink()


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, path, collection: str = "scholar_rag"):
        import chromadb

        self.client = chromadb.PersistentClient(path=str(path))
        self._name = collection
        self.collection = self.client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype="float32")
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=[e.tolist() for e in embeddings],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source": c.source,
                    "page": c.page if c.page is not None else -1,
                    "title": c.title or "",
                }
                for c in chunks
            ],
        )

    def query(self, query_embedding: np.ndarray, k: int = 5) -> list[RetrievedContext]:
        q = np.asarray(query_embedding, dtype="float32").reshape(-1).tolist()
        res = self.collection.query(
            query_embeddings=[q],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievedContext] = []
        if not res.get("ids") or not res["ids"][0]:
            return out
        for _id, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            page = meta.get("page", -1)
            chunk = Chunk(
                id=_id,
                text=doc,
                source=meta.get("source", ""),
                page=None if page in (-1, None) else int(page),
                title=meta.get("title") or None,
            )
            out.append(RetrievedContext(chunk=chunk, score=1.0 - float(dist)))
        return out

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self._name)
        self.collection = self.client.get_or_create_collection(
            self._name, metadata={"hnsw:space": "cosine"}
        )


def get_vector_store(settings=None) -> BaseVectorStore:
    from .config import get_settings

    s = settings or get_settings()
    if s.vector_backend in ("auto", "chroma"):
        try:
            store = ChromaVectorStore(s.chroma_dir)
            log.info("Using ChromaDB vector store at %s", s.chroma_dir)
            return store
        except Exception as exc:
            if s.vector_backend == "chroma":
                raise
            log.warning("ChromaDB unavailable (%s); using NumpyVectorStore", exc)
    store = NumpyVectorStore(s.numpy_store_path)
    log.info("Using NumPy vector store at %s", s.numpy_store_path)
    return store
