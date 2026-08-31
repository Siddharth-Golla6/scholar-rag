from scholar_rag.embeddings import HashingEmbedder
from scholar_rag.schemas import Chunk
from scholar_rag.vector_store import NumpyVectorStore


def _chunks():
    texts = ["cats and dogs are pets", "stock market finance investing", "deep neural network training"]
    return [Chunk(id=str(i), text=t, source="s.txt") for i, t in enumerate(texts)]


def test_add_query_and_persist(tmp_path):
    emb = HashingEmbedder()
    chunks = _chunks()
    store = NumpyVectorStore(tmp_path / "store")
    store.add(chunks, emb.encode([c.text for c in chunks]))
    assert store.count() == 3

    hits = store.query(emb.encode("finance investing stock")[0], k=1)
    assert hits and "finance" in hits[0].chunk.text

    # a fresh instance loads the persisted index
    reloaded = NumpyVectorStore(tmp_path / "store")
    assert reloaded.count() == 3

    reloaded.reset()
    assert reloaded.count() == 0


def test_query_empty_store(tmp_path):
    store = NumpyVectorStore(tmp_path / "empty")
    assert store.query(HashingEmbedder().encode("anything")[0], k=3) == []
