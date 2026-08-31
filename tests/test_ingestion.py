from scholar_rag.ingestion import _chunk_id, chunks_from_text


def test_chunks_from_text_metadata():
    chunks = chunks_from_text("Hello world. " * 60, source="doc.txt", title="Doc",
                              chunk_size=120, overlap=10)
    assert len(chunks) >= 2
    assert all(c.source == "doc.txt" for c in chunks)
    assert all(c.title == "Doc" for c in chunks)
    # ids are unique
    assert len({c.id for c in chunks}) == len(chunks)


def test_chunk_id_is_deterministic():
    assert _chunk_id("a.pdf", 1, 0, "some text") == _chunk_id("a.pdf", 1, 0, "some text")
    assert _chunk_id("a.pdf", 1, 0, "x") != _chunk_id("a.pdf", 1, 1, "x")
