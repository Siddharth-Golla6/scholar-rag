from scholar_rag.chunking import chunk_text


def test_empty_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_respects_chunk_size():
    text = "word " * 500  # ~2500 chars
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    # each chunk is bounded by chunk_size plus the prepended overlap
    assert all(len(c) <= 200 + 20 + 2 for c in chunks)


def test_overlap_creates_continuity():
    text = "A" * 150 + ". " + "B" * 150
    chunks = chunk_text(text, chunk_size=160, overlap=15)
    assert len(chunks) >= 2
    # the second chunk should begin with a tail of the first (overlap)
    assert chunks[1][:15] in chunks[0]
