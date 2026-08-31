from scholar_rag.rag import extract_citations
from scholar_rag.schemas import Chunk, RetrievedContext


def _contexts(n):
    return [
        RetrievedContext(
            chunk=Chunk(id=str(i), text=f"passage {i}", source=f"s{i}.pdf", page=i), score=1.0
        )
        for i in range(1, n + 1)
    ]


def test_extracts_used_markers_in_order():
    cits = extract_citations("Claim one [1]. Claim two [3][1].", _contexts(3))
    assert [c.marker for c in cits] == [1, 3]
    assert cits[0].source == "s1.pdf"
    assert cits[0].page == 1


def test_ignores_out_of_range_markers():
    assert extract_citations("see [9] and [0]", _contexts(2)) == []


def test_no_markers():
    assert extract_citations("no citations here", _contexts(2)) == []
