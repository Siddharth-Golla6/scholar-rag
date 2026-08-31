"""arXiv search tool — queries the public arXiv Atom API using only the stdlib."""
from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from ..logging_utils import get_logger

log = get_logger("tools.arxiv")

ARXIV_API = "http://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom"}


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    summary: str
    published: str
    pdf_url: str = ""
    authors: list[str] = field(default_factory=list)


def search_arxiv(query: str, max_results: int = 5, timeout: int = 20) -> list[ArxivPaper]:
    params = urllib.parse.urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max(1, min(int(max_results), 10)),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    url = f"{ARXIV_API}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ScholarRAG/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except Exception as exc:  # network failures degrade gracefully
        log.warning("arXiv request failed: %s", exc)
        return []
    return _parse(data)


def _parse(data: bytes) -> list[ArxivPaper]:
    papers: list[ArxivPaper] = []
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        log.warning("arXiv parse error: %s", exc)
        return papers
    for entry in root.findall("a:entry", _NS):
        def text(tag: str) -> str:
            el = entry.find(f"a:{tag}", _NS)
            return (el.text or "").strip() if el is not None else ""

        raw_id = text("id")  # e.g. http://arxiv.org/abs/2504.12345v1
        pdf_url = ""
        for link in entry.findall("a:link", _NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
        authors = [
            (a.find("a:name", _NS).text or "").strip()
            for a in entry.findall("a:author", _NS)
            if a.find("a:name", _NS) is not None
        ]
        papers.append(
            ArxivPaper(
                arxiv_id=raw_id.rsplit("/", 1)[-1],
                title=" ".join(text("title").split()),
                summary=" ".join(text("summary").split()),
                published=text("published")[:10],
                pdf_url=pdf_url,
                authors=authors,
            )
        )
    return papers
