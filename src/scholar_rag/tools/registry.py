"""Tool definitions (OpenAI function-calling schema) and their implementations."""
from __future__ import annotations

from ..logging_utils import get_logger
from ..retriever import Retriever
from .arxiv_search import search_arxiv

log = get_logger("tools.registry")

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": (
                "Search the local library of ingested research papers for passages "
                "relevant to a query. Use this FIRST for any question that might be "
                "answered by the user's own papers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A focused search query."},
                    "k": {"type": "integer", "description": "Passages to return (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": (
                "Search arXiv.org for papers when the local library lacks the answer or "
                "the user asks for recent/related work. Returns titles, authors, abstracts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "description": "1-10 (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
]


class ToolRegistry:
    """Executes tool calls and remembers the last corpus hits for citation surfacing."""

    def __init__(self, retriever: Retriever | None = None):
        self.retriever = retriever or Retriever()
        self.last_corpus_contexts = []

    def run(self, name: str, arguments: dict) -> str:
        if name == "search_corpus":
            return self._search_corpus(**arguments)
        if name == "search_arxiv":
            return self._search_arxiv(**arguments)
        return f"Unknown tool: {name}"

    def _search_corpus(self, query: str, k: int = 5) -> str:
        contexts = self.retriever.retrieve(query, k=k)
        self.last_corpus_contexts = contexts
        if not contexts:
            return "No relevant passages found in the local corpus."
        blocks = []
        for i, rc in enumerate(contexts, 1):
            loc = rc.chunk.source + (f" p.{rc.chunk.page}" if rc.chunk.page else "")
            blocks.append(f"[corpus:{i}] (source: {loc}, score={rc.score:.2f})\n{rc.chunk.text}")
        return "\n\n".join(blocks)

    def _search_arxiv(self, query: str, max_results: int = 5) -> str:
        papers = search_arxiv(query, max_results=max_results)
        if not papers:
            return "No results from arXiv (or the network is unavailable)."
        blocks = []
        for p in papers:
            authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
            blocks.append(
                f"[arxiv:{p.arxiv_id}] {p.title} ({p.published})\n{authors}\n{p.summary[:600]}"
            )
        return "\n\n".join(blocks)
