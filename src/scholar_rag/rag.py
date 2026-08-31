"""The core RAG pipeline: retrieve -> prompt -> generate a grounded, cited answer."""
from __future__ import annotations

import re

from .config import get_settings
from .llm import get_llm
from .logging_utils import get_logger
from .prompts import RAG_SYSTEM, build_rag_prompt
from .retriever import Retriever
from .schemas import Answer, Citation, RetrievedContext

log = get_logger("rag")

# tolerant of ASCII [n] and full-width 【n】 brackets (some models emit the latter)
_CITE_RE = re.compile(r"[\[【](\d+)[\]】]")


def extract_citations(answer_text: str, contexts: list[RetrievedContext]) -> list[Citation]:
    """Map the [n] markers the model actually used back to their sources."""
    citations: list[Citation] = []
    for n in sorted({int(m) for m in _CITE_RE.findall(answer_text)}):
        if 1 <= n <= len(contexts):
            rc = contexts[n - 1]
            citations.append(
                Citation(marker=n, source=rc.chunk.source, page=rc.chunk.page,
                         title=rc.chunk.title, snippet=rc.chunk.text[:200])
            )
    return citations


class RAGPipeline:
    def __init__(self, retriever=None, llm=None, settings=None):
        self.settings = settings or get_settings()
        self.retriever = retriever or Retriever(settings=self.settings)
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def _messages(self, question: str, contexts: list[RetrievedContext]) -> list[dict]:
        return [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": build_rag_prompt(question, contexts)},
        ]

    def prepare(self, question: str, k: int | None = None):
        """Retrieve contexts and build chat messages (used by the streaming API)."""
        contexts = self.retriever.retrieve(question, k=k)
        return self._messages(question, contexts), contexts

    def answer(self, question: str, k: int | None = None) -> Answer:
        contexts = self.retriever.retrieve(question, k=k)
        confidence = contexts[0].score if contexts else 0.0
        text = self.llm.chat(self._messages(question, contexts)).strip()
        return Answer(
            question=question, answer=text, citations=extract_citations(text, contexts),
            contexts=contexts, route="corpus", confidence=confidence,
            model=self.settings.llm_model,
        )
