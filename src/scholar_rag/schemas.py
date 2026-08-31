"""Typed data contracts used across the pipeline and the API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    text: str
    source: str  # filename or arXiv id
    page: Optional[int] = None
    title: Optional[str] = None
    extra: dict = Field(default_factory=dict)


class RetrievedContext(BaseModel):
    chunk: Chunk
    score: float


class Citation(BaseModel):
    marker: int  # the [n] used in the answer text
    source: str
    page: Optional[int] = None
    title: Optional[str] = None
    snippet: str = ""


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[RetrievedContext] = Field(default_factory=list)
    route: str = "corpus"  # corpus | arxiv | agent
    confidence: float = 0.0
    model: str = ""


class EvalScores(BaseModel):
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_relevance: float = 0.0
    reasoning: dict = Field(default_factory=dict)


class EvalItem(BaseModel):
    question: str
    answer: str
    scores: EvalScores


class EvalReport(BaseModel):
    n: int = 0
    mean_faithfulness: float = 0.0
    mean_answer_relevance: float = 0.0
    mean_context_relevance: float = 0.0
    items: list[EvalItem] = Field(default_factory=list)
