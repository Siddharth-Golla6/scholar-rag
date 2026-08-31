"""FastAPI service exposing ScholarRAG over HTTP, including token streaming (SSE)."""
from __future__ import annotations

import json
import pathlib
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from scholar_rag.agent import Agent  # noqa: E402
from scholar_rag.config import get_settings  # noqa: E402
from scholar_rag.evaluation import evaluate_answer  # noqa: E402
from scholar_rag.ingestion import ingest_paths  # noqa: E402
from scholar_rag.llm import LLMError  # noqa: E402
from scholar_rag.logging_utils import get_logger  # noqa: E402
from scholar_rag.rag import RAGPipeline, extract_citations  # noqa: E402
from scholar_rag.schemas import Answer  # noqa: E402
from scholar_rag.vector_store import get_vector_store  # noqa: E402

log = get_logger("api")


class AskRequest(BaseModel):
    question: str
    k: int | None = None
    use_agent: bool = False


class IngestRequest(BaseModel):
    paths: list[str]


_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["settings"] = get_settings()
    _state["rag"] = RAGPipeline()
    _state["agent"] = None
    from scholar_rag.ingestion import seed_corpus_if_empty

    seed_corpus_if_empty()  # populate a fresh deploy with starter docs
    log.info("ScholarRAG API ready (LLM key present: %s)", _state["settings"].has_llm_key)
    yield


app = FastAPI(title="ScholarRAG API", version="0.1.0", lifespan=lifespan)


def _agent() -> Agent:
    if _state.get("agent") is None:
        _state["agent"] = Agent()
    return _state["agent"]


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.get("/health")
def health():
    s = _state["settings"]
    return {
        "status": "ok",
        "provider": s.llm_provider,
        "model": s.llm_model,
        "llm_key_present": s.has_llm_key,
        "documents_indexed": get_vector_store(s).count(),
    }


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    try:
        if req.use_agent:
            return _agent().run(req.question)
        return _state["rag"].answer(req.question, k=req.k)
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        log.exception("ask failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    rag: RAGPipeline = _state["rag"]
    messages, contexts = rag.prepare(req.question, k=req.k)

    def generate():
        collected: list[str] = []
        try:
            for delta in rag.llm.stream(messages):
                collected.append(delta)
                yield _sse({"type": "token", "text": delta})
        except LLMError as exc:
            yield _sse({"type": "error", "message": str(exc)})
            return
        citations = [c.model_dump() for c in extract_citations("".join(collected), contexts)]
        ctx = [
            {"marker": i, "source": rc.chunk.source, "page": rc.chunk.page, "score": rc.score}
            for i, rc in enumerate(contexts, 1)
        ]
        yield _sse({"type": "done", "citations": citations, "contexts": ctx})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/ingest")
def ingest(req: IngestRequest):
    n = ingest_paths(req.paths)
    return {"ingested_chunks": n, "documents_indexed": get_vector_store(_state["settings"]).count()}


@app.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...)):
    s = _state["settings"]
    s.papers_dir.mkdir(parents=True, exist_ok=True)
    dest = s.papers_dir / (file.filename or "upload.pdf")
    dest.write_bytes(await file.read())
    n = ingest_paths([dest])
    return {"filename": dest.name, "ingested_chunks": n}


@app.post("/evaluate")
def evaluate(req: AskRequest):
    try:
        ans = _agent().run(req.question) if req.use_agent else _state["rag"].answer(req.question, k=req.k)
        scores = evaluate_answer(ans)
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"answer": ans.model_dump(), "scores": scores.model_dump()}
