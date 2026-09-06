"""Gradio UI for ScholarRAG — the entry point for Hugging Face Spaces (SDK: gradio).

Reuses the same scholar_rag backend as the Streamlit app (app/ui.py); this module
only provides a Gradio front-end so the project can run on HF Spaces' free tier.
"""
from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

import gradio as gr
import spaces

from scholar_rag.config import get_settings
from scholar_rag.ingestion import chunks_from_text, ingest_paths, seed_corpus_if_empty
from scholar_rag.vector_store import get_vector_store

settings = get_settings()

# Seed the corpus once at startup so a fresh Space isn't empty.
seed_corpus_if_empty(store=get_vector_store())

_rag = None
_agent = None


def _get_rag():
    global _rag
    if _rag is None:
        from scholar_rag.rag import RAGPipeline

        _rag = RAGPipeline()
    return _rag


def _get_agent():
    global _agent
    if _agent is None:
        from scholar_rag.agent import Agent

        _agent = Agent()
    return _agent


def status_md() -> str:
    n = get_vector_store().count()
    key = "🟢 answering enabled" if settings.has_llm_key else "🟡 retrieval only — set GROQ_API_KEY"
    return f"**Model** `{settings.llm_model}` &nbsp;·&nbsp; **{n} passages indexed** &nbsp;·&nbsp; {key}"


# ZeroGPU (the free HF Spaces GPU tier) requires at least one @spaces.GPU function
# to exist at startup. ScholarRAG does its real work on CPU — MiniLM embeddings plus
# an external LLM API — so it never needs the GPU for inference. This tiny probe only
# satisfies that startup check and absorbs ZeroGPU's periodic keep-warm ping (which
# calls the registered GPU function with arbitrary args); the real handler, ask(),
# runs on CPU in the main process, avoiding GPU forks, quota use and CUDA edge cases.
@spaces.GPU(duration=1)
def _gpu_probe(*args, **kwargs):
    return True


def ask(question: str, mode: str, top_k: int, do_eval: bool):
    if not question or not question.strip():
        return "Please enter a question.", "", "", ""
    use_agent = mode.startswith("Agent")
    try:
        if use_agent:
            answer = _get_agent().run(question)
        else:
            answer = _get_rag().answer(question, k=int(top_k))
    except Exception as exc:  # surface the real cause with a friendly hint
        msg = str(exc)
        low = msg.lower()
        if "401" in msg or "invalid_api_key" in low or "invalid api key" in low:
            return ("🔑 **GROQ_API_KEY is missing or invalid.** Set it in **Settings → "
                    "Variables and secrets** (paste the raw key, no quotes)."), "", "", ""
        if "429" in msg or "rate_limit" in low or "rate limit" in low:
            return "⏳ **Groq rate limit reached.** Wait a moment and try again.", "", "", ""
        if "404" in msg or "does not exist" in low or "model_not_found" in low:
            return f"🧩 **Model `{settings.llm_model}` unavailable.** Set a valid `LLM_MODEL` secret.", "", "", ""
        return f"**LLM call failed:** {msg}", "", "", ""

    meta = f"\n\n_route **{answer.route}** · confidence {answer.confidence:.2f} · model_ `{answer.model}`"
    answer_md = answer.answer + meta

    cites_md = ""
    if answer.citations:
        rows = []
        for c in answer.citations:
            loc = c.source + (f" p.{c.page}" if c.page else "")
            rows.append(f"- **[{c.marker}]** {loc}" if c.marker else f"- {loc}")
        cites_md = "### Citations\n" + "\n".join(rows)

    ctx_md = "\n\n".join(
        f"**[{i}]** `{rc.chunk.source}` — score {rc.score:.3f}\n\n> {rc.chunk.text[:300]}…"
        for i, rc in enumerate(answer.contexts, 1)
    )

    eval_md = ""
    if do_eval:
        try:
            from scholar_rag.evaluation import evaluate_answer

            s = evaluate_answer(answer)
            eval_md = (
                "### Answer quality (LLM-as-judge)\n"
                f"- Faithfulness **{s.faithfulness:.2f}**\n"
                f"- Answer relevance **{s.answer_relevance:.2f}**\n"
                f"- Context relevance **{s.context_relevance:.2f}**"
            )
            if s.faithfulness < 0.99 and s.reasoning.get("reason"):
                eval_md += f"\n\n_Judge note — {s.reasoning['reason']}_"
        except Exception as exc:
            eval_md = f"_evaluation failed: {exc}_"

    return answer_md, cites_md, ctx_md, eval_md


def ingest_uploaded(files):
    if not files:
        return status_md()
    paths = [p for p in files if str(p).lower().endswith((".pdf", ".txt", ".md"))]
    if paths:
        ingest_paths(paths)
    return status_md()


def fetch_arxiv(query: str):
    if not query or not query.strip():
        return status_md()
    from scholar_rag.embeddings import get_embedder
    from scholar_rag.tools.arxiv_search import search_arxiv

    papers = search_arxiv(query, max_results=5)
    chunks = []
    for p in papers:
        chunks += chunks_from_text(f"{p.title}\n\n{p.summary}", source=f"arXiv:{p.arxiv_id}", title=p.title)
    if chunks:
        store = get_vector_store()
        store.add(chunks, get_embedder().encode([c.text for c in chunks]))
    return status_md()


with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    title="ScholarRAG",
) as demo:
    gr.Markdown(
        "# 📚 ScholarRAG\n"
        "Ask your research library — **grounded, cited answers**, with a live **arXiv** "
        "fallback when your own papers fall short."
    )
    status = gr.Markdown(status_md())

    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="Question",
                placeholder="e.g. How do capsule networks and dynamic routing differ from CNNs?",
                lines=3,
            )
            with gr.Row():
                mode = gr.Radio(
                    ["RAG (local corpus)", "Agent (corpus + arXiv)"],
                    value="RAG (local corpus)",
                    label="Mode",
                )
                top_k = gr.Slider(2, 10, value=settings.top_k, step=1, label="Top-K passages")
                do_eval = gr.Checkbox(label="Evaluate answer (LLM-as-judge)")
            ask_btn = gr.Button("Ask", variant="primary")
            gr.Examples(
                examples=[
                    ["What deep-learning methods are used for pneumonia detection from chest X-rays?"],
                    ["How do capsule networks and dynamic routing differ from CNNs?"],
                    ["How does Grad-CAM make model predictions interpretable?"],
                ],
                inputs=question,
                label="Try one",
            )
        with gr.Column(scale=2):
            with gr.Accordion("Build your library", open=False):
                uploads = gr.File(
                    label="Upload PDFs", file_count="multiple", file_types=[".pdf"], type="filepath"
                )
                ingest_btn = gr.Button("Ingest uploaded")
                arxiv_q = gr.Textbox(label="…or fetch from arXiv", placeholder="capsule networks")
                arxiv_btn = gr.Button("Fetch & ingest")

    answer_out = gr.Markdown()
    cites_out = gr.Markdown()
    with gr.Accordion("Retrieved context", open=False):
        ctx_out = gr.Markdown()
    eval_out = gr.Markdown()

    ask_btn.click(ask, [question, mode, top_k, do_eval], [answer_out, cites_out, ctx_out, eval_out])
    ingest_btn.click(ingest_uploaded, [uploads], [status])
    arxiv_btn.click(fetch_arxiv, [arxiv_q], [status])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
