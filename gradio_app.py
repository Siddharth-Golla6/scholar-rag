"""Gradio UI for ScholarRAG — the entry point for Hugging Face Spaces (SDK: gradio).

Reuses the same scholar_rag backend as the Streamlit app (app/ui.py); this module
only provides a Gradio front-end so the project can run on HF Spaces' free tier.

The visual design is a "research-journal" aesthetic: a deep-navy canvas, an emerald
"grounded / verified" accent, a Crimson Pro serif display face paired with Inter for
UI text. All of it is delivered through a custom theme + injected CSS so the app does
not look like a stock Gradio demo.
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


# --- small inline icons (Lucide-style; no emoji-as-icons) --------------------
_IC_CHIP = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
    'stroke-linecap="round" stroke-linejoin="round"><rect x="7" y="7" width="10" '
    'height="10" rx="2"/><path d="M9 3v2M12 3v2M15 3v2M9 19v2M12 19v2M15 19v2M3 9h2'
    'M3 12h2M3 15h2M19 9h2M19 12h2M19 15h2"/></svg>'
)
_IC_DB = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
    'stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="6" rx="7" '
    'ry="3"/><path d="M5 6v6c0 1.66 3.13 3 7 3s7-1.34 7-3V6"/><path d="M5 12v6c0 '
    '1.66 3.13 3 7 3s7-1.34 7-3v-6"/></svg>'
)


def status_html() -> str:
    n = get_vector_store().count()
    enabled = settings.has_llm_key
    if enabled:
        key_pill = (
            '<span class="sr-pill sr-pill--ok"><span class="sr-dot"></span>'
            "Answering enabled</span>"
        )
    else:
        key_pill = (
            '<span class="sr-pill sr-pill--warn"><span class="sr-dot"></span>'
            "Retrieval only · set GROQ_API_KEY</span>"
        )
    return (
        '<div class="sr-pills">'
        f'<span class="sr-pill"><span class="sr-pico">{_IC_CHIP}</span>'
        f"<span class=\"sr-mono\">{settings.llm_model}</span></span>"
        f'<span class="sr-pill"><span class="sr-pico">{_IC_DB}</span>'
        f"{n} passages indexed</span>"
        f"{key_pill}"
        "</div>"
    )


_EMPTY_ANSWER = (
    '<div class="sr-empty">'
    '<div class="sr-empty__mark">“ ”</div>'
    "<p class=\"sr-empty__t\">Your grounded answer will appear here.</p>"
    '<p class="sr-empty__s">Ask a question and ScholarRAG retrieves the most relevant '
    "passages, then answers with inline&nbsp;<b>[n]</b>&nbsp;citations you can trace "
    "back to the source.</p>"
    "</div>"
)


# ZeroGPU (the free HF Spaces GPU tier) requires at least one @spaces.GPU function
# to exist at startup. ScholarRAG does its real work on CPU — MiniLM embeddings plus
# an external LLM API — so it never needs the GPU for inference. This tiny probe only
# satisfies that startup check and absorbs ZeroGPU's periodic keep-warm ping (which
# calls the registered GPU function with arbitrary args); the real handler, ask(),
# runs on CPU in the main process, avoiding GPU forks, quota use and CUDA edge cases.
@spaces.GPU(duration=1)
def _gpu_probe(*args, **kwargs):
    return True


_ROUTE_LABEL = {"corpus": "local corpus", "agent": "agent", "arxiv": "arXiv"}


def _meta_html(answer) -> str:
    route = _ROUTE_LABEL.get(answer.route, answer.route)
    bits = [f'<span class="sr-metabit">route&nbsp;<b>{route}</b></span>']
    conf = getattr(answer, "confidence", 0) or 0
    if conf > 0.01:
        bits.append(f'<span class="sr-metabit">confidence&nbsp;<b>{conf:.0%}</b></span>')
    bits.append(f'<span class="sr-metabit">model&nbsp;<code>{answer.model}</code></span>')
    return '<div class="sr-meta">' + '<span class="sr-metasep">·</span>'.join(bits) + "</div>"


def _err(msg: str):
    return (
        msg,
        gr.update(value="", visible=False),
        gr.update(value="", visible=False),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(value="", visible=False),
    )


def ask(question: str, mode: str, top_k: int, do_eval: bool):
    if not question or not question.strip():
        return _err("Please enter a question to begin.")
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
            return _err(
                "🔑 **GROQ_API_KEY is missing or invalid.** Set it in **Settings → "
                "Variables and secrets** (paste the raw key, no quotes)."
            )
        if "429" in msg or "rate_limit" in low or "rate limit" in low:
            return _err("⏳ **Groq rate limit reached.** Wait a moment and try again.")
        if "404" in msg or "does not exist" in low or "model_not_found" in low:
            return _err(f"🧩 **Model `{settings.llm_model}` unavailable.** Set a valid `LLM_MODEL` secret.")
        return _err(f"**LLM call failed:** {msg}")

    answer_md = answer.answer
    meta_html = _meta_html(answer)

    cites_md = ""
    if answer.citations:
        rows = []
        for c in answer.citations:
            loc = c.source + (f" · p.{c.page}" if c.page else "")
            rows.append(f"- **[{c.marker}]**&nbsp; {loc}" if c.marker else f"- {loc}")
        cites_md = "### Citations\n" + "\n".join(rows)

    ctx_md = "\n\n".join(
        f"**[{i}]** `{rc.chunk.source}`"
        + (f" — relevance {rc.score:.2f}" if getattr(rc, "score", 0) and rc.score > 0 else "")
        + f"\n\n> {rc.chunk.text[:300].strip()}…"
        for i, rc in enumerate(answer.contexts, 1)
    )

    eval_html = ""
    if do_eval:
        try:
            from scholar_rag.evaluation import evaluate_answer

            s = evaluate_answer(answer)

            def _card(v, k):
                return (
                    f'<div class="sr-metric"><div class="sr-metric__v">{v:.2f}</div>'
                    f'<div class="sr-metric__k">{k}</div></div>'
                )

            eval_html = (
                '<div class="sr-evalhead">Answer quality '
                '<span class="sr-evaltag">LLM-as-judge</span></div>'
                '<div class="sr-metrics">'
                + _card(s.faithfulness, "Faithfulness")
                + _card(s.answer_relevance, "Answer relevance")
                + _card(s.context_relevance, "Context relevance")
                + "</div>"
            )
            if s.faithfulness < 0.99 and s.reasoning.get("reason"):
                eval_html += f'<div class="sr-judgenote">{s.reasoning["reason"]}</div>'
        except Exception as exc:
            eval_html = f'<div class="sr-judgenote">evaluation failed: {exc}</div>'

    return (
        answer_md,
        gr.update(value=meta_html, visible=True),
        gr.update(value=cites_md, visible=bool(cites_md)),
        gr.update(visible=bool(ctx_md)),
        gr.update(value=ctx_md),
        gr.update(value=eval_html, visible=bool(eval_html)),
    )


def ingest_uploaded(files):
    if not files:
        return status_html()
    paths = [p for p in files if str(p).lower().endswith((".pdf", ".txt", ".md"))]
    if paths:
        ingest_paths(paths)
    return status_html()


def fetch_arxiv(query: str):
    if not query or not query.strip():
        return status_html()
    from scholar_rag.embeddings import get_embedder
    from scholar_rag.tools.arxiv_search import search_arxiv

    papers = search_arxiv(query, max_results=5)
    chunks = []
    for p in papers:
        chunks += chunks_from_text(f"{p.title}\n\n{p.summary}", source=f"arXiv:{p.arxiv_id}", title=p.title)
    if chunks:
        store = get_vector_store()
        store.add(chunks, get_embedder().encode([c.text for c in chunks]))
    return status_html()


# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#0B1020",
    body_background_fill_dark="#0B1020",
    body_text_color="#E6EAF3",
    body_text_color_dark="#E6EAF3",
    background_fill_primary="#131C33",
    background_fill_primary_dark="#131C33",
    border_color_primary="rgba(148,163,184,.14)",
    block_background_fill="transparent",
    block_border_width="0px",
    block_shadow="none",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --bg:#0B1020; --surface:#131C33; --surface-2:#18213E;
  --border:rgba(148,163,184,.14); --border-2:rgba(148,163,184,.26);
  --text:#E7ECF6; --muted:#9AA6BD; --faint:#6B7896;
  --accent:#34D399; --accent-2:#22D3EE; --accent-ink:#04160F;
  --warn:#FBBF24; --radius:16px;
  --shadow:0 18px 44px -20px rgba(2,8,23,.75);
}

/* ---- canvas & container ---- */
gradio-app{background:transparent!important;}
body,.gradio-container{
  background:
    radial-gradient(1100px 560px at 12% -8%, rgba(52,211,153,.10), transparent 58%),
    radial-gradient(950px 480px at 108% -4%, rgba(34,211,238,.09), transparent 55%),
    var(--bg)!important;
  color:var(--text);
  font-family:'Inter',system-ui,sans-serif;
}
.gradio-container{max-width:960px!important;margin:0 auto!important;padding:8px 18px 72px!important;}
footer{display:none!important;}
.sr-mono,code{font-family:'JetBrains Mono',ui-monospace,monospace;}

/* ---- hero ---- */
.sr-hero{padding:30px 2px 10px;}
.sr-eyebrow{
  display:inline-flex;align-items:center;gap:8px;
  font-size:12px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);
  background:rgba(52,211,153,.08);border:1px solid rgba(52,211,153,.22);
  padding:6px 12px;border-radius:999px;margin-bottom:18px;
}
.sr-eyebrow .sr-spark{width:7px;height:7px;border-radius:50%;background:var(--accent);
  box-shadow:0 0 0 4px rgba(52,211,153,.18);}
.sr-title{
  font-family:'Crimson Pro',Georgia,serif;font-weight:600;
  font-size:clamp(44px,7vw,68px);line-height:1.02;letter-spacing:-.015em;margin:0;
  background:linear-gradient(180deg,#F4F7FF 0%,#B9C6E4 100%);
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
}
.sr-title .sr-dotmark{color:var(--accent);-webkit-text-fill-color:var(--accent);}
.sr-tag{color:var(--muted);font-size:17px;line-height:1.6;max-width:60ch;margin:14px 0 20px;}
.sr-tag b{color:var(--text);font-weight:600;}

/* ---- status pills ---- */
.sr-pills{display:flex;flex-wrap:wrap;gap:8px;}
.sr-pill{
  display:inline-flex;align-items:center;gap:8px;
  font-size:12.5px;font-weight:500;color:var(--muted);
  background:var(--surface);border:1px solid var(--border);
  padding:7px 12px;border-radius:999px;
}
.sr-pill .sr-mono{color:var(--text);font-size:12px;}
.sr-pico{display:inline-flex;width:14px;height:14px;color:var(--accent-2);}
.sr-pico svg{width:14px;height:14px;}
.sr-pill--ok{color:#B7F7DA;border-color:rgba(52,211,153,.34);background:rgba(52,211,153,.10);}
.sr-pill--warn{color:#FCE9BE;border-color:rgba(251,191,36,.34);background:rgba(251,191,36,.10);}
.sr-dot{width:8px;height:8px;border-radius:50%;background:var(--accent);
  box-shadow:0 0 0 4px rgba(52,211,153,.16);animation:srpulse 2.4s ease-in-out infinite;}
.sr-pill--warn .sr-dot{background:var(--warn);box-shadow:0 0 0 4px rgba(251,191,36,.16);}
@keyframes srpulse{0%,100%{opacity:1}50%{opacity:.45}}

/* ---- composer card ---- */
.sr-card{
  background:linear-gradient(180deg,rgba(24,33,62,.72),rgba(19,28,51,.72))!important;
  border:1px solid var(--border)!important;border-radius:var(--radius)!important;
  padding:18px!important;margin-top:22px!important;box-shadow:var(--shadow)!important;
  backdrop-filter:blur(6px);
}
.sr-card .block{background:transparent!important;}

/* labels */
.gradio-container span[data-testid="block-label"],
.gradio-container .block-label,
.gradio-container label > span{
  color:var(--muted)!important;font-size:12.5px!important;font-weight:600!important;
  letter-spacing:.02em;
}

/* textarea / inputs */
.gradio-container textarea,
.gradio-container input[type=text],
.gradio-container input[type=number]{
  background:#0E1730!important;color:var(--text)!important;
  border:1px solid var(--border)!important;border-radius:12px!important;
  font-size:16px!important;line-height:1.55!important;padding:14px 15px!important;
  transition:border-color .18s ease,box-shadow .18s ease;
}
#sr-q textarea{min-height:96px!important;font-size:17px!important;}
.gradio-container textarea::placeholder,
.gradio-container input::placeholder{color:var(--faint)!important;}
.gradio-container textarea:focus,
.gradio-container input[type=text]:focus,
.gradio-container input[type=number]:focus{
  border-color:rgba(52,211,153,.55)!important;
  box-shadow:0 0 0 3px rgba(52,211,153,.18)!important;outline:none!important;
}

/* mode radio -> segmented control */
.sr-seg fieldset,.sr-seg .wrap{display:flex!important;gap:8px!important;border:0!important;
  background:transparent!important;padding:0!important;}
.sr-seg label{
  cursor:pointer;border:1px solid var(--border)!important;background:#0E1730!important;
  border-radius:11px!important;padding:9px 14px!important;color:var(--muted)!important;
  font-weight:500;font-size:13.5px;transition:all .16s ease;margin:0!important;
  display:flex;align-items:center;gap:8px;
}
.sr-seg label:hover{border-color:var(--border-2)!important;color:var(--text)!important;}
.sr-seg label:has(input:checked){
  border-color:rgba(52,211,153,.55)!important;color:#DAFCEC!important;
  background:rgba(52,211,153,.12)!important;box-shadow:0 0 0 3px rgba(52,211,153,.10);
}
.sr-seg input[type=radio]{accent-color:var(--accent);}
.sr-seg label input{position:absolute;opacity:0;width:0;height:0;}
.sr-seg label::before{content:"";width:8px;height:8px;border-radius:50%;
  border:1.5px solid var(--faint);transition:all .16s ease;}
.sr-seg label:has(input:checked)::before{border-color:var(--accent);background:var(--accent);
  box-shadow:0 0 0 3px rgba(52,211,153,.22);}

/* slider + checkbox use native accent */
.sr-slider input[type=range]{accent-color:var(--accent);}
.sr-check input[type=checkbox]{accent-color:var(--accent);width:17px;height:17px;}

/* Ask button */
#sr-ask{margin-top:4px!important;}
#sr-ask button,#sr-ask{
  background:linear-gradient(180deg,#3CE6A6,#17B981)!important;color:var(--accent-ink)!important;
  font-weight:700!important;font-size:15.5px!important;letter-spacing:.01em;
  border:0!important;border-radius:12px!important;padding:14px 18px!important;
  box-shadow:0 12px 28px -10px rgba(23,185,129,.6)!important;
  transition:transform .12s ease,box-shadow .18s ease,filter .18s ease;cursor:pointer;
}
#sr-ask button:hover{transform:translateY(-1px);filter:brightness(1.04);
  box-shadow:0 18px 34px -10px rgba(23,185,129,.72)!important;}
#sr-ask button:active{transform:translateY(0);}

/* example chips */
.sr-examples{margin-top:14px!important;}
.sr-examples .label-wrap,.sr-examples > .block > span:first-child{color:var(--faint)!important;
  font-size:11.5px!important;text-transform:uppercase;letter-spacing:.14em;}
.sr-examples button,.sr-examples td,.sr-examples .gr-sample-textbox{
  background:transparent!important;border:1px solid var(--border)!important;
  color:var(--muted)!important;border-radius:999px!important;padding:8px 14px!important;
  font-size:13px!important;cursor:pointer;transition:all .16s ease;
}
.sr-examples button:hover,.sr-examples td:hover{
  border-color:rgba(52,211,153,.45)!important;color:var(--text)!important;
  background:rgba(52,211,153,.08)!important;}

/* accordions */
.gradio-container .label-wrap{color:var(--text)!important;font-weight:600!important;}
.sr-lib,.sr-ctx{border:1px solid var(--border)!important;border-radius:14px!important;
  background:rgba(19,28,51,.55)!important;margin-top:14px!important;overflow:hidden;}
.sr-lib .label-wrap,.sr-ctx .label-wrap{padding:14px 16px!important;}
.sr-lib button{border-radius:10px!important;}

/* ---- answer ---- */
.sr-answer{
  margin-top:26px!important;background:var(--surface)!important;
  border:1px solid var(--border)!important;border-left:3px solid var(--accent)!important;
  border-radius:var(--radius)!important;padding:24px 26px!important;box-shadow:var(--shadow)!important;
}
.sr-answer p,.sr-answer li{font-size:16.5px!important;line-height:1.78!important;color:#E9EEF8!important;}
.sr-answer h1,.sr-answer h2,.sr-answer h3{font-family:'Crimson Pro',serif!important;
  color:#F3F6FF!important;letter-spacing:-.01em;}
.sr-answer strong{color:#FFFFFF;}
/* inline [n] citation markers */
.sr-answer p{}
.sr-empty{text-align:center;padding:20px 8px 6px;}
.sr-empty__mark{font-family:'Crimson Pro',serif;font-size:52px;line-height:1;color:rgba(52,211,153,.5);
  margin-bottom:6px;}
.sr-empty__t{font-family:'Crimson Pro',serif;font-size:22px;color:var(--text);margin:0 0 8px;}
.sr-empty__s{color:var(--muted);font-size:14.5px;line-height:1.7;max-width:52ch;margin:0 auto;}

/* meta line under answer */
.sr-metawrap:empty{display:none;}
.sr-meta{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-top:4px;padding:2px 4px;
  color:var(--faint);font-size:12.5px;}
.sr-metabit b{color:var(--muted);font-weight:600;}
.sr-metabit code{background:rgba(148,163,184,.10);border:1px solid var(--border);
  padding:1px 7px;border-radius:6px;font-size:11.5px;color:var(--muted);}
.sr-metasep{color:var(--faint);opacity:.5;}

/* citations */
.sr-cites:empty{display:none;}
.sr-cites{margin-top:14px!important;background:rgba(34,211,238,.05)!important;
  border:1px solid var(--border)!important;border-radius:14px!important;padding:16px 20px!important;}
.sr-cites h3{font-size:12px!important;text-transform:uppercase;letter-spacing:.14em;
  color:var(--accent-2)!important;margin:0 0 8px!important;font-family:'Inter',sans-serif!important;}
.sr-cites ul{margin:0!important;padding-left:2px!important;list-style:none!important;}
.sr-cites li{color:var(--muted)!important;font-size:14px!important;line-height:1.9!important;}
.sr-cites li strong{color:var(--accent-2);font-family:'JetBrains Mono',monospace;}

/* retrieved context */
.sr-ctx blockquote{border-left:2px solid var(--border-2)!important;color:var(--muted)!important;
  background:rgba(14,23,48,.6)!important;border-radius:0 8px 8px 0;padding:8px 14px!important;
  font-size:13.5px!important;}
.sr-ctx code{color:var(--accent-2);}

/* eval metric cards */
.sr-evalwrap:empty{display:none;}
.sr-evalwrap{margin-top:16px!important;}
.sr-evalhead{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.12em;
  margin-bottom:10px;display:flex;align-items:center;gap:10px;}
.sr-evaltag{background:rgba(34,211,238,.12);border:1px solid rgba(34,211,238,.3);color:#A9EEFB;
  padding:2px 8px;border-radius:999px;font-size:10.5px;letter-spacing:.06em;}
.sr-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
.sr-metric{background:var(--surface);border:1px solid var(--border);border-radius:13px;
  padding:16px 14px;text-align:center;}
.sr-metric__v{font-family:'Crimson Pro',serif;font-size:30px;font-weight:600;color:var(--accent);
  line-height:1;}
.sr-metric__k{color:var(--faint);font-size:12px;margin-top:6px;}
.sr-judgenote{margin-top:10px;color:var(--muted);font-size:13px;font-style:italic;
  border-left:2px solid var(--border-2);padding-left:12px;}

/* footer credit */
.sr-footer{margin-top:34px;padding-top:20px;border-top:1px solid var(--border);
  display:flex;flex-wrap:wrap;gap:10px;justify-content:space-between;align-items:center;
  color:var(--faint);font-size:13px;}
.sr-footer a{color:var(--muted);text-decoration:none;border-bottom:1px solid transparent;
  transition:all .16s ease;}
.sr-footer a:hover{color:var(--accent);border-bottom-color:rgba(52,211,153,.5);}

/* keyboard focus visibility */
.gradio-container button:focus-visible,
.gradio-container a:focus-visible,
.sr-seg label:focus-within{outline:2px solid rgba(52,211,153,.6)!important;outline-offset:2px;}

@media (max-width:640px){
  .sr-metrics{grid-template-columns:1fr;}
  .sr-answer{padding:20px 18px!important;}
  .sr-hero{padding-top:20px;}
}
@media (prefers-reduced-motion:reduce){
  *{animation:none!important;transition:none!important;}
}
"""

# Force the dark palette regardless of the viewer's system theme, so the design
# renders as intended for everyone (standard HF Spaces technique).
FORCE_DARK = """
() => {
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'dark') {
    u.searchParams.set('__theme', 'dark');
    window.location.replace(u.toString());
  }
}
"""

HERO = """
<div class="sr-hero">
  <span class="sr-eyebrow"><span class="sr-spark"></span>Agentic RAG · Research Assistant</span>
  <h1 class="sr-title">Scholar<span class="sr-dotmark">RAG</span></h1>
  <p class="sr-tag">Ask your research library and get <b>grounded, cited answers</b> —
  every claim traces back to a source, with a live <b>arXiv</b> fallback when your own
  papers fall short.</p>
</div>
"""

FOOTER = """
<div class="sr-footer">
  <span>Retrieval-augmented generation · MiniLM embeddings · tool-calling agent · LLM-as-judge</span>
  <a href="https://github.com/Siddharth-Golla6/scholar-rag" target="_blank" rel="noopener">View source on GitHub ↗</a>
</div>
"""


with gr.Blocks(theme=theme, css=CSS, js=FORCE_DARK, title="ScholarRAG") as demo:
    gr.HTML(HERO)
    status = gr.HTML(status_html(), elem_classes="sr-statuswrap")

    with gr.Group(elem_classes="sr-card"):
        question = gr.Textbox(
            label="Your question",
            placeholder="e.g. How do capsule networks and dynamic routing differ from CNNs?",
            lines=3,
            elem_id="sr-q",
        )
        with gr.Row():
            mode = gr.Radio(
                ["RAG (local corpus)", "Agent (corpus + arXiv)"],
                value="RAG (local corpus)",
                label="Mode",
                elem_classes="sr-seg",
                scale=3,
            )
            top_k = gr.Slider(
                2, 10, value=settings.top_k, step=1, label="Top-K passages",
                elem_classes="sr-slider", scale=2,
            )
        do_eval = gr.Checkbox(
            label="Evaluate answer quality (LLM-as-judge)", elem_classes="sr-check"
        )
        ask_btn = gr.Button("Ask ScholarRAG", variant="primary", elem_id="sr-ask")

    with gr.Column(elem_classes="sr-examples"):
        gr.Examples(
            examples=[
                ["What deep-learning methods are used for pneumonia detection from chest X-rays?"],
                ["How do capsule networks and dynamic routing differ from CNNs?"],
                ["How does Grad-CAM make model predictions interpretable?"],
            ],
            inputs=question,
            label="Try one",
        )

    with gr.Accordion("Build your library — add PDFs or fetch from arXiv", open=False, elem_classes="sr-lib"):
        uploads = gr.File(
            label="Upload PDFs", file_count="multiple", file_types=[".pdf"], type="filepath"
        )
        ingest_btn = gr.Button("Ingest uploaded")
        arxiv_q = gr.Textbox(label="…or fetch from arXiv", placeholder="capsule networks")
        arxiv_btn = gr.Button("Fetch & ingest")

    answer_out = gr.Markdown(_EMPTY_ANSWER, elem_classes="sr-answer")
    meta_out = gr.HTML(elem_classes="sr-metawrap", visible=False)
    cites_out = gr.Markdown(elem_classes="sr-cites", visible=False)
    ctx_acc = gr.Accordion("Retrieved context", open=False, elem_classes="sr-ctx", visible=False)
    with ctx_acc:
        ctx_out = gr.Markdown()
    eval_out = gr.HTML(elem_classes="sr-evalwrap", visible=False)

    gr.HTML(FOOTER)

    _outputs = [answer_out, meta_out, cites_out, ctx_acc, ctx_out, eval_out]
    ask_btn.click(ask, [question, mode, top_k, do_eval], _outputs)
    question.submit(ask, [question, mode, top_k, do_eval], _outputs)
    ingest_btn.click(ingest_uploaded, [uploads], [status])
    arxiv_btn.click(fetch_arxiv, [arxiv_q], [status])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
