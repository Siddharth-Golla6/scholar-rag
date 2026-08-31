"""Streamlit UI for ScholarRAG. Run: streamlit run app/ui.py"""
from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from scholar_rag.config import get_settings  # noqa: E402
from scholar_rag.ingestion import chunks_from_text, ingest_paths  # noqa: E402
from scholar_rag.vector_store import get_vector_store  # noqa: E402

st.set_page_config(page_title="ScholarRAG", page_icon="📚", layout="centered")

# On Streamlit Community Cloud, secrets set in the app dashboard are exposed via
# st.secrets; mirror them into the environment so config.get_settings() picks them up.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --ease-out: cubic-bezier(0.23,1,0.32,1);
  --bg:#08090c;
  --panel: rgba(255,255,255,0.035);
  --panel-2: rgba(255,255,255,0.06);
  --border: rgba(255,255,255,0.09);
  --border-strong: rgba(255,255,255,0.18);
  --text:#eef0f3;
  --muted:#98a2b3;
  --accent:#7c82ff;
  --accent-2:#b06bff;
  --accent-3:#4dd9ff;
  --grad: linear-gradient(120deg,#7c82ff,#b06bff 55%,#4dd9ff);
}

[data-testid="stDecoration"]{display:none;}
#MainMenu, footer{visibility:hidden;}
[data-testid="stHeader"]{background:transparent;}

html,body,.stApp,[data-testid="stAppViewContainer"]{
  color:var(--text);
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  -webkit-font-smoothing:antialiased;
}
.stApp{
  background:
    radial-gradient(55% 45% at 12% -8%, rgba(124,130,255,0.20), transparent 60%),
    radial-gradient(50% 42% at 100% -5%, rgba(176,107,255,0.16), transparent 55%),
    radial-gradient(45% 40% at 88% 10%, rgba(77,217,255,0.09), transparent 55%),
    var(--bg);
  background-attachment:fixed;
}
[data-testid="stAppViewContainer"]::before{
  content:"";position:fixed;top:0;left:0;right:0;height:3px;z-index:9999;background:var(--grad);
}

.block-container{padding-top:3.2rem;max-width:880px;}
h1,h2,h3,h4{letter-spacing:-0.02em;font-weight:600;}
[data-testid="stCaptionContainer"]{color:var(--muted);}

[data-testid="stMarkdownContainer"] code, code, kbd{
  font-family:'JetBrains Mono',ui-monospace,monospace;font-size:.8em;
  background:var(--panel-2);border:1px solid var(--border);border-radius:6px;
  padding:.06em .42em;color:#cfd3ff;
}

/* hero */
.hero-title{
  font-size:3.1rem;font-weight:800;letter-spacing:-0.035em;line-height:1.02;margin:0;
  background:var(--grad);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
}
.hero-sub{color:var(--muted);font-size:1.06rem;margin:.55rem 0 0;max-width:37rem;}
.pill-row{display:flex;gap:.5rem;flex-wrap:wrap;margin:1.1rem 0 .3rem;}
.pill{font-size:.78rem;color:#d7daff;border:1px solid var(--border);background:var(--panel-2);
  padding:.3rem .75rem;border-radius:999px;backdrop-filter:blur(6px);}

/* sidebar */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.012));
  border-right:1px solid var(--border);backdrop-filter:blur(8px);
}
.brand{display:flex;align-items:center;gap:.55rem;font-weight:700;font-size:1.2rem;letter-spacing:-0.02em;}
.brand .logo{font-size:1.4rem;}

/* status pill with live dot */
.status{display:inline-flex;align-items:center;gap:.5rem;font-size:.84rem;font-weight:500;
  padding:.42rem .8rem;border-radius:999px;border:1px solid var(--border);}
.status.ok{background:rgba(70,192,138,0.12);color:#7ee0b0;border-color:rgba(70,192,138,0.28);}
.status.warn{background:rgba(230,180,95,0.12);color:#f0cd8c;border-color:rgba(230,180,95,0.30);}
.status .dot{width:8px;height:8px;border-radius:50%;background:currentColor;animation:pulse 2.4s var(--ease-out) infinite;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(126,224,176,0.55);}70%{box-shadow:0 0 0 7px rgba(126,224,176,0);}100%{box-shadow:0 0 0 0 rgba(126,224,176,0);}}

/* buttons */
.stButton > button{
  border-radius:11px;border:1px solid var(--border-strong);background:var(--panel-2);color:var(--text);
  font-weight:500;padding:.55rem 1.1rem;
  transition:transform 160ms var(--ease-out),background 160ms var(--ease-out),border-color 160ms var(--ease-out),box-shadow 160ms var(--ease-out);
}
.stButton > button:active{transform:scale(0.98);}
@media (hover:hover) and (pointer:fine){.stButton>button:hover{border-color:var(--accent);background:rgba(124,130,255,0.12);}}
.stButton > button[kind="primary"]{
  background:var(--grad);border:none;color:#0a0b0f;font-weight:700;
  box-shadow:0 10px 26px -8px rgba(124,130,255,0.55);
}
@media (hover:hover) and (pointer:fine){.stButton>button[kind="primary"]:hover{filter:brightness(1.07);box-shadow:0 14px 34px -8px rgba(124,130,255,0.72);}}

/* inputs */
.stTextArea textarea,.stTextInput input{
  background:var(--panel-2);border:1px solid var(--border);border-radius:12px;color:var(--text);font-size:1rem;
  transition:border-color 160ms var(--ease-out),box-shadow 160ms var(--ease-out);
}
.stTextArea textarea:focus,.stTextInput input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(124,130,255,0.20);}

/* glass cards */
[data-testid="stExpander"] details,[data-testid="stVerticalBlockBorderWrapper"]{
  background:linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
  border:1px solid var(--border) !important;border-radius:16px !important;overflow:hidden;
  box-shadow:0 14px 40px -18px rgba(0,0,0,0.75);backdrop-filter:blur(8px);
  transition:border-color 160ms var(--ease-out);
}
[data-testid="stExpander"] details:hover{border-color:var(--border-strong);}
[data-testid="stExpander"] summary{font-weight:500;}

/* metric */
[data-testid="stMetric"]{
  background:linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
  border:1px solid var(--border);border-radius:14px;padding:.8rem 1rem;box-shadow:0 10px 30px -18px rgba(0,0,0,0.7);
}
[data-testid="stMetricValue"]{
  background:var(--grad);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
  font-weight:800;font-variant-numeric:tabular-nums;
}

[data-testid="stAlert"]{border-radius:12px;}
a,a:visited{color:var(--accent-3);text-decoration:none;} a:hover{text-decoration:underline;}
[data-testid="stSlider"] [role="slider"]{background:var(--accent) !important;}
.chips-label{color:var(--muted);font-size:.82rem;margin:.35rem 0 .15rem;}
"""
st.markdown("<style>" + _CSS + "</style>", unsafe_allow_html=True)


@st.cache_resource
def _pipeline():
    from scholar_rag.rag import RAGPipeline

    return RAGPipeline()


@st.cache_resource
def _agent():
    from scholar_rag.agent import Agent

    return Agent()


@st.cache_resource
def _store():
    from scholar_rag.ingestion import seed_corpus_if_empty

    store = get_vector_store()
    seed_corpus_if_empty(store=store)  # populate a fresh deploy with starter docs
    return store


def _set_q(q: str) -> None:
    st.session_state.question = q


settings = get_settings()
st.session_state.setdefault("question", "")

with st.sidebar:
    st.markdown('<div class="brand"><span class="logo">📚</span> ScholarRAG</div>', unsafe_allow_html=True)
    st.caption("Agentic RAG research assistant")
    store = _store()
    st.metric("Passages indexed", store.count())
    st.write(f"**Model** `{settings.llm_model}` · `{settings.llm_provider}`")
    if settings.has_llm_key:
        st.markdown('<div class="status ok"><span class="dot"></span> Answering enabled</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status warn"><span class="dot"></span> Retrieval only — add GROQ_API_KEY</div>', unsafe_allow_html=True)

    st.divider()
    mode = st.radio("Mode", ["RAG (local corpus)", "Agent (corpus + arXiv)"])
    top_k = st.slider("Top-K passages", 2, 10, settings.top_k)
    run_eval = st.checkbox("Evaluate answer (LLM-as-judge)")

    st.divider()
    st.subheader("Build your library")
    uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploads and st.button("Ingest uploaded PDFs", use_container_width=True):
        settings.papers_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in uploads:
            dest = settings.papers_dir / f.name
            dest.write_bytes(f.getbuffer())
            paths.append(dest)
        with st.spinner("Ingesting…"):
            n = ingest_paths(paths)
        _store.clear()
        st.success(f"Ingested {n} chunks from {len(paths)} file(s).")

    arxiv_q = st.text_input("…or fetch from arXiv")
    if arxiv_q and st.button("Fetch & ingest arXiv abstracts", use_container_width=True):
        from scholar_rag.embeddings import get_embedder
        from scholar_rag.tools.arxiv_search import search_arxiv

        papers = search_arxiv(arxiv_q, max_results=5)
        chunks = []
        for p in papers:
            chunks += chunks_from_text(f"{p.title}\n\n{p.summary}", source=f"arXiv:{p.arxiv_id}", title=p.title)
        if chunks:
            store.add(chunks, get_embedder().encode([c.text for c in chunks]))
            _store.clear()
            st.success(f"Added {len(papers)} arXiv abstracts.")
        else:
            st.error("No results (or network unavailable).")

# --- Hero ---
st.markdown(
    '<div class="hero-title">ScholarRAG</div>'
    '<div class="hero-sub">Ask your research library — grounded, cited answers, with a live '
    'arXiv fallback when your own papers fall short.</div>'
    '<div class="pill-row">'
    '<span class="pill">📎 Inline citations</span>'
    '<span class="pill">🔎 arXiv agent</span>'
    '<span class="pill">✅ Self-evaluating</span>'
    '</div>',
    unsafe_allow_html=True,
)
st.write("")

question = st.text_area(
    "Question",
    key="question",
    placeholder="Ask anything about your indexed papers…",
    height=100,
    label_visibility="collapsed",
)

EXAMPLES = [
    ("🫁 Pneumonia detection", "What deep-learning methods are used for pneumonia detection from chest X-rays?"),
    ("🧬 Capsule networks", "How do capsule networks and dynamic routing differ from CNNs?"),
    ("🔬 Grad-CAM", "How does Grad-CAM make model predictions interpretable?"),
]
st.markdown('<div class="chips-label">Try one</div>', unsafe_allow_html=True)
chip_cols = st.columns(len(EXAMPLES))
for col, (short, full) in zip(chip_cols, EXAMPLES):
    col.button(short, key=f"chip_{short}", on_click=_set_q, args=(full,), use_container_width=True)

st.write("")
ask = st.button("Ask", type="primary", use_container_width=True)

if ask and question.strip():
    use_agent = mode.startswith("Agent")

    if not settings.has_llm_key:
        st.info("No LLM key set — showing retrieved passages only.")
        from scholar_rag.retriever import Retriever

        for i, rc in enumerate(Retriever().retrieve(question, k=top_k), 1):
            with st.expander(f"[{i}] {rc.chunk.source} · score {rc.score:.2f}"):
                st.write(rc.chunk.text)
    else:
        try:
            with st.spinner("Thinking…"):
                answer = _agent().run(question) if use_agent else _pipeline().answer(question, k=top_k)
        except Exception as exc:  # Streamlit redacts uncaught errors — surface the real cause
            msg = str(exc)
            low = msg.lower()
            if "401" in msg or "invalid_api_key" in low or "invalid api key" in low:
                st.error("🔑 GROQ_API_KEY is missing or invalid. Open **Manage app → Settings → Secrets** and re-check it (format: `GROQ_API_KEY = \"gsk_...\"`).")
            elif "429" in msg or "rate_limit" in low or "rate limit" in low:
                st.error("⏳ Groq free-tier rate limit reached (per-minute or daily quota). Wait a bit and retry, or add a fresh key.")
            elif "404" in msg or "model_not_found" in low or "does not exist" in low:
                st.error(f"🧩 Model `{settings.llm_model}` isn't available for this key. Set a valid `LLM_MODEL` in Secrets.")
            else:
                st.error(f"LLM call failed: {msg}")
            st.stop()

        with st.container(border=True):
            st.markdown(answer.answer)
            st.caption(
                f"Route **{answer.route}**  ·  confidence {answer.confidence:.2f}  ·  model `{answer.model}`"
            )

        if answer.citations:
            st.markdown("##### Citations")
            for c in answer.citations:
                loc = c.source + (f" p.{c.page}" if c.page else "")
                label = f"[{c.marker}] {loc}" if c.marker else loc
                with st.expander(label):
                    st.write(c.snippet)

        with st.expander(f"Retrieved context · {len(answer.contexts)} passages"):
            for i, rc in enumerate(answer.contexts, 1):
                st.markdown(f"**[{i}]** `{rc.chunk.source}` — score {rc.score:.3f}")
                st.caption(rc.chunk.text[:400])

        if run_eval:
            from scholar_rag.evaluation import evaluate_answer

            st.markdown("##### Answer quality (LLM-as-judge)")
            with st.spinner("Scoring answer…"):
                scores = evaluate_answer(answer)
            c1, c2, c3 = st.columns(3)
            c1.metric("Faithfulness", f"{scores.faithfulness:.2f}")
            c2.metric("Answer relevance", f"{scores.answer_relevance:.2f}")
            c3.metric("Context relevance", f"{scores.context_relevance:.2f}")
            note = scores.reasoning.get("reason")
            if scores.faithfulness < 0.99 and note:
                st.caption("Judge note — " + note)
