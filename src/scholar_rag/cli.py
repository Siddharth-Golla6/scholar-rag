"""Command-line interface: ingest, fetch-arxiv, ask, eval, stats."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import get_settings
from .logging_utils import get_logger

log = get_logger("cli")

_EXTS = (".pdf", ".txt", ".md")


def _expand(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out += [x for x in path.glob("**/*") if x.suffix.lower() in _EXTS]
        elif path.suffix.lower() in _EXTS:
            out.append(path)
    return out


def cmd_ingest(args) -> None:
    from .ingestion import ingest_directory, ingest_paths

    if args.paths:
        n = ingest_paths(_expand(args.paths))
    else:
        n = ingest_directory()
    print(f"Ingested {n} chunks.")


def cmd_fetch_arxiv(args) -> None:
    from .embeddings import get_embedder
    from .ingestion import chunks_from_text
    from .tools.arxiv_search import search_arxiv
    from .vector_store import get_vector_store

    papers = search_arxiv(args.query, max_results=args.n)
    if not papers:
        print("No results (or network unavailable).")
        return
    chunks = []
    for p in papers:
        chunks += chunks_from_text(f"{p.title}\n\n{p.summary}", source=f"arXiv:{p.arxiv_id}", title=p.title)
    store = get_vector_store()
    store.add(chunks, get_embedder().encode([c.text for c in chunks]))
    print(f"Fetched {len(papers)} papers -> {len(chunks)} chunks. Store now holds {store.count()}.")
    for p in papers:
        print(f"  {p.arxiv_id}  {p.title}")


def cmd_ask(args) -> None:
    from .agent import Agent
    from .rag import RAGPipeline

    pipeline = Agent() if args.agent else RAGPipeline()
    ans = pipeline.answer(args.question, k=args.k)
    print("\n" + ans.answer + "\n")
    if ans.citations:
        print("Sources:")
        for c in ans.citations:
            loc = c.source + (f" p.{c.page}" if c.page else "")
            print(f"  [{c.marker}] {loc}" if c.marker else f"  - {loc}")
    print(f"\n(route={ans.route}, confidence={ans.confidence:.2f}, model={ans.model})")
    if args.evaluate:
        from .evaluation import evaluate_answer

        sc = evaluate_answer(ans)
        print(
            f"Eval -> faithfulness={sc.faithfulness:.2f} "
            f"answer_relevance={sc.answer_relevance:.2f} "
            f"context_relevance={sc.context_relevance:.2f}"
        )


def _load_questions(path: str) -> list[str]:
    questions = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                questions.append(json.loads(line)["question"])
    return questions


def cmd_eval(args) -> None:
    from .agent import Agent
    from .evaluation import evaluate_dataset
    from .rag import RAGPipeline

    questions = _load_questions(args.dataset)
    pipeline = Agent() if args.agent else RAGPipeline()
    report = evaluate_dataset(pipeline, questions)
    print("\n=== Evaluation report ===")
    print(f"Questions: {report.n}")
    print(f"  faithfulness      {report.mean_faithfulness:.3f}")
    print(f"  answer_relevance  {report.mean_answer_relevance:.3f}")
    print(f"  context_relevance {report.mean_context_relevance:.3f}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"Saved detailed report -> {out}")


def cmd_stats(args) -> None:
    from .vector_store import get_vector_store

    s = get_settings()
    store = get_vector_store(s)
    print(f"Provider : {s.llm_provider}")
    print(f"Model    : {s.llm_model}")
    print(f"LLM key  : {'present' if s.has_llm_key else 'MISSING (answering disabled)'}")
    print(f"Backend  : {type(store).__name__}")
    print(f"Indexed  : {store.count()} passages")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scholar", description="ScholarRAG CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Ingest PDFs/txt/md into the vector store")
    p.add_argument("paths", nargs="*", help="Files or dirs (default: data/papers)")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("fetch-arxiv", help="Fetch arXiv abstracts and ingest them")
    p.add_argument("query")
    p.add_argument("-n", type=int, default=5, help="Number of papers (default 5)")
    p.set_defaults(func=cmd_fetch_arxiv)

    p = sub.add_parser("ask", help="Ask a question")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=None, help="Top-K passages")
    p.add_argument("--agent", action="store_true", help="Use the arXiv-capable agent")
    p.add_argument("--evaluate", action="store_true", help="Also score the answer")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="Evaluate over a JSONL question set")
    p.add_argument("--dataset", default="eval/qa_dataset.jsonl")
    p.add_argument("--agent", action="store_true")
    p.add_argument("--out", default=None, help="Write a JSON report here")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("stats", help="Show configuration and index size")
    p.set_defaults(func=cmd_stats)

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
