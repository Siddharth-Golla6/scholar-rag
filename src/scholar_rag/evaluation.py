"""LLM-as-judge evaluation of RAG answers.

Three reference-free metrics, each scored 0-1 by the model in a SINGLE judge call
(cheaper and gentler on rate limits than one call per metric):
  * faithfulness      — is every claim grounded in the retrieved context?
  * answer_relevance  — does the answer actually address the question?
  * context_relevance — were the retrieved passages relevant to the question?

This is the piece most portfolio projects skip; it turns "it seems to work" into
a measurable quality signal and directly surfaces hallucinations.
"""
from __future__ import annotations

import json
import re
from statistics import mean

from .llm import LLMError, get_llm
from .logging_utils import get_logger
from .schemas import Answer, EvalItem, EvalReport, EvalScores

log = get_logger("evaluation")

JUDGE_SYSTEM = "You are a strict evaluator of RAG systems. Respond ONLY with compact JSON."

COMBINED_JUDGE_PROMPT = """Evaluate a RAG answer on three axes, each a float in [0, 1]:
- faithfulness: is EVERY claim in the ANSWER supported by the CONTEXT? (1.0 fully grounded, 0.0 mostly hallucinated)
- answer_relevance: does the ANSWER address the QUESTION? (judge relevance/completeness, not correctness)
- context_relevance: what fraction of the CONTEXT passages are relevant to the QUESTION?

Output the three score keys FIRST so they are never truncated.
Return JSON:
{{"faithfulness": <float>, "answer_relevance": <float>, "context_relevance": <float>, "reason": "<short>"}}

QUESTION: {question}

CONTEXT:
{context}

ANSWER:
{answer}"""


def _parse_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # Salvage: score keys are emitted first, so recover them even if the JSON
    # was truncated at max_tokens before the closing brace.
    out: dict = {}
    for key in ("faithfulness", "answer_relevance", "context_relevance", "score"):
        m = re.search(rf'"{key}"\s*:\s*([0-9]*\.?[0-9]+)', raw)
        if m:
            out[key] = float(m.group(1))
    if out:
        log.warning("judge JSON malformed/truncated; salvaged %s", list(out))
    return out


def _clip(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _judge(llm, prompt: str) -> dict:
    messages = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}]
    # Reasoning models (e.g. gpt-oss) spend tokens on internal reasoning before
    # emitting content, so the judge needs generous headroom or it returns empty.
    try:  # force strict JSON output where the provider supports it
        raw = llm.chat(messages, temperature=0.0, max_tokens=1500,
                       response_format={"type": "json_object"})
    except LLMError:
        raw = llm.chat(messages, temperature=0.0, max_tokens=1500)
    if not raw.strip():
        log.warning("judge returned empty content (reasoning likely hit max_tokens)")
    return _parse_json(raw)


def evaluate_answer(ans: Answer, llm=None) -> EvalScores:
    llm = llm or get_llm()
    context = "\n\n".join(
        f"[{i}] {rc.chunk.text}" for i, rc in enumerate(ans.contexts, 1)
    ) or "(no context)"
    result = _judge(
        llm, COMBINED_JUDGE_PROMPT.format(question=ans.question, context=context, answer=ans.answer)
    )
    return EvalScores(
        faithfulness=_clip(result.get("faithfulness")),
        answer_relevance=_clip(result.get("answer_relevance")),
        context_relevance=_clip(result.get("context_relevance")),
        reasoning={"reason": result.get("reason", "")},
    )


def evaluate_dataset(pipeline, questions: list[str], llm=None) -> EvalReport:
    llm = llm or get_llm()
    items: list[EvalItem] = []
    for q in questions:
        ans = pipeline.answer(q)
        scores = evaluate_answer(ans, llm=llm)
        items.append(EvalItem(question=q, answer=ans.answer, scores=scores))
        log.info(
            "eval | faith=%.2f ans_rel=%.2f ctx_rel=%.2f | %s",
            scores.faithfulness, scores.answer_relevance, scores.context_relevance, q[:50],
        )
    if not items:
        return EvalReport()
    return EvalReport(
        n=len(items),
        mean_faithfulness=mean(i.scores.faithfulness for i in items),
        mean_answer_relevance=mean(i.scores.answer_relevance for i in items),
        mean_context_relevance=mean(i.scores.context_relevance for i in items),
        items=items,
    )
