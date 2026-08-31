"""Vercel serverless function for ScholarRAG (torch-free).

Vercel can't run the full torch/Gradio app, so this is a lightweight variant:
keyword retrieval over a small baked-in corpus + a direct Groq call via stdlib.
Only dependency is an env var GROQ_API_KEY (set in the Vercel project settings).
"""
from http.server import BaseHTTPRequestHandler
import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter

# --- baked-in starter corpus (same topics as the full app's seed docs) ---
CORPUS = {
    "capsule_networks.md": (
        "Capsule networks (CapsNets), introduced by Sabour, Frosst and Hinton in 2017, represent "
        "visual entities as groups of neurons called capsules whose activity vector encodes both the "
        "probability that an entity is present and its pose, orientation and scale. Unlike CNNs that "
        "rely on scalar feature detectors and max-pooling, capsules preserve spatial hierarchies "
        "between parts and wholes. The key mechanism is dynamic routing by agreement: lower-level "
        "capsules send outputs to higher-level capsules whose predictions agree, replacing pooling "
        "which discards spatial information. CapsNets are more robust to affine transformations and "
        "viewpoint changes and generalize from fewer examples, at higher computational cost. In "
        "medical imaging they have been explored for chest X-ray classification."
    ),
    "transformers_attention.md": (
        "The Transformer, from 'Attention Is All You Need' (Vaswani et al., 2017), replaced recurrence "
        "and convolution with self-attention. Self-attention computes a weighted sum over all positions "
        "using query-key similarity, letting every token attend to every other token and capturing "
        "long-range dependencies in a single layer, which recurrent networks struggle with. Multi-head "
        "attention attends to several representation subspaces at once, and positional encodings inject "
        "token order. Transformers parallelize across positions, enabling training on huge corpora and "
        "the rise of large language models. Vision Transformers (ViT) split an image into patches as "
        "tokens for image classification and medical imaging."
    ),
    "retrieval_augmented_generation.md": (
        "Retrieval-Augmented Generation (RAG) grounds a language model in an external knowledge source. "
        "Documents are chunked, embedded into dense vectors and stored in a vector database; at query "
        "time the question is embedded and the most similar chunks are retrieved, often re-ranked with "
        "Maximal Marginal Relevance (MMR) to balance relevance and diversity. The retrieved passages are "
        "inserted into the prompt and the model answers conditioned on them, citing the passages it "
        "used. RAG reduces hallucination because claims trace to sources. Production systems add "
        "evaluation of faithfulness and answer relevance, and guardrails that say 'I don't know' when "
        "context is insufficient. Agentic RAG lets the model call tools such as an arXiv search."
    ),
    "gradcam_interpretability.md": (
        "Gradient-weighted Class Activation Mapping (Grad-CAM), by Selvaraju et al. (2017), explains CNN "
        "predictions. It uses the gradients of a target class flowing into the final convolutional layer "
        "to produce a coarse heat map highlighting the image regions most responsible for the "
        "prediction, and works across CNN architectures without retraining. In medical imaging, Grad-CAM "
        "heat maps overlaid on chest X-rays show which lung regions drove a diagnosis, confirming the "
        "model attends to disease-relevant areas rather than spurious cues, central to explainable AI "
        "and clinical trust. It also enables weakly supervised localization from image-level labels. "
        "Variants include Grad-CAM++ and axiom-based Grad-CAM."
    ),
    "pneumonia_detection_cxr.md": (
        "Deep-learning pneumonia detection from chest X-rays mostly uses convolutional neural networks "
        "pretrained on natural images and fine-tuned via transfer learning, with backbones such as "
        "ResNet, DenseNet, EfficientNet and MobileNet. To handle limited, imbalanced data, practitioners "
        "use data augmentation, focal loss and patient-wise splits that prevent leakage. Ensembles of "
        "heterogeneous backbones, sometimes including Vision Transformers, improve robustness. "
        "Interpretability comes from Grad-CAM heat maps highlighting influential lung regions. Systems "
        "aim to raise sensitivity and reduce false negatives so fewer pneumonia cases are missed. "
        "Lightweight and knowledge-distilled models target on-device or point-of-care triage."
    ),
    "wearable_stress_wesad.md": (
        "WESAD (Wearable Stress and Affect Detection) is a public multimodal dataset for stress and "
        "affect from physiological signals, with chest- and wrist-worn sensors capturing electrodermal "
        "activity (EDA/GSR), heart rate and blood volume pulse, skin temperature, respiration and "
        "accelerometer data across baseline, stress and amusement conditions. Pipelines segment signals "
        "into windows and extract engineered features (heart-rate-variability statistics, EDA phasic and "
        "tonic components, temperature trends) before training classifiers such as LightGBM or XGBoost. "
        "Generalizing to unseen users is hard because baselines vary between people, so leave-one-subject-"
        "out evaluation is used. Real-time systems stream data from wearables such as the ESP32 and show "
        "stress insights on a dashboard."
    ),
}

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "to", "in", "on", "for",
    "and", "or", "as", "at", "by", "with", "that", "this", "these", "those", "it", "its",
    "how", "do", "does", "did", "what", "which", "why", "when", "where", "who", "can", "you",
    "your", "from", "into", "using", "use", "used", "about", "they", "their", "them", "we",
}

SYSTEM = (
    "You are ScholarRAG, a precise research assistant. Answer the user's question using ONLY the "
    "numbered context passages. Cite every claim with its passage number in square brackets, e.g. "
    "[1] or [2][3]. If the passages do not contain the answer, reply exactly: \"I don't have enough "
    "information in the provided sources to answer that.\" Be concise and technical; never invent facts."
)


def retrieve(query: str, k: int = 3):
    terms = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in _STOP and len(w) > 2]
    scored = []
    for source, text in CORPUS.items():
        counts = Counter(re.findall(r"[a-z0-9]+", text.lower()))
        score = sum(counts.get(w, 0) for w in terms)
        scored.append((score, source, text))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [(s, t) for sc, s, t in scored[:k] if sc > 0]
    return top or [(scored[0][1], scored[0][2])]


def _build_prompt(question: str, contexts):
    blocks = [f"[{i}] (source: {src})\n{text}" for i, (src, text) in enumerate(contexts, 1)]
    return "Context passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n\nAnswer (cite [n]):"


def _call_groq(system: str, user: str) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set in the Vercel project's Environment Variables.")
    payload = {
        "model": (os.environ.get("LLM_MODEL") or "openai/gpt-oss-120b").strip(),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {key.strip()}",
            "Content-Type": "application/json",
            "User-Agent": "ScholarRAG/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def answer_question(question: str) -> dict:
    contexts = retrieve(question, k=3)
    text = _call_groq(SYSTEM, _build_prompt(question, contexts)).strip()
    used = sorted({int(m) for m in re.findall(r"[\[【](\d+)[\]】]", text)})
    citations = [
        {"marker": n, "source": contexts[n - 1][0]} for n in used if 1 <= n <= len(contexts)
    ]
    return {
        "answer": text,
        "citations": citations,
        "contexts": [{"source": s, "snippet": t[:220]} for s, t in contexts],
    }


class handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            question = (body.get("question") or "").strip()
            if not question:
                self._send(400, {"error": "A question is required."})
                return
            self._send(200, answer_question(question))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="ignore")
            if exc.code == 401:
                hint = "GROQ_API_KEY is missing or invalid. Set it in Vercel → Settings → Environment Variables."
            elif exc.code == 429:
                hint = "Groq rate limit reached. Wait a moment and try again."
            else:
                hint = f"Upstream error {exc.code}: {detail[:200]}"
            self._send(200, {"error": hint})
        except Exception as exc:  # noqa: BLE001
            self._send(200, {"error": str(exc)[:300]})
