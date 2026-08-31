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
import base64

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


INDEX_HTML = base64.b64decode("PCFkb2N0eXBlIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9InV0Zi04IiAvPgo8bWV0YSBuYW1lPSJ2aWV3cG9ydCIgY29udGVudD0id2lkdGg9ZGV2aWNlLXdpZHRoLCBpbml0aWFsLXNjYWxlPTEiIC8+Cjx0aXRsZT5TY2hvbGFyUkFHPC90aXRsZT4KPGxpbmsgcmVsPSJwcmVjb25uZWN0IiBocmVmPSJodHRwczovL2ZvbnRzLmdvb2dsZWFwaXMuY29tIiAvPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ3N0YXRpYy5jb20iIGNyb3Nzb3JpZ2luIC8+CjxsaW5rIGhyZWY9Imh0dHBzOi8vZm9udHMuZ29vZ2xlYXBpcy5jb20vY3NzMj9mYW1pbHk9SW50ZXI6d2dodEA0MDA7NTAwOzYwMDs3MDA7ODAwJmZhbWlseT1KZXRCcmFpbnMrTW9ubzp3Z2h0QDQwMDs1MDAmZGlzcGxheT1zd2FwIiByZWw9InN0eWxlc2hlZXQiIC8+CjxzdHlsZT4KICA6cm9vdHsKICAgIC0tZWFzZS1vdXQ6Y3ViaWMtYmV6aWVyKDAuMjMsMSwwLjMyLDEpOwogICAgLS1iZzojMDgwOTBjOyAtLXBhbmVsOnJnYmEoMjU1LDI1NSwyNTUsLjA0KTsgLS1wYW5lbC0yOnJnYmEoMjU1LDI1NSwyNTUsLjA2KTsKICAgIC0tYm9yZGVyOnJnYmEoMjU1LDI1NSwyNTUsLjA5KTsgLS1ib3JkZXItc3Ryb25nOnJnYmEoMjU1LDI1NSwyNTUsLjE3KTsKICAgIC0tdGV4dDojZWVmMGYzOyAtLW11dGVkOiM5OGEyYjM7IC0tYWNjZW50OiM3YzgyZmY7IC0tYWNjZW50LTI6I2IwNmJmZjsgLS1hY2NlbnQtMzojNGRkOWZmOwogICAgLS1ncmFkOmxpbmVhci1ncmFkaWVudCgxMjBkZWcsIzdjODJmZiwjYjA2YmZmIDU1JSwjNGRkOWZmKTsKICB9CiAgKntib3gtc2l6aW5nOmJvcmRlci1ib3h9CiAgaHRtbCxib2R5e21hcmdpbjowfQogIGJvZHl7CiAgICBiYWNrZ3JvdW5kOgogICAgICByYWRpYWwtZ3JhZGllbnQoNTUlIDQ1JSBhdCAxMiUgLTglLHJnYmEoMTI0LDEzMCwyNTUsLjIwKSx0cmFuc3BhcmVudCA2MCUpLAogICAgICByYWRpYWwtZ3JhZGllbnQoNTAlIDQyJSBhdCAxMDAlIC01JSxyZ2JhKDE3NiwxMDcsMjU1LC4xNiksdHJhbnNwYXJlbnQgNTUlKSwKICAgICAgcmFkaWFsLWdyYWRpZW50KDQ1JSA0MCUgYXQgODglIDEwJSxyZ2JhKDc3LDIxNywyNTUsLjA5KSx0cmFuc3BhcmVudCA1NSUpLAogICAgICB2YXIoLS1iZyk7CiAgICBiYWNrZ3JvdW5kLWF0dGFjaG1lbnQ6Zml4ZWQ7IGNvbG9yOnZhcigtLXRleHQpOwogICAgZm9udC1mYW1pbHk6J0ludGVyJywtYXBwbGUtc3lzdGVtLEJsaW5rTWFjU3lzdGVtRm9udCwnU2Vnb2UgVUknLHNhbnMtc2VyaWY7CiAgICAtd2Via2l0LWZvbnQtc21vb3RoaW5nOmFudGlhbGlhc2VkOyBtaW4taGVpZ2h0OjEwMHZoOwogIH0KICBib2R5OjpiZWZvcmV7Y29udGVudDoiIjtwb3NpdGlvbjpmaXhlZDtpbnNldDowIDAgYXV0byAwO2hlaWdodDozcHg7YmFja2dyb3VuZDp2YXIoLS1ncmFkKTt6LWluZGV4Ojk5fQogIC53cmFwe21heC13aWR0aDo4MjBweDttYXJnaW46MCBhdXRvO3BhZGRpbmc6NjRweCAyMnB4IDgwcHh9CiAgLnRpdGxle2ZvbnQtc2l6ZTpjbGFtcCgyLjZyZW0sNnZ3LDMuM3JlbSk7Zm9udC13ZWlnaHQ6ODAwO2xldHRlci1zcGFjaW5nOi0uMDM1ZW07bGluZS1oZWlnaHQ6MS4wMjttYXJnaW46MDsKICAgIGJhY2tncm91bmQ6dmFyKC0tZ3JhZCk7LXdlYmtpdC1iYWNrZ3JvdW5kLWNsaXA6dGV4dDtiYWNrZ3JvdW5kLWNsaXA6dGV4dDstd2Via2l0LXRleHQtZmlsbC1jb2xvcjp0cmFuc3BhcmVudH0KICAuc3Vie2NvbG9yOnZhcigtLW11dGVkKTtmb250LXNpemU6MS4wNnJlbTttYXJnaW46LjZyZW0gMCAwO21heC13aWR0aDo0MHJlbX0KICAucGlsbHN7ZGlzcGxheTpmbGV4O2dhcDouNXJlbTtmbGV4LXdyYXA6d3JhcDttYXJnaW46MS4xcmVtIDAgMH0KICAucGlsbHtmb250LXNpemU6Ljc4cmVtO2NvbG9yOiNkN2RhZmY7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JhY2tncm91bmQ6dmFyKC0tcGFuZWwtMik7CiAgICBwYWRkaW5nOi4zcmVtIC43NXJlbTtib3JkZXItcmFkaXVzOjk5OXB4O2JhY2tkcm9wLWZpbHRlcjpibHVyKDZweCl9CiAgLmNhcmR7YmFja2dyb3VuZDpsaW5lYXItZ3JhZGllbnQoMTgwZGVnLHJnYmEoMjU1LDI1NSwyNTUsLjA1KSxyZ2JhKDI1NSwyNTUsMjU1LC4wMTUpKTsKICAgIGJvcmRlcjoxcHggc29saWQgdmFyKC0tYm9yZGVyKTtib3JkZXItcmFkaXVzOjE2cHg7Ym94LXNoYWRvdzowIDE0cHggNDBweCAtMThweCByZ2JhKDAsMCwwLC43NSk7CiAgICBiYWNrZHJvcC1maWx0ZXI6Ymx1cig4cHgpO3BhZGRpbmc6MThweH0KICAuYXNrLWNhcmR7bWFyZ2luLXRvcDoyOHB4fQogIHRleHRhcmVhe3dpZHRoOjEwMCU7YmFja2dyb3VuZDp2YXIoLS1wYW5lbC0yKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlcik7Ym9yZGVyLXJhZGl1czoxMnB4O2NvbG9yOnZhcigtLXRleHQpOwogICAgZm9udDppbmhlcml0O2ZvbnQtc2l6ZToxcmVtO3BhZGRpbmc6MTRweDtyZXNpemU6dmVydGljYWw7bWluLWhlaWdodDo5NnB4O3RyYW5zaXRpb246Ym9yZGVyLWNvbG9yIC4xNnMgdmFyKC0tZWFzZS1vdXQpLGJveC1zaGFkb3cgLjE2cyB2YXIoLS1lYXNlLW91dCl9CiAgdGV4dGFyZWE6Zm9jdXN7b3V0bGluZTpub25lO2JvcmRlci1jb2xvcjp2YXIoLS1hY2NlbnQpO2JveC1zaGFkb3c6MCAwIDAgM3B4IHJnYmEoMTI0LDEzMCwyNTUsLjIpfQogIC5jaGlwc3tkaXNwbGF5OmZsZXg7Z2FwOi41cmVtO2ZsZXgtd3JhcDp3cmFwO21hcmdpbjoxMnB4IDAgNHB4fQogIC5jaGlwe2ZvbnQtc2l6ZTouODVyZW07Y29sb3I6dmFyKC0tdGV4dCk7YmFja2dyb3VuZDp2YXIoLS1wYW5lbC0yKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWJvcmRlci1zdHJvbmcpOwogICAgYm9yZGVyLXJhZGl1czoxMHB4O3BhZGRpbmc6LjVyZW0gLjhyZW07Y3Vyc29yOnBvaW50ZXI7dHJhbnNpdGlvbjp0cmFuc2Zvcm0gLjE2cyB2YXIoLS1lYXNlLW91dCksYm9yZGVyLWNvbG9yIC4xNnMgdmFyKC0tZWFzZS1vdXQpLGJhY2tncm91bmQgLjE2cyB2YXIoLS1lYXNlLW91dCl9CiAgLmNoaXA6aG92ZXJ7Ym9yZGVyLWNvbG9yOnZhcigtLWFjY2VudCk7YmFja2dyb3VuZDpyZ2JhKDEyNCwxMzAsMjU1LC4xMil9CiAgLmNoaXA6YWN0aXZle3RyYW5zZm9ybTpzY2FsZSguOTgpfQogIC5hc2stYnRue3dpZHRoOjEwMCU7bWFyZ2luLXRvcDoxNHB4O2JvcmRlcjpub25lO2JvcmRlci1yYWRpdXM6MTJweDtjb2xvcjojMGEwYjBmO2ZvbnQtd2VpZ2h0OjcwMDtmb250LXNpemU6MXJlbTsKICAgIHBhZGRpbmc6MTRweDtjdXJzb3I6cG9pbnRlcjtiYWNrZ3JvdW5kOnZhcigtLWdyYWQpO2JveC1zaGFkb3c6MCAxMHB4IDI2cHggLThweCByZ2JhKDEyNCwxMzAsMjU1LC41NSk7CiAgICB0cmFuc2l0aW9uOnRyYW5zZm9ybSAuMTZzIHZhcigtLWVhc2Utb3V0KSxmaWx0ZXIgLjE2cyB2YXIoLS1lYXNlLW91dCl9CiAgLmFzay1idG46aG92ZXJ7ZmlsdGVyOmJyaWdodG5lc3MoMS4wNyl9IC5hc2stYnRuOmFjdGl2ZXt0cmFuc2Zvcm06c2NhbGUoLjk5KX0KICAuYXNrLWJ0bjpkaXNhYmxlZHtmaWx0ZXI6Z3JheXNjYWxlKC40KSBicmlnaHRuZXNzKC44KTtjdXJzb3I6ZGVmYXVsdH0KICAjb3V0e21hcmdpbi10b3A6MjJweDtkaXNwbGF5Om5vbmV9CiAgLmFuc3dlcnt3aGl0ZS1zcGFjZTpwcmUtd3JhcDtsaW5lLWhlaWdodDoxLjY7Zm9udC1zaXplOjEuMDNyZW19CiAgLmFuc3dlciBzdXB7Y29sb3I6dmFyKC0tYWNjZW50LTMpO2ZvbnQtd2VpZ2h0OjYwMDtmb250LXNpemU6LjcyZW19CiAgLm1ldGF7Y29sb3I6dmFyKC0tbXV0ZWQpO2ZvbnQtc2l6ZTouODJyZW07bWFyZ2luLXRvcDoxMnB4O2ZvbnQtZmFtaWx5OidKZXRCcmFpbnMgTW9ubycsbW9ub3NwYWNlfQogIGgzLnNlY3tmb250LXNpemU6Ljk1cmVtO21hcmdpbjoyMnB4IDAgOHB4O2NvbG9yOnZhcigtLXRleHQpfQogIC5jaXRle2Rpc3BsYXk6ZmxleDtnYXA6LjVyZW07YWxpZ24taXRlbXM6Y2VudGVyO2ZvbnQtc2l6ZTouOXJlbTtjb2xvcjojY2RkMmZmO2JhY2tncm91bmQ6dmFyKC0tcGFuZWwpOwogICAgYm9yZGVyOjFweCBzb2xpZCB2YXIoLS1ib3JkZXIpO2JvcmRlci1yYWRpdXM6MTBweDtwYWRkaW5nOi41cmVtIC43cmVtO21hcmdpbi1ib3R0b206NnB4fQogIC5jaXRlIGJ7Y29sb3I6dmFyKC0tYWNjZW50KX0KICAuZXJye2NvbG9yOiNmNmIwYjA7YmFja2dyb3VuZDpyZ2JhKDIzMCw5MCw5MCwuMSk7Ym9yZGVyOjFweCBzb2xpZCByZ2JhKDIzMCw5MCw5MCwuMjgpO2JvcmRlci1yYWRpdXM6MTJweDtwYWRkaW5nOjE0cHh9CiAgLnNwaW5uZXJ7ZGlzcGxheTppbmxpbmUtYmxvY2s7d2lkdGg6MTZweDtoZWlnaHQ6MTZweDtib3JkZXI6MnB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsLjI1KTtib3JkZXItdG9wLWNvbG9yOiMwYTBiMGY7CiAgICBib3JkZXItcmFkaXVzOjUwJTthbmltYXRpb246c3BpbiAuN3MgbGluZWFyIGluZmluaXRlO3ZlcnRpY2FsLWFsaWduOi0zcHg7bWFyZ2luLXJpZ2h0OjhweH0KICBAa2V5ZnJhbWVzIHNwaW57dG97dHJhbnNmb3JtOnJvdGF0ZSgzNjBkZWcpfX0KICAuZm9vdHttYXJnaW4tdG9wOjQwcHg7Y29sb3I6dmFyKC0tbXV0ZWQpO2ZvbnQtc2l6ZTouODVyZW07dGV4dC1hbGlnbjpjZW50ZXJ9CiAgLmZvb3QgYXtjb2xvcjp2YXIoLS1hY2NlbnQtMyk7dGV4dC1kZWNvcmF0aW9uOm5vbmV9IC5mb290IGE6aG92ZXJ7dGV4dC1kZWNvcmF0aW9uOnVuZGVybGluZX0KICBAbWVkaWEgKHByZWZlcnMtcmVkdWNlZC1tb3Rpb246cmVkdWNlKXsqe2FuaW1hdGlvbjpub25lIWltcG9ydGFudDt0cmFuc2l0aW9uOm5vbmUhaW1wb3J0YW50fX0KPC9zdHlsZT4KPC9oZWFkPgo8Ym9keT4KICA8bWFpbiBjbGFzcz0id3JhcCI+CiAgICA8aDEgY2xhc3M9InRpdGxlIj5TY2hvbGFyUkFHPC9oMT4KICAgIDxwIGNsYXNzPSJzdWIiPkFzayBhIHJlc2VhcmNoIHF1ZXN0aW9uIGFuZCBnZXQgYSA8c3Ryb25nPmdyb3VuZGVkLCBjaXRlZCBhbnN3ZXI8L3N0cm9uZz4gZnJvbSBhIGN1cmF0ZWQgY29ycHVzIOKAlCByZXRyaWV2YWwgKyBhbiBMTE0sIHJ1bm5pbmcgb24gVmVyY2VsLjwvcD4KICAgIDxkaXYgY2xhc3M9InBpbGxzIj4KICAgICAgPHNwYW4gY2xhc3M9InBpbGwiPvCfk44gSW5saW5lIGNpdGF0aW9uczwvc3Bhbj4KICAgICAgPHNwYW4gY2xhc3M9InBpbGwiPuKaoSBTZXJ2ZXJsZXNzPC9zcGFuPgogICAgICA8c3BhbiBjbGFzcz0icGlsbCI+8J+noCBHcm9xIExMTTwvc3Bhbj4KICAgIDwvZGl2PgoKICAgIDxzZWN0aW9uIGNsYXNzPSJjYXJkIGFzay1jYXJkIj4KICAgICAgPHRleHRhcmVhIGlkPSJxIiBwbGFjZWhvbGRlcj0iZS5nLiBIb3cgZG8gY2Fwc3VsZSBuZXR3b3JrcyBhbmQgZHluYW1pYyByb3V0aW5nIGRpZmZlciBmcm9tIENOTnM/Ij48L3RleHRhcmVhPgogICAgICA8ZGl2IGNsYXNzPSJjaGlwcyIgaWQ9ImNoaXBzIj4KICAgICAgICA8YnV0dG9uIGNsYXNzPSJjaGlwIiBkYXRhLXE9IldoYXQgZGVlcC1sZWFybmluZyBtZXRob2RzIGFyZSB1c2VkIGZvciBwbmV1bW9uaWEgZGV0ZWN0aW9uIGZyb20gY2hlc3QgWC1yYXlzPyI+8J+rgSBQbmV1bW9uaWEgZGV0ZWN0aW9uPC9idXR0b24+CiAgICAgICAgPGJ1dHRvbiBjbGFzcz0iY2hpcCIgZGF0YS1xPSJIb3cgZG8gY2Fwc3VsZSBuZXR3b3JrcyBhbmQgZHluYW1pYyByb3V0aW5nIGRpZmZlciBmcm9tIENOTnM/Ij7wn6esIENhcHN1bGUgbmV0d29ya3M8L2J1dHRvbj4KICAgICAgICA8YnV0dG9uIGNsYXNzPSJjaGlwIiBkYXRhLXE9IkhvdyBkb2VzIEdyYWQtQ0FNIG1ha2UgbW9kZWwgcHJlZGljdGlvbnMgaW50ZXJwcmV0YWJsZT8iPvCflKwgR3JhZC1DQU08L2J1dHRvbj4KICAgICAgPC9kaXY+CiAgICAgIDxidXR0b24gY2xhc3M9ImFzay1idG4iIGlkPSJhc2siPkFzazwvYnV0dG9uPgogICAgPC9zZWN0aW9uPgoKICAgIDxzZWN0aW9uIGlkPSJvdXQiPjwvc2VjdGlvbj4KCiAgICA8cCBjbGFzcz0iZm9vdCI+VG9yY2gtZnJlZSBWZXJjZWwgZWRpdGlvbiDCtyB0aGUgZnVsbCBhcHAgKHNlbWFudGljIGVtYmVkZGluZ3MsIGFnZW50LCBldmFsdWF0aW9uKSBpcyBvbgogICAgICA8YSBocmVmPSJodHRwczovL2dpdGh1Yi5jb20vU2lkZGhhcnRoLUdvbGxhNi9zY2hvbGFyLXJhZyIgdGFyZ2V0PSJfYmxhbmsiIHJlbD0ibm9vcGVuZXIiPkdpdEh1YjwvYT4uPC9wPgogIDwvbWFpbj4KCjxzY3JpcHQ+CiAgY29uc3QgJCA9IChzKSA9PiBkb2N1bWVudC5xdWVyeVNlbGVjdG9yKHMpOwogIGNvbnN0IHEgPSAkKCIjcSIpLCBvdXQgPSAkKCIjb3V0IiksIGFza0J0biA9ICQoIiNhc2siKTsKCiAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgiLmNoaXAiKS5mb3JFYWNoKChjKSA9PgogICAgYy5hZGRFdmVudExpc3RlbmVyKCJjbGljayIsICgpID0+IHsgcS52YWx1ZSA9IGMuZGF0YXNldC5xOyBxLmZvY3VzKCk7IH0pCiAgKTsKCiAgY29uc3QgZXNjYXBlSHRtbCA9IChzKSA9PiBzLnJlcGxhY2UoL1smPD5dL2csIChtKSA9PiAoeyAiJiI6ICImYW1wOyIsICI8IjogIiZsdDsiLCAiPiI6ICImZ3Q7IiB9W21dKSk7CiAgY29uc3QgcmVuZGVyQW5zd2VyID0gKHQpID0+CiAgICBlc2NhcGVIdG1sKHQpLnJlcGxhY2UoL1tcW+OAkF0oXGQrKVtcXeOAkV0vZywgJzxzdXA+WyQxXTwvc3VwPicpLnJlcGxhY2UoL1wqXCooLis/KVwqXCovZywgIjxzdHJvbmc+JDE8L3N0cm9uZz4iKTsKCiAgYXN5bmMgZnVuY3Rpb24gYXNrKCkgewogICAgY29uc3QgcXVlc3Rpb24gPSBxLnZhbHVlLnRyaW0oKTsKICAgIGlmICghcXVlc3Rpb24pIHsgcS5mb2N1cygpOyByZXR1cm47IH0KICAgIGFza0J0bi5kaXNhYmxlZCA9IHRydWU7CiAgICBvdXQuc3R5bGUuZGlzcGxheSA9ICJibG9jayI7CiAgICBvdXQuaW5uZXJIVE1MID0gJzxkaXYgY2xhc3M9ImNhcmQiPjxzcGFuIGNsYXNzPSJzcGlubmVyIj48L3NwYW4+VGhpbmtpbmfigKY8L2Rpdj4nOwogICAgdHJ5IHsKICAgICAgY29uc3QgcmVzID0gYXdhaXQgZmV0Y2goIi9hcGkvYXNrIiwgewogICAgICAgIG1ldGhvZDogIlBPU1QiLAogICAgICAgIGhlYWRlcnM6IHsgIkNvbnRlbnQtVHlwZSI6ICJhcHBsaWNhdGlvbi9qc29uIiB9LAogICAgICAgIGJvZHk6IEpTT04uc3RyaW5naWZ5KHsgcXVlc3Rpb24gfSksCiAgICAgIH0pOwogICAgICBjb25zdCBkYXRhID0gYXdhaXQgcmVzLmpzb24oKTsKICAgICAgaWYgKGRhdGEuZXJyb3IpIHsKICAgICAgICBvdXQuaW5uZXJIVE1MID0gJzxkaXYgY2xhc3M9ImVyciI+JyArIGVzY2FwZUh0bWwoZGF0YS5lcnJvcikgKyAiPC9kaXY+IjsKICAgICAgICByZXR1cm47CiAgICAgIH0KICAgICAgbGV0IGh0bWwgPSAnPGRpdiBjbGFzcz0iY2FyZCI+PGRpdiBjbGFzcz0iYW5zd2VyIj4nICsgcmVuZGVyQW5zd2VyKGRhdGEuYW5zd2VyKSArICI8L2Rpdj4iOwogICAgICBodG1sICs9ICc8ZGl2IGNsYXNzPSJtZXRhIj5yZXRyaWV2YWwgKyBHcm9xIMK3ICcgKyAoZGF0YS5jb250ZXh0cyA/IGRhdGEuY29udGV4dHMubGVuZ3RoIDogMCkgKyAiIHNvdXJjZXM8L2Rpdj48L2Rpdj4iOwogICAgICBpZiAoZGF0YS5jaXRhdGlvbnMgJiYgZGF0YS5jaXRhdGlvbnMubGVuZ3RoKSB7CiAgICAgICAgaHRtbCArPSAnPGgzIGNsYXNzPSJzZWMiPkNpdGF0aW9uczwvaDM+JzsKICAgICAgICBkYXRhLmNpdGF0aW9ucy5mb3JFYWNoKChjKSA9PiB7CiAgICAgICAgICBodG1sICs9ICc8ZGl2IGNsYXNzPSJjaXRlIj48Yj5bJyArIGMubWFya2VyICsgIl08L2I+ICIgKyBlc2NhcGVIdG1sKGMuc291cmNlKSArICI8L2Rpdj4iOwogICAgICAgIH0pOwogICAgICB9CiAgICAgIG91dC5pbm5lckhUTUwgPSBodG1sOwogICAgfSBjYXRjaCAoZSkgewogICAgICBvdXQuaW5uZXJIVE1MID0gJzxkaXYgY2xhc3M9ImVyciI+UmVxdWVzdCBmYWlsZWQ6ICcgKyBlc2NhcGVIdG1sKFN0cmluZyhlKSkgKyAiPC9kaXY+IjsKICAgIH0gZmluYWxseSB7CiAgICAgIGFza0J0bi5kaXNhYmxlZCA9IGZhbHNlOwogICAgfQogIH0KICBhc2tCdG4uYWRkRXZlbnRMaXN0ZW5lcigiY2xpY2siLCBhc2spOwogIHEuYWRkRXZlbnRMaXN0ZW5lcigia2V5ZG93biIsIChlKSA9PiB7IGlmICgoZS5tZXRhS2V5IHx8IGUuY3RybEtleSkgJiYgZS5rZXkgPT09ICJFbnRlciIpIGFzaygpOyB9KTsKPC9zY3JpcHQ+CjwvYm9keT4KPC9odG1sPgo=").decode()


class handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        body = INDEX_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
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
