"""Provider-agnostic LLM client over the OpenAI-compatible Chat Completions API.

Works with Groq (default, free tier), OpenAI, or any local/self-hosted
OpenAI-compatible server (Ollama, LM Studio, vLLM) by changing only .env.
"""
from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Iterator, Optional

from openai import APIConnectionError, APITimeoutError, RateLimitError

from .config import Settings, get_settings
from .logging_utils import get_logger

log = get_logger("llm")


class LLMError(RuntimeError):
    """Raised for configuration or upstream API failures."""


class LLMClient:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        if not self.settings.has_llm_key:
            raise LLMError(
                f"No API key configured for provider '{self.settings.llm_provider}'. "
                "Add GROQ_API_KEY to your .env (free key: https://console.groq.com/keys)."
            )
        from openai import OpenAI

        kwargs: dict = {"api_key": self.settings.llm_api_key or "not-needed"}
        if self.settings.llm_base_url:
            kwargs["base_url"] = self.settings.llm_base_url
        self.client = OpenAI(**kwargs)
        self.model = self.settings.llm_model
        log.info("LLM client ready: provider=%s model=%s", self.settings.llm_provider, self.model)

    def complete(self, messages, tools=None, tool_choice=None,
                 temperature: float | None = None, max_tokens: int | None = None,
                 response_format: dict | None = None):
        """Return the raw assistant message (may include ``tool_calls``)."""
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            temperature=self.settings.llm_temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.settings.llm_max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if response_format:
            kwargs["response_format"] = response_format
        try:
            resp = self._with_retry(lambda: self.client.chat.completions.create(**kwargs))
        except Exception as exc:  # normalise provider errors
            raise LLMError(str(exc)) from exc
        return resp.choices[0].message

    def chat(self, messages, **kwargs) -> str:
        return self.complete(messages, **kwargs).content or ""

    def stream(self, messages, temperature: float | None = None,
               max_tokens: int | None = None) -> Iterator[str]:
        try:
            stream = self._with_retry(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.settings.llm_temperature if temperature is None else temperature,
                    max_tokens=max_tokens or self.settings.llm_max_tokens,
                    stream=True,
                )
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            raise LLMError(str(exc)) from exc

    def _with_retry(self, fn, max_retries: int = 5):
        """Retry transient failures (429 rate limits, connection blips) with
        exponential backoff, honouring the provider's suggested wait when given."""
        delay = 2.0
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
                if attempt >= max_retries:
                    raise
                wait = self._retry_after(exc) or delay
                log.warning(
                    "Transient LLM error (%s); retrying in %.1fs [%d/%d]",
                    type(exc).__name__, wait, attempt + 1, max_retries,
                )
                time.sleep(wait)
                delay = min(delay * 2, 30.0)

    @staticmethod
    def _retry_after(exc) -> float | None:
        try:
            retry_after = exc.response.headers.get("retry-after")
            if retry_after:
                return float(retry_after) + 0.3
        except Exception:
            pass
        match = re.search(r"try again in ([\d.]+)\s*s", str(exc))
        return float(match.group(1)) + 0.5 if match else None


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    return LLMClient()
