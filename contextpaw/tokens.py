"""
Token counting that works on ANY model without shipping a tokenizer.

Three strategies, in order of preference:

  1. LlamaCppCounter  — POST /tokenize on a llama.cpp server. Exact, free.
  2. HFCounter        — a real tokenizer via `tokenizers`. Exact. Opt-in.
  3. Calibrated       — heuristic chars->tokens ratio that CALIBRATES ITSELF against
                        the `prompt_eval_count` the server reports back. Works on any
                        model, zero downloads.

The calibrated counter is the interesting one. Ollama tells you exactly how many prompt
tokens it evaluated. We use that as ground truth to correct our own estimate, and we
always keep the WORST (highest tokens-per-char) ratio we have ever seen, so the estimate
errs toward over-counting. Over-counting is safe: it makes us compact slightly too early.
Under-counting is what gets your prompt silently truncated.
"""
from __future__ import annotations

import threading


class Counter:
    name = "base"

    def count(self, text: str) -> int:
        raise NotImplementedError

    def exact(self) -> bool:
        return False

    def observe(self, text: str, actual_tokens: int) -> None:
        """Feed back ground truth from a server response. No-op for exact counters."""


class CalibratedCounter(Counter):
    """Heuristic + self-calibration against server-reported prompt_eval_count."""

    name = "calibrated"

    # Conservative starting point. Real ratios sit around 0.25-0.30 tok/char for
    # English, higher for code and for languages with heavy accent usage.
    INITIAL_RATIO = 0.34
    SAFETY = 1.05

    def __init__(self) -> None:
        self._ratio = self.INITIAL_RATIO
        self._samples = 0
        self._lock = threading.Lock()

    def count(self, text: str) -> int:
        with self._lock:
            ratio = self._ratio
        return int(len(text) * ratio * self.SAFETY) + 1

    # Below this, the chat template's fixed overhead dominates the count and the
    # measured ratio is garbage. Found in production: a 26-char "say hello" prompt
    # reported ~19 prompt_eval tokens -> 0.73 tok/char, which (because we keep the
    # MAX) permanently poisoned the ratio and made everything compact far too early.
    # One short prompt was enough to do it.
    MIN_SAMPLE_CHARS = 1000

    def observe(self, text: str, actual_tokens: int) -> None:
        # Only learn from samples that look like a full, uncached prompt evaluation.
        # A cache hit reports FEWER tokens than were really in the prompt; learning
        # from that would teach us to under-count, which is the one failure we cannot
        # afford. Requiring the sample to raise the ratio filters those out for free.
        if not text or actual_tokens <= 0 or len(text) < self.MIN_SAMPLE_CHARS:
            return
        observed = actual_tokens / len(text)
        with self._lock:
            self._samples += 1
            if observed > self._ratio:
                self._ratio = observed

    def stats(self) -> dict:
        with self._lock:
            return {"ratio": round(self._ratio, 4), "samples": self._samples}


class LlamaCppCounter(Counter):
    """Exact counts via llama.cpp's /tokenize. Costs one cheap HTTP call."""

    name = "llamacpp"

    def __init__(self, session, base_url: str) -> None:
        self._session = session
        self._base = base_url.rstrip("/")

    def exact(self) -> bool:
        return True

    async def acount(self, text: str) -> int:
        async with self._session.post(
            f"{self._base}/tokenize", json={"content": text}
        ) as r:
            r.raise_for_status()
            d = await r.json()
        return len(d["tokens"])

    def count(self, text: str) -> int:  # pragma: no cover - async path is the real one
        raise RuntimeError("use acount() for LlamaCppCounter")


class HFCounter(Counter):
    """Exact counts from a real tokenizer. Opt-in: --tokenizer <hf-repo-or-path>."""

    name = "hf"

    def __init__(self, repo_or_path: str) -> None:
        from tokenizers import Tokenizer  # local import: keep it an optional dep

        try:
            self._tok = Tokenizer.from_pretrained(repo_or_path)
        except Exception:
            self._tok = Tokenizer.from_file(repo_or_path)
        self.repo = repo_or_path

    def exact(self) -> bool:
        return True

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)
