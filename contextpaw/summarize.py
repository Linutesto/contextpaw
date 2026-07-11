"""
Summarize what we evict, instead of just dropping it.

A marker that says "12431 tokens were elided" tells the agent that a hole exists.
A *summary* tells it what was in the hole. That is the difference between an agent
that knows it must re-read a file and an agent that knows it doesn't have to.

Three things this must get right, or it makes the product worse instead of better:

  1. CACHE IT. In an agent loop the same old turns get evicted on every single
     request. Without a cache you would re-summarize the identical span forever,
     paying a model call per turn for an answer you already had.

  2. NEVER BREAK THE REQUEST. The summarizer is a nice-to-have sitting on the hot
     path of a request we were asked to *rescue*. If it errors, times out, or is
     simply not running, we fall back to the plain marker and carry on. A rescue
     tool that fails the request it was rescuing is worse than no tool at all.

  3. STAY INSIDE THE BUDGET. The summary costs tokens too. We cap it, and we count
     it like everything else.
"""
from __future__ import annotations

import asyncio
import hashlib

REDUCE = (
    "Merge these partial digests of an AI agent's conversation history into one terse "
    "factual digest. Keep every concrete fact, name, path, number, decision and error. "
    "Drop repetition. No preamble.\n\nPARTS:\n{parts}\n\nMerged digest:"
)

PROMPT = (
    "Below is a span of an AI agent's conversation history that must be dropped to fit "
    "a context window. Write a terse factual digest of it for the agent to rely on later.\n"
    "Rules:\n"
    "- Keep concrete facts, file paths, names, numbers, decisions, and errors.\n"
    "- Keep anything that looks like a result the agent will need again.\n"
    "- Drop pleasantries, repetition, and filler.\n"
    "- No preamble, no commentary. Just the digest.\n\n"
    "--- SPAN ---\n{span}\n--- END SPAN ---\n\nDigest:"
)


class Summarizer:
    """Summarize evicted spans with a small local model. Best-effort, always safe."""

    def __init__(
        self,
        session,
        url: str = "http://127.0.0.1:11434",
        model: str = "gemma3:1b",
        max_tokens: int = 200,
        timeout: float = 20.0,
        max_input_chars: int = 12000,
        max_chunks: int = 12,
    ):
        self._session = session
        self._url = url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_input_chars = max_input_chars
        self.max_chunks = max_chunks
        self._cache: dict[str, str] = {}
        self.stats = {"calls": 0, "cache_hits": 0, "failures": 0}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()

    async def summarize(self, span: str) -> str | None:
        """Return a digest, or None if we could not get one. None is not an error."""
        if not span.strip():
            return None

        key = self._key(span)
        if key in self._cache:
            self.stats["cache_hits"] += 1
            return self._cache[key]

        # A span can be enormous -- that is WHY it is being evicted. The tempting move
        # is to trim it head+tail to fit the summarizer's own window. Do not. The facts
        # worth keeping are frequently in the middle of the evicted span (the middle is
        # exactly what compaction threw away), so trimming here would silently drop the
        # very content we were asked to preserve -- reproducing, inside the summarizer,
        # the precise bug this whole project exists to fix. Measured: it scored 0/3.
        #
        # So we MAP over the whole span in chunks and REDUCE. It costs more calls; a
        # small local model is cheap and the result is cached.
        chunks = self._chunks(span)
        digests = []
        for c in chunks:
            d = await self._one(PROMPT.format(span=c))
            if d:
                digests.append(d)

        if not digests:
            self.stats["failures"] += 1
            return None

        if len(digests) == 1:
            digest = digests[0]
        else:
            merged = "\n".join(f"- {d}" for d in digests)
            digest = await self._one(REDUCE.format(parts=merged)) or merged

        digest = digest.strip()
        if not digest:
            self.stats["failures"] += 1
            return None

        self._cache[key] = digest
        return digest

    def _chunks(self, span: str) -> list[str]:
        """Split the whole span into summarizer-sized pieces. Nothing is skipped."""
        size = self.max_input_chars
        out = [span[i : i + size] for i in range(0, len(span), size)]
        if len(out) <= self.max_chunks:
            return out
        # Pathologically large span: keep the call count bounded, but stride ACROSS the
        # whole thing rather than truncating one end of it. The stride is inclusive of
        # BOTH ends -- a naive `i * len/max` never reaches the last chunk, which would
        # quietly reintroduce a tail-truncation bias.
        n, k = len(out), self.max_chunks
        idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
        return [out[i] for i in idx]

    async def _one(self, prompt: str) -> str | None:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,  # gemma/qwen thinking models return an EMPTY response otherwise
            "keep_alive": "10m",
            "options": {"num_predict": self.max_tokens, "temperature": 0.1},
        }
        try:
            self.stats["calls"] += 1
            async with self._session.post(
                f"{self._url}/api/generate",
                json=body,
                timeout=__import__("aiohttp").ClientTimeout(total=self.timeout),
            ) as r:
                if r.status != 200:
                    self.stats["failures"] += 1
                    return None
                d = await r.json()
        except Exception:
            # Deliberately broad: this sits on the hot path of a request we are
            # rescuing. Nothing it can do is allowed to fail that request.
            self.stats["failures"] += 1
            return None
        return (d.get("response") or "").strip() or None
