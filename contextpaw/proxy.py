"""
ContextPaw proxy.

Sits in front of Ollama and/or llama.cpp and makes context overflow HONEST.

    client ──► contextpaw :11434 ──► ollama    :11435
                                └──► llama.cpp :8091

What it fixes, per backend:

  Ollama     silently truncates an oversized prompt, returns 200 OK, and lets the model
             invent an answer over the hole. We preflight the token count and compact
             deliberately (keeping the head) instead, or refuse loudly in strict mode.

  llama.cpp  returns a hard 400 `exceed_context_size_error` that kills the agent turn.
             We catch it, compact, and retry once, so the agent survives.

Same box, opposite failures. Both become: fit the window, keep what matters, and TELL
THE CALLER what was lost.
"""
from __future__ import annotations

import json
import time

from aiohttp import web, ClientSession, ClientTimeout

from .compact import compact_messages, compact_prompt, TooLongToCompact
from .tokens import CalibratedCounter, HFCounter

DEFAULT_MARGIN = 256  # headroom for the chat template, BOS, and our own marker


class ContextPaw:
    def __init__(
        self,
        ollama="http://127.0.0.1:11435",
        llamacpp="http://127.0.0.1:8091",
        policy="compact",            # compact | strict | off
        default_ctx=8192,
        margin=DEFAULT_MARGIN,
        tokenizer=None,
        keep_recent=4,
    ):
        self.ollama = ollama.rstrip("/")
        self.llamacpp = llamacpp.rstrip("/")
        self.policy = policy
        self.default_ctx = default_ctx
        self.margin = margin
        self.keep_recent = keep_recent
        self.counter = HFCounter(tokenizer) if tokenizer else CalibratedCounter()
        self.session: ClientSession | None = None
        self.stats = {"requests": 0, "compacted": 0, "refused": 0, "retried_400": 0}

    # ---------- budgeting ----------

    def _budget(self, body: dict) -> tuple[int, int]:
        opts = body.get("options") or {}
        num_ctx = opts.get("num_ctx") or body.get("num_ctx") or self.default_ctx
        num_predict = (
            opts.get("num_predict")
            or body.get("max_tokens")
            or body.get("num_predict")
            or 512
        )
        if num_predict < 0:  # -1 / -2 mean "until the window is full"
            num_predict = num_ctx // 4
        return num_ctx, max(num_ctx - num_predict - self.margin, 512)

    def _count(self, text: str) -> int:
        return self.counter.count(text)

    # ---------- the actual work ----------

    def _apply(self, body: dict) -> tuple[dict, dict | None]:
        """Preflight + compact. Returns (possibly rewritten body, report|None)."""
        if self.policy == "off":
            return body, None

        num_ctx, budget = self._budget(body)

        if "messages" in body:
            msgs, rep = compact_messages(
                body["messages"], budget, self._count, keep_recent=self.keep_recent
            )
            if rep.compacted:
                body = {**body, "messages": msgs}
        elif "prompt" in body and isinstance(body["prompt"], str):
            text, rep = compact_prompt(body["prompt"], budget, self._count)
            if rep.compacted:
                body = {**body, "prompt": text}
        else:
            return body, None

        rep.exact_count = self.counter.exact()
        if not rep.compacted:
            return body, None

        if self.policy == "strict":
            self.stats["refused"] += 1
            raise web.HTTPRequestEntityTooLarge(
                max_size=budget,
                actual_size=rep.original_tokens,
                text=json.dumps(
                    {
                        "error": {
                            "type": "context_overflow",
                            "message": (
                                f"prompt is {rep.original_tokens} tokens, budget is {budget} "
                                f"(num_ctx={num_ctx}). Refusing rather than truncating silently. "
                                f"Use policy=compact to have ContextPaw compact it for you."
                            ),
                            "n_prompt_tokens": rep.original_tokens,
                            "budget": budget,
                            "num_ctx": num_ctx,
                        }
                    }
                ),
                content_type="application/json",
            )

        self.stats["compacted"] += 1
        return body, rep.as_dict()

    # ---------- routing ----------

    def _upstream(self, request) -> str:
        # explicit override wins; otherwise Ollama-shaped paths go to Ollama
        forced = request.headers.get("X-ContextPaw-Backend")
        if forced == "llamacpp":
            return self.llamacpp
        if forced == "ollama":
            return self.ollama
        if request.path.startswith("/api/"):
            return self.ollama
        return self.llamacpp if self._llamacpp_up else self.ollama

    _llamacpp_up = False

    async def handle(self, request: web.Request) -> web.StreamResponse:
        self.stats["requests"] += 1
        try:
            body = await request.json()
        except Exception:
            return await self._passthrough(request, None)

        try:
            body, report = self._apply(body)
        except TooLongToCompact as e:
            self.stats["refused"] += 1
            return web.json_response(
                {
                    "error": {
                        "type": "context_overflow_uncompactable",
                        "message": str(e),
                    }
                },
                status=413,
            )

        upstream = self._upstream(request) + request.path
        streaming = bool(body.get("stream"))

        resp = await self._forward(upstream, body, streaming, request, report)
        return resp

    async def _forward(self, url, body, streaming, request, report):
        assert self.session
        headers = {"Content-Type": "application/json"}

        async with self.session.post(url, json=body, headers=headers) as up:
            # --- llama.cpp's hard 400: compact and retry once instead of dying ---
            if up.status == 400:
                raw = await up.text()
                if "exceed_context_size" in raw and self.policy == "compact":
                    self.stats["retried_400"] += 1
                    d = json.loads(raw).get("error", {})
                    real_ctx = d.get("n_ctx") or self.default_ctx
                    # trust the server's own numbers over our estimate, then retry
                    body2 = dict(body)
                    body2.setdefault("options", {})
                    tight = max(real_ctx - 512, 512)
                    if "messages" in body2:
                        msgs, rep = compact_messages(
                            body2["messages"], tight, self._count,
                            keep_recent=self.keep_recent,
                        )
                        body2["messages"] = msgs
                    else:
                        text, rep = compact_prompt(body2["prompt"], tight, self._count)
                        body2["prompt"] = text
                    rep_d = rep.as_dict()
                    rep_d["recovered_from_upstream_400"] = True
                    return await self._forward(url, body2, streaming, request, rep_d)
                return web.Response(
                    body=raw.encode(), status=400, content_type="application/json"
                )

            if not streaming:
                data = await up.json()
                self._calibrate(body, data)
                if report:
                    data["contextpaw"] = report
                return web.json_response(data, status=up.status, headers=self._hdrs(report))

            out = web.StreamResponse(status=up.status, headers=self._hdrs(report))
            out.content_type = up.content_type or "application/json"
            await out.prepare(request)
            async for chunk in up.content.iter_any():
                await out.write(chunk)
            await out.write_eof()
            return out

    def _hdrs(self, report) -> dict:
        h = {"X-ContextPaw": "1"}
        if report:
            h["X-ContextPaw-Compacted"] = "true"
            h["X-ContextPaw-Evicted-Tokens"] = str(report.get("evicted_tokens", 0))
        else:
            h["X-ContextPaw-Compacted"] = "false"
        return h

    def _calibrate(self, body, data):
        """Learn the real chars->tokens ratio from what the server actually read."""
        actual = data.get("prompt_eval_count")
        if not actual:
            return
        text = (
            body.get("prompt")
            if isinstance(body.get("prompt"), str)
            else "".join(str(m.get("content", "")) for m in body.get("messages", []))
        )
        if text:
            self.counter.observe(text, actual)

    async def _passthrough(self, request, _):
        assert self.session
        url = self._upstream(request) + request.path
        raw = await request.read()
        async with self.session.request(
            request.method, url, data=raw, headers={"Content-Type": "application/json"}
        ) as up:
            return web.Response(
                body=await up.read(), status=up.status, content_type=up.content_type
            )

    async def health(self, _):
        s = {"status": "ok", "policy": self.policy, "stats": self.stats,
             "counter": self.counter.name, "exact": self.counter.exact()}
        if hasattr(self.counter, "stats"):
            s["calibration"] = self.counter.stats()
        return web.json_response(s)


def build_app(paw: ContextPaw) -> web.Application:
    app = web.Application(client_max_size=1024 ** 3)

    async def _startup(_):
        paw.session = ClientSession(timeout=ClientTimeout(total=1800))
        try:
            async with paw.session.get(f"{paw.llamacpp}/health") as r:
                paw._llamacpp_up = r.status == 200
        except Exception:
            paw._llamacpp_up = False

    async def _cleanup(_):
        if paw.session:
            await paw.session.close()

    app.on_startup.append(_startup)
    app.on_cleanup.append(_cleanup)

    app.router.add_get("/contextpaw/health", paw.health)
    for p in ("/api/generate", "/api/chat", "/v1/completions", "/v1/chat/completions"):
        app.router.add_post(p, paw.handle)
    app.router.add_route("*", "/{tail:.*}", paw._passthrough)
    return app
