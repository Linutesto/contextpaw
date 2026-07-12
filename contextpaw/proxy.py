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

import asyncio
import json
import time

from aiohttp import web, ClientSession, ClientTimeout
from aiohttp.client_exceptions import ClientConnectionResetError

from .compact import compact_messages, compact_prompt, inject_digest, TooLongToCompact
from .runtime import RuntimeManager, RuntimePinned
from .summarize import Summarizer
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
        summarize=False,
        summarizer_model="gemma3:1b",
        summarizer_url=None,
        arbitrate=False,
        llamacpp_cmd=None,
        min_hold=20.0,
        no_think=False,
    ):
        self.ollama = ollama.rstrip("/")
        self.llamacpp = llamacpp.rstrip("/")
        self.policy = policy
        self.default_ctx = default_ctx
        self.margin = margin
        self.keep_recent = keep_recent
        self.counter = HFCounter(tokenizer) if tokenizer else CalibratedCounter()
        self.session: ClientSession | None = None
        self.stats = {"requests": 0, "compacted": 0, "refused": 0, "retried_400": 0,
                      "client_disconnects": 0}

        self.summarize_enabled = summarize
        self.summarizer_model = summarizer_model
        self.summarizer_url = summarizer_url or ollama
        self.summarizer: Summarizer | None = None

        self.arbitrate = arbitrate
        self.llamacpp_cmd = llamacpp_cmd
        self.min_hold = min_hold
        self.rt: RuntimeManager | None = None

        # Thinking models (gemma4, qwen3.x) put their tokens in `thinking` and return an
        # EMPTY `response`. An app that doesn't know about the field just gets "" and has
        # no idea why -- another silent lie. This injects `think: false` for clients that
        # never asked for thinking, so they get text instead of nothing.
        self.no_think = no_think

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

    async def _apply(self, body: dict, backend: str = "ollama") -> tuple[dict, dict | None]:
        """Preflight + compact. Returns (possibly rewritten body, report|None)."""
        if self.no_think:
            # The two backends spell "don't think" completely differently, and each one
            # fails the same silent way if you get it wrong: Ollama returns an empty
            # `response`, llama.cpp returns an empty `content` with the text stranded in
            # `reasoning_content`. Sending Ollama's flag to llama.cpp does nothing at all
            # -- so the option would quietly lie about what it did.
            if backend == "ollama":
                if "think" not in body:
                    body = {**body, "think": False}
            else:
                kw = dict(body.get("chat_template_kwargs") or {})
                if "enable_thinking" not in kw:
                    kw["enable_thinking"] = False
                    body = {**body, "chat_template_kwargs": kw}

        if self.policy == "off":
            return body, None

        num_ctx, budget = self._budget(body)

        if "messages" in body:
            msgs, rep = compact_messages(
                body["messages"], budget, self._count, keep_recent=self.keep_recent
            )
            if rep.compacted:
                msgs = await self._maybe_summarize(msgs, rep, budget)
                body = {**body, "messages": msgs}
        elif "prompt" in body and isinstance(body["prompt"], str):
            text, rep = compact_prompt(body["prompt"], budget, self._count)
            if rep.compacted:
                text = await self._maybe_summarize(text, rep, budget)
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

    async def _maybe_summarize(self, payload, rep, budget):
        """Replace the 'this is gone' marker with an actual digest of what went."""
        if not self.summarizer or not rep.evicted_text:
            return payload
        digest = await self.summarizer.summarize(rep.evicted_text)
        if not digest:
            return payload  # best-effort: keep the plain marker, never fail the request

        out = inject_digest(payload, rep, digest)

        # The digest costs tokens too. If it pushed us back over budget, drop it --
        # a summary that re-overflows the window would recreate the very bug we exist
        # to prevent.
        if isinstance(out, str):
            size = self._count(out)
        else:
            size = sum(self._count(str(m.get("content", ""))) for m in out)
        if size > budget:
            rep.summarized = False
            return payload
        rep.final_tokens = size
        return out

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

        # OpenAI-shaped path (/v1/*).
        #
        # When the arbiter is running, route to llama.cpp even if it is currently DOWN --
        # bringing it up is precisely the arbiter's job. Gating on `_llamacpp_up` (a flag
        # sampled once at startup) would mean llama.cpp never gets started: every request
        # would see it down, fall back to Ollama, and the arbiter would never fire. The
        # snake eats its own tail.
        if self.arbitrate and self.llamacpp_cmd:
            return self.llamacpp
        return self.llamacpp if self._llamacpp_up else self.ollama

    _llamacpp_up = False

    async def handle(self, request: web.Request) -> web.StreamResponse:
        self.stats["requests"] += 1
        try:
            body = await request.json()
        except Exception:
            return await self._passthrough(request, None)

        if self.rt and self.rt.enabled:
            want = "llamacpp" if self._upstream(request) == self.llamacpp else "ollama"
            try:
                await self.rt.ensure(want)
            except RuntimePinned as e:
                return web.json_response(
                    {"error": {"type": "runtime_pinned", "message": str(e),
                               "pinned": self.rt.pinned, "wanted": want}},
                    status=409,
                )
            except Exception as e:
                return web.json_response(
                    {"error": {"type": "runtime_switch_failed", "message": str(e)}},
                    status=503,
                )

        backend = "llamacpp" if self._upstream(request) == self.llamacpp else "ollama"
        try:
            body, report = await self._apply(body, backend)
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
            try:
                async for chunk in up.content.iter_any():
                    await out.write(chunk)
                await out.write_eof()
            except (ConnectionResetError, ClientConnectionResetError, asyncio.CancelledError):
                # The caller hung up mid-stream. This is NORMAL, not an error: a game
                # cancels an NPC line when the player walks away, an agent aborts a turn.
                # Measured in production: 4 tracebacks in one Skyrim session, all benign.
                self.stats["client_disconnects"] += 1
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
        """
        Learn the real chars->tokens ratio from what the server actually read.

        The two backends report this under DIFFERENT names, and getting that wrong is
        not a cosmetic bug: it means the counter silently never learns. Measured in
        production -- 55 real requests through llama.cpp, `samples: 0`, because we were
        only looking for Ollama's field. The self-calibration this tool advertises was
        simply not running on half its backends.

            Ollama     -> prompt_eval_count
            llama.cpp  -> usage.prompt_tokens   (OpenAI shape)
        """
        actual = data.get("prompt_eval_count") or (data.get("usage") or {}).get(
            "prompt_tokens"
        )
        if not actual:
            return
        text = (
            body.get("prompt")
            if isinstance(body.get("prompt"), str)
            else "".join(str(m.get("content", "")) for m in body.get("messages", []))
        )
        if text:
            self.counter.observe(text, actual)

    async def _passthrough(self, request, _=None):
        """
        Everything we do not compact (/api/tags, /api/pull, /api/embed, ...) is relayed
        verbatim.

        This STREAMS. Buffering the whole response would be invisible on /api/tags and
        infuriating on /api/pull, where the body IS a progress stream -- the download bar
        would freeze and then vomit at the end. If ContextPaw is going to sit on port 11434
        in front of a real stack, "transparent" has to mean it, for every endpoint we are
        not deliberately touching.
        """
        assert self.session
        url = self._upstream(request) + request.path_qs
        raw = await request.read()

        hdrs = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "accept-encoding")
        }

        async with self.session.request(
            request.method, url, data=raw or None, headers=hdrs, allow_redirects=False
        ) as up:
            out = web.StreamResponse(status=up.status)
            ct = up.headers.get("Content-Type")
            if ct:
                out.headers["Content-Type"] = ct
            await out.prepare(request)
            try:
                async for chunk in up.content.iter_any():
                    await out.write(chunk)
                await out.write_eof()
            except (ConnectionResetError, ClientConnectionResetError, asyncio.CancelledError):
                self.stats["client_disconnects"] += 1
            return out

    async def models(self, request):
        """
        GET /v1/models — but with the ONE number an agent harness actually needs.

        A harness cannot manage context it cannot measure, and neither backend tells it
        the truth:

          Ollama     /v1/models returns {id, object, created, owned_by}. No context length
                     at all. It *knows* the value -- /api/show reports
                     gemma4.context_length: 131072 -- it just never puts it on the OpenAI
                     endpoint.
          llama.cpp  reports meta.n_ctx correctly, but under a non-standard key.

        And the advertised number is a trap anyway: 131072 is what the model was TRAINED
        with. The runtime had actually loaded it at 16384. A harness trusting the
        advertised figure overruns by 8x and gets silently truncated for its trouble.

        So we report `context_length` = what is ACTUALLY LOADED right now, plus
        `contextpaw.budget` = what we will let a prompt occupy after reserving room for
        the reply. That is the number a compaction step should be planning against.
        """
        backend = self._upstream(request)
        upstream = backend + "/v1/models"
        try:
            async with self.session.get(upstream) as up:
                data = await up.json()
        except Exception as e:
            return web.json_response({"error": str(e)}, status=502)

        effective = await self._effective_ctx(backend)
        for m in data.get("data", []) or data.get("models", []) or []:
            if not isinstance(m, dict):
                continue
            trained = ((m.get("meta") or {}).get("n_ctx_train")) or None
            if effective:
                m["context_length"] = effective
                m["contextpaw"] = {
                    "context_length_loaded": effective,
                    "context_length_trained": trained,
                    "budget": max(effective - 512 - self.margin, 512),
                    "note": (
                        "context_length is what the runtime ACTUALLY loaded, not what the "
                        "model was trained with. Plan compaction against `budget`."
                    ),
                }
        return web.json_response(data)

    async def _effective_ctx(self, backend: str) -> int | None:
        """The context the model is loaded with RIGHT NOW — not its advertised maximum."""
        try:
            if backend == self.llamacpp:
                async with self.session.get(f"{self.llamacpp}/props") as r:
                    d = await r.json()
                return (d.get("default_generation_settings") or {}).get("n_ctx")
            async with self.session.get(f"{self.ollama}/api/ps") as r:
                d = await r.json()
            for m in d.get("models", []):
                ctx = m.get("context_length") or m.get("context")
                if ctx:
                    return int(ctx)
        except Exception:
            pass
        return self.default_ctx  # nothing loaded yet: what we would budget against

    async def health(self, _):
        s = {"status": "ok", "policy": self.policy, "stats": self.stats,
             "counter": self.counter.name, "exact": self.counter.exact()}
        if hasattr(self.counter, "stats"):
            s["calibration"] = self.counter.stats()
        if self.summarizer:
            s["summarizer"] = {"model": self.summarizer.model, **self.summarizer.stats}
        if self.rt:
            s["runtime"] = await self.rt.status()
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

        if paw.summarize_enabled:
            paw.summarizer = Summarizer(
                paw.session, url=paw.summarizer_url, model=paw.summarizer_model
            )
        paw.rt = RuntimeManager(
            paw.session,
            ollama_url=paw.ollama,
            llamacpp_url=paw.llamacpp,
            llamacpp_cmd=paw.llamacpp_cmd,
            min_hold=paw.min_hold,
            enabled=paw.arbitrate,
        )
        if paw.arbitrate and paw._llamacpp_up:
            paw.rt.active = "llamacpp"

    async def _cleanup(_):
        if paw.session:
            await paw.session.close()

    app.on_startup.append(_startup)
    app.on_cleanup.append(_cleanup)

    app.router.add_get("/contextpaw/health", paw.health)
    app.router.add_get("/v1/models", paw.models)
    app.router.add_get("/contextpaw/runtime", lambda r: _runtime_status(paw, r))
    app.router.add_post("/contextpaw/runtime", lambda r: _runtime_pin(paw, r))
    for p in ("/api/generate", "/api/chat", "/v1/completions", "/v1/chat/completions"):
        app.router.add_post(p, paw.handle)
    app.router.add_route("*", "/{tail:.*}", paw._passthrough)
    return app


async def _runtime_status(paw: ContextPaw, _):
    if not paw.rt:
        return web.json_response({"enabled": False})
    return web.json_response(await paw.rt.status())


async def _runtime_pin(paw: ContextPaw, request):
    """POST {"pin": "llamacpp"} to take the GPU off the table; {"pin": null} to release."""
    if not paw.rt or not paw.rt.enabled:
        return web.json_response({"error": "arbiter is not enabled"}, status=400)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if "pin" not in body:
        return web.json_response(
            {"error": 'send {"pin": "llamacpp"|"ollama"|null}'}, status=400
        )
    try:
        return web.json_response(await paw.rt.pin(body["pin"]))
    except ValueError:
        return web.json_response({"error": f"unknown backend: {body['pin']}"}, status=400)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=503)
