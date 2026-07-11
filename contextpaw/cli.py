"""contextpaw serve — run the proxy."""
from __future__ import annotations
import argparse
import shlex
from aiohttp import web
from .proxy import ContextPaw, build_app


def main() -> None:
    p = argparse.ArgumentParser("contextpaw")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=11434,
                   help="take Ollama's port so no client needs changing")
    p.add_argument("--ollama", default="http://127.0.0.1:11435",
                   help="where the REAL ollama listens (move it off 11434)")
    p.add_argument("--llamacpp", default="http://127.0.0.1:8091")
    p.add_argument("--policy", choices=["compact", "strict", "off"], default="compact",
                   help="compact: shrink and report. strict: refuse loudly. off: passthrough.")
    p.add_argument("--default-ctx", type=int, default=8192)
    p.add_argument("--margin", type=int, default=256)
    p.add_argument("--keep-recent", type=int, default=4)
    p.add_argument("--tokenizer", default=None,
                   help="optional HF repo/path for EXACT counts (else self-calibrating)")

    # summarize evicted context instead of just dropping it
    p.add_argument("--summarize", action="store_true",
                   help="digest the evicted span with a small model instead of dropping it")
    p.add_argument("--summarizer-model", default="gemma3:1b",
                   help="small, fast model for the digest (default: gemma3:1b)")
    p.add_argument("--summarizer-url", default=None,
                   help="where the summarizer model lives (default: the --ollama url)")

    # VRAM arbiter: exactly one runtime owns the GPU
    p.add_argument("--arbitrate", action="store_true",
                   help="evict the other runtime before serving; one runtime owns the GPU")
    p.add_argument("--llamacpp-cmd", default=None,
                   help="command to start llama-server, so the arbiter can bring it up "
                        "(quote it: --llamacpp-cmd '/path/llama-server -m ... --port 8091')")
    p.add_argument("--min-hold", type=float, default=20.0,
                   help="seconds a runtime keeps the GPU before it can be switched away")
    a = p.parse_args()

    paw = ContextPaw(
        ollama=a.ollama, llamacpp=a.llamacpp, policy=a.policy,
        default_ctx=a.default_ctx, margin=a.margin,
        tokenizer=a.tokenizer, keep_recent=a.keep_recent,
        summarize=a.summarize, summarizer_model=a.summarizer_model,
        summarizer_url=a.summarizer_url,
        arbitrate=a.arbitrate,
        llamacpp_cmd=shlex.split(a.llamacpp_cmd) if a.llamacpp_cmd else None,
        min_hold=a.min_hold,
    )
    extra = []
    if a.summarize:
        extra.append(f"summarize={a.summarizer_model}")
    if a.arbitrate:
        extra.append(f"arbitrate(min_hold={a.min_hold}s)")
    print(f"contextpaw :{a.port}  policy={a.policy}  ollama={a.ollama}  "
          f"llamacpp={a.llamacpp}  {' '.join(extra)}")
    web.run_app(build_app(paw), host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
