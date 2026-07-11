"""contextpaw serve — run the proxy."""
from __future__ import annotations
import argparse
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
    a = p.parse_args()

    paw = ContextPaw(
        ollama=a.ollama, llamacpp=a.llamacpp, policy=a.policy,
        default_ctx=a.default_ctx, margin=a.margin,
        tokenizer=a.tokenizer, keep_recent=a.keep_recent,
    )
    print(f"contextpaw :{a.port}  policy={a.policy}  ollama={a.ollama}  llamacpp={a.llamacpp}")
    web.run_app(build_app(paw), host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
