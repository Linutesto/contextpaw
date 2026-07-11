"""
VRAM arbiter: exactly one runtime owns the GPU at a time.

A 24 GB card cannot hold Ollama's model AND a llama.cpp server at once. Measured on an
RTX 4090: gemma-4-12b in Ollama at 4x32k = 15.1 GB, qwythos-9b in llama.cpp at 4x32k =
11.0 GB. 26.1 GB > 24 GB. They do not coexist.

The dangerous part is that nothing tells you this. Ollama will happily load a model on
top of a running llama.cpp server and, with GGML_CUDA_ENABLE_UNIFIED_MEMORY=1, spill to
system RAM *silently* -- 20x slower, and `ollama ps` will still cheerfully report
"100% GPU". Same disease as silent truncation: the system would rather lie than say no.

So ContextPaw arbitrates. It sees every request, so it knows which runtime the caller
actually wants; it evicts the other one first, then serves.

Measured switch cost on this hardware:
    llama.cpp -> Ollama    5.1s   (0.3s evict + 4.8s load)
    Ollama -> llama.cpp    2.3s   (0.7s evict + 1.5s start)
Cheap enough to do on demand. A `min_hold` guard stops it oscillating if two callers
fight over the card.
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time


class RuntimeManager:
    def __init__(
        self,
        session,
        ollama_url="http://127.0.0.1:11435",
        llamacpp_url="http://127.0.0.1:8091",
        llamacpp_cmd: list[str] | None = None,
        min_hold: float = 20.0,
        enabled: bool = False,
    ):
        self.session = session
        self.ollama_url = ollama_url.rstrip("/")
        self.llamacpp_url = llamacpp_url.rstrip("/")
        self.llamacpp_cmd = llamacpp_cmd
        self.min_hold = min_hold
        self.enabled = enabled

        self.active: str | None = None
        self._proc: subprocess.Popen | None = None
        self._since = 0.0
        self._lock = asyncio.Lock()
        self.stats = {"switches": 0, "held": 0}

    # ---------- probes ----------

    async def _llamacpp_alive(self) -> bool:
        try:
            async with self.session.get(f"{self.llamacpp_url}/health", timeout=_t(2)) as r:
                return r.status == 200
        except Exception:
            return False

    async def _ollama_loaded(self) -> list[str]:
        try:
            async with self.session.get(f"{self.ollama_url}/api/ps", timeout=_t(5)) as r:
                d = await r.json()
            return [m["name"] for m in d.get("models", [])]
        except Exception:
            return []

    # ---------- eviction ----------

    async def _stop_llamacpp(self):
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except Exception:
                self._proc.terminate()
        for _ in range(50):
            if not await self._llamacpp_alive():
                return
            await asyncio.sleep(0.2)

    async def _stop_ollama_models(self):
        """Unload every resident Ollama model. keep_alive=0 is the documented way."""
        for name in await self._ollama_loaded():
            try:
                async with self.session.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": name, "keep_alive": 0},
                    timeout=_t(30),
                ):
                    pass
            except Exception:
                pass
        for _ in range(50):
            if not await self._ollama_loaded():
                return
            await asyncio.sleep(0.3)

    async def _start_llamacpp(self):
        if not self.llamacpp_cmd:
            raise RuntimeError(
                "cannot start llama.cpp: no --llamacpp-cmd configured"
            )
        self._proc = subprocess.Popen(
            self.llamacpp_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # own process group, so we can kill the whole tree
        )
        for _ in range(300):
            if await self._llamacpp_alive():
                return
            if self._proc.poll() is not None:
                raise RuntimeError("llama.cpp exited during startup")
            await asyncio.sleep(0.2)
        raise RuntimeError("llama.cpp did not become healthy in time")

    # ---------- the arbiter ----------

    async def ensure(self, backend: str) -> dict:
        """Guarantee `backend` owns the GPU. Returns a note about what happened."""
        if not self.enabled:
            return {"arbitrated": False}

        async with self._lock:
            if self.active == backend:
                self.stats["held"] += 1
                return {"arbitrated": False, "active": backend}

            held = time.time() - self._since
            if self.active and held < self.min_hold:
                # Someone else just took the card. Refusing to flip-flop is the whole
                # point of min_hold: two callers alternating every few seconds would
                # spend all their time reloading models and none of it generating.
                return {
                    "arbitrated": False,
                    "active": self.active,
                    "refused": (
                        f"{self.active} has held the GPU for {held:.1f}s "
                        f"(min_hold={self.min_hold}s); not switching to {backend} yet"
                    ),
                }

            t0 = time.time()
            if backend == "llamacpp":
                await self._stop_ollama_models()
                await self._start_llamacpp()
            elif backend == "ollama":
                await self._stop_llamacpp()
                # Ollama loads lazily on the next request; nothing to start.
            else:
                raise ValueError(backend)

            self.active = backend
            self._since = time.time()
            self.stats["switches"] += 1
            return {
                "arbitrated": True,
                "active": backend,
                "switch_seconds": round(time.time() - t0, 2),
            }

    async def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "active": self.active,
            "llamacpp_alive": await self._llamacpp_alive(),
            "ollama_models_loaded": await self._ollama_loaded(),
            "stats": self.stats,
            "min_hold": self.min_hold,
        }


def _t(seconds: float):
    import aiohttp

    return aiohttp.ClientTimeout(total=seconds)
