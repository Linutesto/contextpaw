#!/usr/bin/env python3
"""
The demo that justifies the whole project.

A secret is placed at the START of a prompt that overflows the context window.
Then we ask for it back.

  Ollama, direct     -> 200 OK, head silently discarded, model invents an answer.
  Ollama + ContextPaw -> head preserved, middle evicted, eviction reported, answer correct.

Same model. Same prompt. Same context size. The only difference is who is allowed to lie.
"""
import json
import sys

import httpx

MODEL = "huihui_ai/gemma-4-abliterated:12b"
SECRET = "Le mot de passe secret est ANANAS-7734."
NUM_CTX = 32768

filler = "Ceci est du texte de remplissage sans importance. " * 9000
PROMPT = (
    f"{SECRET}\n\n{filler}\n\n"
    "Question: quel est le mot de passe secret mentionne au tout debut?"
)

BODY = {
    "model": MODEL,
    "prompt": PROMPT,
    "stream": False,
    "think": False,
    "keep_alive": "10m",
    "options": {"num_ctx": NUM_CTX, "num_predict": 60, "temperature": 0},
}


def run(label, url):
    print(f"\n{'='*66}\n{label}\n{'='*66}")
    try:
        r = httpx.post(f"{url}/api/generate", json=BODY, timeout=900)
    except Exception as e:
        print("  transport error:", e)
        return None
    print(f"  HTTP {r.status_code}")
    if r.status_code != 200:
        print("  body:", r.text[:300])
        return None
    d = r.json()
    paw = d.get("contextpaw")
    if paw:
        print(f"  contextpaw : compacted={paw['compacted']} "
              f"{paw['original_tokens']} -> {paw['final_tokens']} tokens "
              f"(budget {paw['budget']})")
        print(f"               strategy: {paw['strategy']}")
        print(f"               evicted : {paw['evicted_tokens']} tokens")
    else:
        print("  contextpaw : (absent — raw backend, no accounting)")
    print(f"  prompt_eval_count (tokens the server actually read): {d.get('prompt_eval_count')}")
    ans = (d.get("response") or "").strip()
    print(f"  answer     : {ans[:150]!r}")
    ok = "ANANAS" in ans.upper()
    print(f"\n  >>> secret recovered? {'YES' if ok else 'NO  <-- it made something up'}")
    return ok


if __name__ == "__main__":
    direct = run("A. Ollama, direct (no ContextPaw)", "http://127.0.0.1:11434")
    viapaw = run("B. Ollama, through ContextPaw :11500", "http://127.0.0.1:11500")

    print(f"\n{'='*66}\nVERDICT\n{'='*66}")
    print(f"  direct        : {'kept the secret' if direct else 'LOST the secret and hallucinated'}")
    print(f"  via contextpaw: {'kept the secret' if viapaw else 'LOST the secret'}")
    sys.exit(0 if viapaw else 1)
