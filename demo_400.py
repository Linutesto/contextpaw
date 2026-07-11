#!/usr/bin/env python3
"""
The other half of the pitch.

llama.cpp does the honest thing on overflow: it returns a hard 400. That is *correct*
behaviour for a server -- and it is *fatal* for an agent, which dies mid-loop and loses
the run. ContextPaw catches the 400, compacts, retries once, and the agent survives.

  llama.cpp direct     -> 400 exceed_context_size_error, turn is dead
  llama.cpp + ContextPaw -> compacted, retried, 200 OK with a full eviction report
"""
import httpx

SECRET = "Le mot de passe secret est ANANAS-7734."
filler = "Ceci est du texte de remplissage sans importance. " * 3000
PROMPT = (f"{SECRET}\n\n{filler}\n\n"
          "Question: quel est le mot de passe secret mentionne au tout debut?")

BODY = {"prompt": PROMPT, "max_tokens": 50, "temperature": 0, "stream": False}


def run(label, url, headers=None):
    print(f"\n{'='*66}\n{label}\n{'='*66}")
    r = httpx.post(f"{url}/v1/completions", json=BODY, headers=headers or {}, timeout=900)
    print(f"  HTTP {r.status_code}")
    if r.status_code != 200:
        err = r.json().get("error", {})
        print(f"  error type : {err.get('type')}")
        print(f"  message    : {err.get('message')}")
        print("\n  >>> the agent's turn is DEAD here. No answer, no recovery.")
        return False
    d = r.json()
    paw = d.get("contextpaw")
    if paw:
        print(f"  contextpaw : {paw['original_tokens']} -> {paw['final_tokens']} tokens, "
              f"{paw['evicted_tokens']} evicted")
        print(f"               recovered_from_upstream_400="
              f"{paw.get('recovered_from_upstream_400', False)}")
    ans = (d.get("choices") or [{}])[0].get("text", "").strip()
    print(f"  answer     : {ans[:130]!r}")
    ok = "ANANAS" in ans.upper()
    print(f"\n  >>> survived? {'YES' if ok else 'answered, but lost the secret'}")
    return ok


if __name__ == "__main__":
    run("A. llama.cpp direct (ctx 4096/slot)", "http://127.0.0.1:8091")
    run("B. llama.cpp through ContextPaw :11500", "http://127.0.0.1:11500",
        headers={"X-ContextPaw-Backend": "llamacpp"})
