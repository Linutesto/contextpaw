#!/usr/bin/env python3
"""
Does summarizing the evicted span actually save information, or does it just feel good?

The needle demo put the secret at the HEAD, which compaction protects by design. That
proves we don't destroy the head. It does NOT prove the digest is worth anything.

So here we bury three facts in the MIDDLE -- squarely inside the region that gets
evicted. Without a summarizer they are gone, full stop. With one, they should come back
in the digest.

  --summarize off -> facts are gone. Model cannot answer. (Correctly! It was TOLD they
                     were gone, so it should say so rather than invent -- that alone is
                     already better than Ollama's silent hallucination.)
  --summarize on  -> facts survive in the digest.
"""
import httpx

MODEL = "huihui_ai/gemma-4-abliterated:12b"
NUM_CTX = 8192

FACTS = (
    "FAIT 1: le serveur de production s'appelle orion-7.\n"
    "FAIT 2: le deploiement a echoue a cause d'un certificat TLS expire.\n"
    "FAIT 3: le correctif a ete applique par Marie-Claude le 3 mars.\n"
)

filler_a = "Bavardage sans importance au debut. " * 900
filler_b = "Bavardage sans importance a la fin. " * 900

PROMPT = (
    "Tu es un assistant technique. Reponds uniquement a partir du contexte fourni.\n\n"
    + filler_a
    + "\n\n" + FACTS + "\n\n"          # <-- buried in the middle, will be evicted
    + filler_b
    + "\n\nQuestion: comment s'appelle le serveur de production, pourquoi le deploiement "
      "a-t-il echoue, et qui a applique le correctif? Si tu ne sais pas, dis-le."
)

BODY = {
    "model": MODEL, "prompt": PROMPT, "stream": False, "think": False,
    "keep_alive": "10m",
    "options": {"num_ctx": NUM_CTX, "num_predict": 150, "temperature": 0},
}


def run(label, port):
    print(f"\n{'='*68}\n{label}\n{'='*68}")
    r = httpx.post(f"http://127.0.0.1:{port}/api/generate", json=BODY, timeout=900)
    d = r.json()
    paw = d.get("contextpaw", {})
    print(f"  compacted  : {paw.get('original_tokens')} -> {paw.get('final_tokens')} tokens, "
          f"{paw.get('evicted_tokens')} evicted")
    print(f"  summarized : {paw.get('summarized')}")
    ans = (d.get("response") or "").strip()
    print(f"  answer     : {ans[:260]!r}")
    hits = [f for f in ("orion-7", "TLS", "Marie-Claude") if f.lower() in ans.lower()]
    print(f"\n  >>> facts recovered from the EVICTED middle: {hits or 'NONE'}  ({len(hits)}/3)")
    return len(hits)


if __name__ == "__main__":
    a = run("A. ContextPaw, no summarizer (:11500)", 11500)
    b = run("B. ContextPaw + gemma3:1b summarizer (:11501)", 11501)
    print(f"\n{'='*68}\nVERDICT\n{'='*68}")
    print(f"  drop-only  : {a}/3 facts survived the eviction")
    print(f"  summarized : {b}/3 facts survived the eviction")
