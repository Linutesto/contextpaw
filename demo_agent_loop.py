#!/usr/bin/env python3
"""
The case ContextPaw was written for, and the one it had never actually seen.

Sixty requests through the proxy in production and `compacted: 0`. Every one of them fit.
The message-level eviction — protect `system`, drop the biggest stale tool output first,
shrink a huge *recent* one in place — had only ever run in synthetic unit tests.

An agent loop is where context actually explodes: every read_file, every HTTP body, every
grep result piles up and never leaves. So we build one, exactly the shape AgentOS's
OllamaProvider sends (POST /api/chat, `messages` array), and overflow it on purpose.

The test is not "does it fit". Anything can make it fit by deleting things. The test is:

  1. does the SYSTEM PROMPT survive?          (the tool definitions the agent needs)
  2. does the TASK GOAL survive?              (what it was asked to do)
  3. does the RECENT question survive?
  4. is the STALE tool dump the thing that got dropped?  (not the reasoning)
  5. can the agent still answer using a fact from an EVICTED tool output,
     recovered through the digest?

That last one is the whole product. Points 1-4 are just not being reckless.
"""
import json
import sys

import httpx

PROXY = "http://127.0.0.1:11434"
MODEL = "huihui_ai/gemma-4-abliterated:12b"
NUM_CTX = 8192          # small on purpose — force the overflow an agent hits eventually

SYSTEM = (
    "You are a coding agent. You have these tools:\n"
    "  read_file(path)   -> returns file contents\n"
    "  grep(pattern)     -> returns matching lines\n"
    "  run_tests()       -> returns the test report\n"
    "RULE-7: you must never modify files under /vendor. This rule is absolute.\n"
)

GOAL = "GOAL: find why the deploy failed, and tell me who fixed it and what the fix was."

# The needle sits in an OLD tool output — squarely in the region that gets evicted.
DEPLOY_LOG = (
    "=== deploy.log ===\n"
    "[08:14] starting deploy to orion-7\n"
    "[08:15] ERROR: TLS certificate expired on orion-7\n"
    "[08:16] deploy aborted\n"
    "[08:41] certificate renewed by Marie-Claude\n"
    "[08:42] deploy succeeded\n"
) + ("[trace] " + "x" * 90 + "\n") * 260          # a realistic, bulky log dump

BULK = ("def helper_%d():\n    return %d\n\n" % (0, 0)) * 1
FILLER_FILE = "".join(f"def helper_{i}():\n    return {i}\n\n" for i in range(900))

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": GOAL},

    {"role": "assistant", "content": "I'll read the deploy log."},
    {"role": "tool", "content": DEPLOY_LOG},                    # <-- THE NEEDLE, old + big

    {"role": "assistant", "content": "Now let me look at the source."},
    {"role": "tool", "content": "=== app/main.py ===\n" + FILLER_FILE},   # big, useless

    {"role": "assistant", "content": "And the vendored dependencies."},
    {"role": "tool", "content": "=== vendor/lib.py ===\n" + FILLER_FILE}, # big, useless

    {"role": "assistant", "content": "Let me check the test report."},
    {"role": "tool", "content": "=== tests ===\n" + ("all 412 tests passed\n" * 300)},

    {"role": "user", "content":
        "So: which server failed, why did it fail, and who fixed it? "
        "Also restate RULE-7 exactly. If you don't know something, say so — do not guess."},
]


def main():
    body = {
        "model": MODEL, "messages": messages, "stream": False,
        "options": {"num_ctx": NUM_CTX, "num_predict": 220, "temperature": 0},
    }
    print(f"[*] agent transcript: {len(messages)} messages, "
          f"{sum(len(m['content']) for m in messages):,} chars")
    print(f"[*] num_ctx={NUM_CTX} — this will not fit. It is not supposed to.\n")

    r = httpx.post(f"{PROXY}/api/chat", json=body, timeout=900)
    print(f"HTTP {r.status_code}")
    d = r.json()

    paw = d.get("contextpaw")
    if not paw:
        print("\n!!! ContextPaw did NOT compact. The transcript fit, or the proxy is off.")
        return 1

    print(f"\n--- compaction ---")
    print(f"  {paw['original_tokens']:,} -> {paw['final_tokens']:,} tokens "
          f"(budget {paw['budget']:,})")
    print(f"  strategy   : {paw['strategy']}")
    print(f"  summarized : {paw['summarized']}")
    print(f"  evicted    :")
    for e in paw["evicted"]:
        print(f"      - {e}")

    ans = (d.get("message") or {}).get("content", "").strip()
    print(f"\n--- the agent's answer ---\n{ans[:520]}\n")

    a = ans.lower()
    checks = [
        ("system prompt survived (RULE-7 restated)", "vendor" in a or "rule-7" in a),
        ("failure cause recovered  (TLS)",           "tls" in a or "certificat" in a),
        ("server recovered         (orion-7)",       "orion" in a),
        ("person recovered         (Marie-Claude)",  "marie" in a),
    ]
    print("--- did anything important survive? ---")
    for label, ok in checks:
        print(f"  [{'OK ' if ok else 'LOST'}]  {label}")

    passed = sum(1 for _, ok in checks if ok)
    print(f"\n>>> {passed}/4 survived the eviction")
    return 0 if passed == 4 else 1


if __name__ == "__main__":
    sys.exit(main())
