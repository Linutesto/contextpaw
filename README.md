# 🐾 ContextPaw

**Your LLM server lies to you when the context window overflows. ContextPaw makes it stop.**

Two servers, two opposite lies, same broken agent:

| | What it does on overflow | Why that's bad |
|---|---|---|
| **Ollama** | Silently drops the front of your prompt, returns **`200 OK`** | Your agent loses its system prompt and tool definitions, then **confidently makes something up**. Nothing is logged. You will never know. |
| **llama.cpp** | Returns a hard **`400 exceed_context_size_error`** | Honest — but it **kills the agent turn**. The run is dead and the work is lost. |

ContextPaw sits in front of either one and turns overflow into something an agent can
actually survive: **fit the window, keep what matters, and say out loud what was lost.**

```
your app ──► contextpaw :11434 ──► ollama    :11435
                              └──► llama.cpp :8091
```

Drop-in: it speaks Ollama's API *and* the OpenAI API, so it takes port `11434` and
**nothing in your stack needs to change**.

---

## The proof

A secret is placed at the **start** of a prompt that overflows the window. Then we ask for
it back. Same model (`gemma-4-abliterated:12b`), same prompt, same `num_ctx`.

**Ollama, direct:**
```
HTTP 200
prompt_eval_count: 16387          <- it read 16k of a 160k-token prompt. It did not say so.
answer: "Le mot de passe secret est : **remplissage**."
                                  ^^^^^^^^^^^^ it invented this
```

**Ollama, through ContextPaw:**
```
HTTP 200
contextpaw: compacted=True  160689 -> 26024 tokens  (budget 32452)
            strategy: middle-out (head+tail preserved)
            evicted : 134727 tokens
answer: "Le mot de passe secret est **ANANAS-7734**."
                                     ^^^^^^^^^^^ correct
```

And on llama.cpp, where the same prompt is simply fatal:

```
direct        -> HTTP 400 exceed_context_size_error   (agent turn is dead)
via contextpaw-> HTTP 200, 53589 -> 3094 tokens, 50556 evicted, answer correct
```

Reproduce both: `python3 demo_needle.py` and `python3 demo_400.py`.

---

## Install & run

```bash
pip install contextpaw

# move the real Ollama off 11434 so ContextPaw can take it
OLLAMA_HOST=127.0.0.1:11435 ollama serve

contextpaw --port 11434 --ollama http://127.0.0.1:11435
```

Nothing else changes. Your clients keep pointing at `:11434`.

```bash
curl localhost:11434/contextpaw/health   # policy, stats, calibration
```

---

## Two rules it never breaks

**1. Never rewrite the head.**
Both servers cache the prompt *prefix*. Compacting from the front invalidates that cache on
every turn and your TTFT explodes — and it throws away the system prompt, the tool schemas,
and the task goal, which is exactly what the agent cannot work without. **ContextPaw evicts
from the middle and keeps both ends.** (Trimming the head is precisely what Ollama's built-in
truncation does. It is why it produces confident nonsense.)

**2. Never evict silently.**
Every eviction is reported back — in the response body (`contextpaw` field), in headers
(`X-ContextPaw-Compacted`, `X-ContextPaw-Evicted-Tokens`), and **inline to the model itself**:

> `[contextpaw: 12431 tokens of earlier conversation and tool output elided to fit the context window. This information is GONE from your context — if you need it, fetch it again rather than guessing.]`

An agent that *knows* it lost the output of tool call #7 can go read the file again. An agent
that was never told will hallucinate over the hole. **The silence is the bug.**

---

## What gets evicted, in what order

For chat messages (`/api/chat`, `/v1/chat/completions`):

| Priority | | Treatment |
|---|---|---|
| **PINNED** | `system` messages | never touched — tool defs, instructions |
| **RECENT** | last `--keep-recent` (default 4) | never *deleted* |
| **EVICTABLE** | everything else | dropped **biggest-first** |

Biggest-first is deliberate: in an agent loop the giant messages are almost always *stale
tool output* — a file read, an HTTP body. Dropping one 12k-token file dump recovers more room
than dropping twenty turns of dialogue, and loses far less of the reasoning thread.

And the case everyone gets wrong: **when the newest message is itself a huge tool dump**, it
sits in the protected window and would deadlock compaction. Being protected from *deletion*
must not mean being protected from *shrinking* — so ContextPaw shrinks it in place, middle-out,
keeping its head and tail. (This exact bug was found by the test suite, not by luck. See
`test_huge_recent_tool_output_is_shrunk_not_deadlocked`.)

---

## Policies

```bash
contextpaw --policy compact   # (default) shrink it, report it
contextpaw --policy strict    # refuse loudly with 413 rather than compact
contextpaw --policy off       # pure passthrough
```

`strict` is the one to reach for while debugging: it gives you the clean, machine-readable
error that Ollama should have given you in the first place.

## Token counting

By default ContextPaw ships **no tokenizer** and needs none. It estimates, then **calibrates
itself** against the `prompt_eval_count` every server already reports — and it only ever
ratchets the ratio *upward*. Over-counting is safe (you compact a little early); under-counting
is what gets your prompt silently truncated. It refuses to learn an unsafe ratio.

Want exact counts? Opt in:

```bash
pip install contextpaw[exact]
contextpaw --tokenizer google/gemma-3-1b-it
```

---

## Limitations (v0.1)

- Evicted content is **dropped, not summarized**. Summarizing the evicted span into a compact
  note is the obvious next step (a small local model is more than enough for that job).
- On **streaming** responses the eviction report is delivered in headers only — the body is
  proxied through untouched.
- Raw-prompt compaction (`/api/generate`) is head+tail; only *chat messages* get true
  semantic, per-message eviction. Structure your calls as messages if you can.
- Character-proportional slicing, not token-offset slicing, when using the calibrated counter.
  The safety margin absorbs it; use `--tokenizer` if you want it tight.

## Tests

```bash
python3 -m pytest tests/ -q     # 12 passed
```

MIT.
