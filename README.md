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
git clone https://github.com/Linutesto/contextpaw
cd contextpaw && pip install -e .          # not on PyPI yet

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

## Summarize what you evict (`--summarize`)

A marker saying *"12431 tokens were elided"* tells the agent a hole exists. A **digest**
tells it what was in the hole.

```bash
contextpaw --summarize --summarizer-model gemma3:1b
```

Three facts were buried in the **middle** of an over-long prompt — squarely inside the
region compaction evicts. Then we asked for them back:

| | facts recovered | the model said |
|---|---|---|
| evict only | **0 / 3** | *"Production_Server … erreur de syntaxe … Marc"* |
| `--summarize` | **3 / 3** | *"Orion-7 … certificats TLS expirés … Marie-Claude"* |

Read the first row again. **The model hallucinated even though the marker explicitly told
it the information was gone.** Telling an agent "you lost something" does not stop it
inventing — you have to give the content back. That is the whole case for the summarizer,
and we only know it because we measured it.

Implementation notes that matter:

- **Map-reduce over the whole span, never head+tail.** The first version trimmed the span
  to fit the summarizer's own window — and scored **0/3**, because the facts were in the
  middle of the evicted span, so it dropped them again. It reproduced, inside the
  summarizer, the exact bug this project exists to fix. Now every chunk is summarized and
  merged. (`test_summarizer_chunks_cover_the_whole_span`)
- **Cached by content hash.** In an agent loop the same old turns are evicted every single
  turn; without a cache you would pay for the identical digest forever.
- **Best-effort, never fatal.** If the summarizer is down, times out, or returns nothing,
  we fall back to the plain marker. A rescue tool that fails the request it was rescuing
  is worse than no tool at all.

---

## One runtime owns the GPU (`--arbitrate`)

A 24 GB card cannot hold Ollama's model *and* a llama.cpp server. Measured on an RTX 4090:
gemma-4-12b in Ollama at 4×32k = **15.1 GB**; qwythos-9b in llama.cpp at 4×32k = **11.0 GB**.
26.1 GB > 24 GB. They do not coexist.

The dangerous part: **nothing tells you.** Ollama will load a model on top of a running
llama.cpp server and — with `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` — spill to system RAM
*silently*, 20× slower, while `ollama ps` still cheerfully reports `100% GPU`. Same disease
as silent truncation: **the system would rather lie than say no.**

ContextPaw sees every request, so it knows which runtime the caller wants. It evicts the
other one first.

```bash
contextpaw --arbitrate \
  --llamacpp-cmd '/path/to/llama-server -m model.gguf --parallel 4 -c 32768 -ngl 99 --port 8091'
```

```
$ curl localhost:11434/contextpaw/runtime
{"active": "llamacpp", "llamacpp_alive": true, "ollama_models_loaded": [], "stats": {"switches": 2}}
```

Measured switch cost: **llama.cpp → Ollama 5.1s**, **Ollama → llama.cpp 2.3s**. Cheap enough
to do on demand. `--min-hold` (default 20s) stops two callers flip-flopping the card and
spending all their time reloading models instead of generating.

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

- On **streaming** responses the eviction report is delivered in headers only — the body is
  proxied through untouched.
- The summarizer adds latency on the turn where it runs (5 calls to a 1B model, ~10s for a
  17k-token span) — then it is cached. Enable it for agent loops, not for chat.
- Raw-prompt compaction (`/api/generate`) is head+tail; only *chat messages* get true
  semantic, per-message eviction. Structure your calls as messages if you can.
- Character-proportional slicing, not token-offset slicing, when using the calibrated counter.
  The safety margin absorbs it; use `--tokenizer` if you want it tight.

## Tests

```bash
python3 -m pytest tests/ -q     # 14 passed
```

MIT.
