# Reddit post — draft (do not post without reading it back)

**Where:** r/LocalLLaMA is the primary target (this is exactly their pain). Secondary:
r/ollama, r/MachineLearning (Project flair). Post to r/LocalLLaMA first, alone, and let it
breathe for a day — cross-posting the same hour reads as spam.

**Title options** (pick one; the first is the strongest — it is concrete, falsifiable, and
names the thing people can check on their own box in 30 seconds):

1. `Ollama silently truncates your prompt on context overflow and returns 200 OK. I measured it, and the model just makes something up.`
2. `PSA: when your prompt exceeds num_ctx, Ollama throws away the FRONT of it — your system prompt — and tells you nothing`
3. `I built a proxy that stops Ollama from silently eating your system prompt (and stops llama.cpp's 400 from killing your agent)`

**Flair:** Resources / Tutorial (r/LocalLLaMA)

---

## Body

I spent an afternoon trying to prove my 4090 could serve 4 concurrent requests. I came out
of it with a tool, because I found something I hadn't seen written down anywhere.

**Send Ollama a prompt bigger than `num_ctx` and it silently drops the front of it, returns
`200 OK`, and lets the model invent an answer.**

Not an error. Not a warning. No `truncated` field. Nothing in the logs.

I put a secret at the very start of a 160,689-token prompt with `num_ctx=32768`, then asked
for it back:

    HTTP 200 OK
    prompt_eval_count: 16387        <- read 16k of 160k. Said nothing.

    Q: what is the secret password mentioned at the very beginning?
    A: "The secret password is: **filler**."

`filler` was the padding text. It never saw the head, so it made something up — confidently,
with a 200.

The part that should worry you if you run agents: **what Ollama throws away is the FRONT** —
your system prompt, your tool definitions, your task goal. Your agent doesn't crash. It gets
quietly, progressively dumber as tool outputs accumulate, and you have nothing to grep for.

llama.cpp does the opposite. It returns a clean, machine-readable `400
exceed_context_size_error`. Which is *correct* — and it kills the agent turn stone dead.

Two servers, two opposite failures, same broken agent.

---

So I wrote **ContextPaw** — a proxy that sits in front of either one.

    pip install contextpaw

It speaks the Ollama API *and* the OpenAI API and takes port 11434, so nothing in your stack
changes. Two rules:

**1. Never rewrite the head.** Both servers cache the prompt *prefix* — trimming from the
front nukes that cache every turn AND destroys your system prompt. ContextPaw evicts from the
**middle** and keeps both ends. For chat messages it evicts biggest-first (stale tool dumps
are the cheap win, not old dialogue), never touches `system`, and shrinks a huge *recent* tool
output in place rather than deadlocking.

**2. Never evict silently.** Every eviction is reported in the body, the headers, and inline
to the model.

Same prompt through the proxy:

    contextpaw: 160689 -> 26024 tokens, 134727 evicted from the middle
    A: "The secret password is ANANAS-7734."   <- correct

---

## The bit I did not expect

Rule 2 assumes that **telling** the model it lost something stops it inventing. So I tested
that assumption. Three facts buried in the **middle** of an over-long prompt (i.e. in the
evicted region), with a marker in plain English saying the info was removed and it should
fetch it rather than guess:

| | facts recovered |
|---|---|
| marker only | **0 / 3** — invented a server name, a cause, and a person |
| `--summarize` | **3 / 3** |

**Telling an agent it lost information does not stop it hallucinating. You have to give the
content back.** So `--summarize` digests the evicted span with a small local model
(`gemma3:1b` by default) and splices it into the marker.

And a confession, because it's the most instructive part: my first summarizer **trimmed the
evicted span head+tail** to fit its own context window. It scored 0/3 — because the facts were
in the *middle of the evicted span*. I had reproduced, inside the summarizer, the exact bug
the whole project exists to fix. It map-reduces over every chunk now. There's a test.

---

## Reproduce everything

    pip install contextpaw
    git clone https://github.com/Linutesto/contextpaw && cd contextpaw

    python3 demo_needle.py      # Ollama invents an answer; ContextPaw doesn't
    python3 demo_400.py         # llama.cpp's 400 kills the turn; ContextPaw survives it
    python3 demo_summarize.py   # 0/3 facts -> 3/3 with --summarize
    python3 -m pytest tests/ -q # 14 passed

If a number doesn't reproduce on your box, open an issue — I'd rather be corrected than
quoted.

**Code (MIT):** https://github.com/Linutesto/contextpaw
**Write-up with the full numbers:** https://yandesbiens.com/blog/contextpaw-silent-truncation/

---

## Bonus, since it's the same disease

Chasing 4-way concurrency the same afternoon:

- **Ollama accepts `OLLAMA_NUM_PARALLEL=4` and then ignores it** for the `qwen35` arch —
  it forces `Parallel:1` (hybrid linear attention can't be batched in its engine). Logs a WARN
  and moves on; your requests just queue. TTFT hit 14s for the 4th of 4. Still true on 0.31.
  Same GGUF through `llama-server --parallel 4`: real 4-way, **407 tok/s aggregate vs 110**.
- **`ollama ps` said `100% GPU` while the journal said `model weights device=CPU 667.5 MiB`.**
- **`OLLAMA_NUM_CTX` isn't a real variable** (it's `OLLAMA_CONTEXT_LENGTH`). Mine had been sitting
  in my systemd drop-in for months doing absolutely nothing. Nothing warned me.
- **gemma4 returns an empty `response`** unless you pass `think: false` — the tokens go to the
  `thinking` field.

Every one of those is a component choosing to *look* successful over *being* correct.

---

## Notes for posting

- **Do not** open with the tool. Open with the bug. r/LocalLLaMA reflexively downvotes
  "I built a thing" posts and upvotes "here is a thing that is broken, here is the receipt."
  The draft above is already ordered that way — keep it.
- Expect the top comment to be *"just set num_ctx bigger"*. The answer: that doesn't scale in
  an agent loop where the prompt grows every turn, and it doesn't change the fact that the
  failure is **silent**. A bigger window postpones it; it doesn't make it observable.
- Expect *"this is documented behaviour"*. Ask for the link. I couldn't find one, and
  `prompt_eval_count` is the only tell — you have to already suspect it to look.
- Reply to people who reproduce it, especially anyone who *doesn't*. If it behaves differently
  on another Ollama version, that is a real finding and it goes in the README.
