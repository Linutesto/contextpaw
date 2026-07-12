"""
Compaction: make an oversized prompt fit WITHOUT lying about it.

Two rules drive every decision here.

RULE 1 — Never rewrite the head.
    llama.cpp and Ollama both cache the prompt prefix. If you compact by trimming from
    the front, you invalidate that cache on every single turn and your TTFT explodes.
    You also throw away the system prompt, the tool definitions, and the task goal --
    i.e. exactly the tokens the agent cannot function without. This is what Ollama's
    built-in truncation does, and it is why it produces confidently wrong answers.
    We evict from the MIDDLE and keep both ends.

RULE 2 — Never evict silently.
    Every eviction is reported back to the caller: how many tokens went, and from where.
    An agent that KNOWS it lost the output of tool call #7 can go read the file again.
    An agent that was never told will hallucinate over the hole. Silence is the bug.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class CompactionReport:
    compacted: bool = False
    strategy: str = "none"
    original_tokens: int = 0
    final_tokens: int = 0
    evicted_tokens: int = 0
    budget: int = 0
    evicted: list = field(default_factory=list)  # human-readable list of what went
    exact_count: bool = False
    summarized: bool = False
    # raw text we threw away, so a summarizer can be run over it afterwards, plus the
    # exact marker string we inserted, so the digest can be spliced into its place
    evicted_text: str = ""
    marker: str = ""
    marker_what: str = "content"

    def as_dict(self) -> dict:
        return {
            "compacted": self.compacted,
            "strategy": self.strategy,
            "original_tokens": self.original_tokens,
            "final_tokens": self.final_tokens,
            "evicted_tokens": self.evicted_tokens,
            "budget": self.budget,
            "evicted": self.evicted,
            "exact_count": self.exact_count,
            "summarized": self.summarized,
        }


def _marker(n_tokens: int, what: str = "content", digest: str | None = None) -> str:
    head = (
        f"\n\n[contextpaw: {n_tokens} tokens of {what} elided to fit the context window."
    )
    if digest:
        return (
            f"{head}\n"
            f"Here is a digest of what was removed — treat it as a summary, not as the "
            f"original text. If you need exact details, fetch them again rather than "
            f"guessing.\n\nDIGEST:\n{digest}]\n\n"
        )
    return (
        f"{head} This information is GONE from your context — if you need it, fetch it "
        f"again rather than guessing.]\n\n"
    )


def inject_digest(payload, rep: "CompactionReport", digest: str):
    """
    Splice a summary of the evicted span into the marker we already inserted.

    `payload` is either the prompt string or the message list. We replace the exact
    marker text, so this is safe whichever shape we are dealing with.
    """
    if not digest or not rep.marker:
        return payload
    new_marker = _marker(rep.evicted_tokens, rep.marker_what, digest)
    rep.summarized = True

    if isinstance(payload, str):
        return payload.replace(rep.marker, new_marker)

    out = []
    for m in payload:
        c = m.get("content")
        if isinstance(c, str) and rep.marker.strip() and rep.marker.strip() in c:
            out.append({**m, "content": c.replace(rep.marker.strip(), new_marker.strip())})
        else:
            out.append(m)
    return out


def rescue_orphan_tools(messages: list) -> tuple[list, int]:
    """
    A `role: "tool"` message that is NOT preceded by an assistant message carrying
    `tool_calls` is an ORPHAN, and the chat template silently discards it. Ollama returns
    200. The model then answers without ever having seen the tool output -- it will even
    say "please provide the log", and a harness will faithfully record that as the answer.

    Measured on gemma-4-abliterated-12b, same log, same question, only the role changed:

        orphan role=tool ............ 0/3 facts   "Please provide the log you are
                                                   referring to!"
        assistant.tool_calls + tool .. 3/3 facts
        remapped to role=user ....... 3/3 facts

    This matters to US more than to anyone: compaction can CREATE orphans. Evict the
    assistant message that carried the tool_calls, keep the tool result that followed it,
    and we have just fed the runtime something it will throw away without a word. The
    tool output survives our budget and dies in the template.

    So: any orphan we find -- whether it arrived that way or we made it -- is remapped to
    `user` with an explicit label. Ugly, and correct. The alternative is silent deletion.
    """
    out, rescued = [], 0
    for i, m in enumerate(messages):
        if m.get("role") != "tool":
            out.append(m)
            continue
        prev = messages[i - 1] if i else None
        parented = bool(prev and prev.get("role") == "assistant" and prev.get("tool_calls"))
        if parented:
            out.append(m)
            continue
        name = m.get("tool_name") or m.get("name") or "tool"
        content = m.get("content")
        content = content if isinstance(content, str) else str(content)
        out.append({
            "role": "user",
            "content": f"[TOOL OUTPUT — {name}]\n{content}",
        })
        rescued += 1
    return out, rescued


def collapse_repeats(text: str, min_run: int = 3) -> str:
    """
    Fold repeated lines before evicting anything. This is what a human does when they read
    a log: they do not read 300 copies of "all tests passed", they notice it happened 300
    times and move on.

    It is also the fix for a failure that nearly sank the whole design. Eviction used to
    work on whole messages, biggest-first, on the theory that a big stale tool output is
    the cheapest thing to lose. Measured against a realistic agent loop, that dropped the
    ONE tool output holding the answer -- a deploy log -- precisely BECAUSE it was big,
    while keeping 300 identical lines of "all 412 tests passed".

    And you cannot fix it with an information-density score either: a real log is mostly
    repetitive noise with a few lines that matter, so it scores as low-information as the
    junk. The needle is inside the haystack, not next to it. So do not choose between
    them -- burn the hay and keep the needle.

    Measured on that log: 25,936 chars -> 342, with every unique line (the ERROR, the
    server name, the person who fixed it) intact.
    """
    lines = text.split("\n")
    counts = Counter(lines)
    out, seen = [], set()
    for ln in lines:
        n = counts[ln]
        if n >= min_run and ln.strip():
            if ln in seen:
                continue
            seen.add(ln)
            out.append(f"{ln}    [x{n}, repetitions collapsed by contextpaw]")
        else:
            out.append(ln)
    return "\n".join(out)


class TooLongToCompact(Exception):
    """Even after evicting everything evictable, it still does not fit."""


def compact_prompt(text: str, budget: int, count, *, head_frac=0.35, tail_frac=0.45,
                   best_effort: bool = False):
    """
    Compact a raw prompt string (Ollama /api/generate, llama.cpp /v1/completions).

    We keep a head slice and a tail slice and drop the middle. The head carries the
    instructions and the framing; the tail carries the most recent context and the
    actual question. The middle is where bulk filler and stale tool output live.
    """
    total = count(text)
    rep = CompactionReport(original_tokens=total, budget=budget, final_tokens=total)
    if total <= budget:
        return text, rep

    # Work in characters, proportionally: we do not have token offsets for the
    # calibrated counter, and slicing on character boundaries is good enough because
    # the budget already carries a safety margin.
    ratio = len(text) / max(total, 1)
    head_chars = int(budget * head_frac * ratio)
    tail_chars = int(budget * tail_frac * ratio)

    head = text[:head_chars]
    tail = text[-tail_chars:] if tail_chars else ""
    dropped = total - count(head) - count(tail)

    evicted_text = text[head_chars: len(text) - tail_chars] if tail_chars else text[head_chars:]
    marker = _marker(max(dropped, 0))

    out = head + marker + tail
    # The marker itself costs tokens; if we overshot, trim the tail until it fits.
    while count(out) > budget and len(tail) > 200:
        tail = tail[len(tail) // 5 :]
        out = head + marker + tail

    if count(out) > budget and not best_effort:
        raise TooLongToCompact(
            f"prompt still {count(out)} tokens after compaction, budget {budget}"
        )
    if count(out) >= total:
        # best_effort and we made it no smaller: hand the original back untouched rather
        # than paying for a marker that bought nothing.
        return text, rep

    rep.compacted = True
    rep.strategy = "middle-out (head+tail preserved)"
    rep.final_tokens = count(out)
    rep.evicted_tokens = max(dropped, 0)
    rep.evicted = [f"{max(dropped,0)} tokens from the middle of the prompt"]
    rep.evicted_text = evicted_text
    rep.marker = marker
    rep.marker_what = "content"
    return out, rep


def compact_messages(messages: list, budget: int, count, *, keep_recent: int = 4):
    """
    Compact a chat message list (Ollama /api/chat, OpenAI /v1/chat/completions).

    This is the one that matters for agents. Priority order:

      PINNED    system messages           -> never evicted (tool defs, instructions)
      RECENT    last `keep_recent` msgs   -> never evicted (the current task)
      EVICTABLE everything else           -> evicted biggest-first

    Biggest-first is deliberate: in an agent loop the giant messages are almost always
    stale tool output (a file read, an HTTP body). Dropping one 12k-token file dump
    recovers more room than dropping twenty turns of dialogue, and loses far less of
    the reasoning thread.
    """
    def msg_text(m):
        c = m.get("content")
        return c if isinstance(c, str) else str(c)

    # STEP 0: rescue orphan tool messages BEFORE anything else. If we evict an assistant
    # that carried tool_calls, its tool result becomes an orphan and the template deletes
    # it silently -- so we must fix orphans that already exist AND re-check after we are
    # done making our own.
    messages, _ = rescue_orphan_tools(messages)

    original_total = sum(count(msg_text(m)) for m in messages)

    # STEP 1: burn the hay. Collapsing repeated lines is free, lossless in any sense the
    # agent cares about, and it routinely removes 98% of a tool dump. Do it before
    # considering whether to throw anything away -- most of the time, it means we do not
    # have to.
    messages = [
        {**m, "content": collapse_repeats(msg_text(m))} if len(msg_text(m)) > 400 else m
        for m in messages
    ]

    sizes = [count(msg_text(m)) for m in messages]
    total = sum(sizes)
    rep = CompactionReport(original_tokens=original_total, budget=budget, final_tokens=total)
    if total <= budget:
        rep.compacted = original_total != total
        rep.strategy = "collapsed repeated lines (no eviction needed)"
        rep.evicted_tokens = 0
        if rep.compacted:
            rep.evicted = [f"{original_total - total} tokens of repeated lines collapsed"]
        return messages, rep

    n = len(messages)
    pinned = {i for i, m in enumerate(messages) if m.get("role") == "system"}

    # The GOAL. The first user message is what the agent was ASKED to do. It is not
    # `system`, and in a long loop it is nowhere near `recent` -- so nothing protected it.
    # Measured: it was evicted at 29 tokens, and the agent then confidently answered a
    # question it no longer knew it had been asked.
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            pinned.add(i)
            break

    recent = set(range(max(0, n - keep_recent), n))
    protected = pinned | recent

    messages = list(messages)
    freed = 0
    evicted_desc: list = []
    evicted_chunks: list = []

    # STEP 2: SHRINK before you DELETE.
    #
    # This ordering is the whole ballgame, and getting it backwards produced the worst bug
    # in the project. Deleting a 123-token tool result that happens to hold the answer
    # frees 123 tokens and loses everything. Shrinking a 10,000-token pile of filler
    # middle-out frees 8,000 and loses almost nothing, because its head and tail survive.
    #
    # The old order -- evict whole messages biggest-first, shrink only as a last resort --
    # meant a big USELESS message protected by the `recent` window pushed the algorithm
    # into deleting a small USEFUL one to make up the difference. Shrink first, and that
    # can never happen.
    def total_of(ms):
        return sum(count(msg_text(m)) for m in ms)

    # One pass, biggest first, each message shrunk at most once. Deliberately NOT a
    # while-loop that re-picks the biggest every iteration: if compact_prompt cannot hit
    # the target for one message, we must SKIP that message and keep going -- an early
    # `break` there abandoned the entire shrink phase on the first stubborn message, and
    # the whole compaction then failed with a 413.
    # Progressively harsher floors. DELETING a message is always worse than shrinking it to
    # the bone: a tool result cut down to its head still tells the agent "there was output
    # here, and it began like this". Deleted, it leaves a hole the model will cheerfully
    # hallucinate over. So squeeze before you cut -- and only cut what squeezing cannot
    # save.
    for floor in (max(64, budget // 4), max(48, budget // 8), 32):
        if total_of(messages) <= budget:
            break
        order = sorted(
            (i for i in range(n) if messages[i].get("role") != "system"),
            key=lambda i: count(msg_text(messages[i])),
            reverse=True,
        )
        for i in order:
            cur = total_of(messages)
            if cur <= budget:
                break
            # The floor must scale with the BUDGET, not be a constant. A fixed 256-token floor
            # means two shrunk messages already cost 512 -- which does not fit a 300-token
            # budget at all, and the whole compaction then fails. Small budgets need small
            # floors; the only hard limit is that the elision marker must still fit.
            size = count(msg_text(messages[i]))
            if size < floor:
                continue  # nothing to win here, and the marker alone would cost more
            over = cur - budget
            target = max(size - over, floor)
            # best_effort: when shrinking a message IN PLACE, any reduction is progress. The
            # elision marker has a fixed cost, so an aggressive target can be unreachable --
            # and refusing to shrink at all because we could not hit it exactly left a
            # 4,005-token message completely untouched and failed the whole compaction.
            shrunk, sub = compact_prompt(msg_text(messages[i]), target, count, best_effort=True)
            if not sub.compacted:
                continue
            gained = size - count(shrunk)
            if gained <= 0:
                continue
            messages[i] = {**messages[i], "content": shrunk}
            freed += gained
            evicted_chunks.append((i, messages[i].get("role", "?"), sub.evicted_text))
            evicted_desc.append(
                f"message #{i} ({messages[i].get('role','?')}) shrunk in place, "
                f"{sub.evicted_tokens} tokens elided from its middle"
            )

    strategy = "collapse repeated lines -> shrink-in-place (system + goal + recent preserved)"

    # STEP 3: last resort — KEEP THE HEAD, cut the rest. Do not delete.
    #
    # The elision marker has a fixed cost (~30 tokens), so middle-out shrinking bottoms
    # out: below a certain size the marker IS the message. The old code gave up there and
    # deleted the whole thing -- which threw away a tool result holding the answer to keep
    # a budget it was already nearly meeting.
    #
    # A tool output cut down to "NEEDLE-ANANAS pad pad ... [+387 tokens cut]" still tells
    # the agent what it found and that there was more. Deleted, it leaves a hole, and the
    # model will hallucinate over a hole every single time. Truncation loses detail;
    # deletion loses the fact that anything ever happened.
    keep = set(range(n))
    if total_of(messages) > budget:
        evictable = sorted(
            (i for i in range(n) if i not in protected),
            key=lambda i: count(msg_text(messages[i])),
            reverse=True,
        )
        for i in evictable:
            cur = total_of([messages[j] for j in sorted(keep)])
            if cur <= budget:
                break
            size = count(msg_text(messages[i]))
            over = cur - budget
            # The "[contextpaw: +N tokens truncated here]" suffix costs tokens too. Not
            # budgeting for it left us 4 tokens over and failed the whole compaction --
            # the same class of off-by-a-marker mistake as the shrink floor.
            SUFFIX = 12
            head_tokens = max(size - over - SUFFIX, 16)
            if head_tokens >= size:
                continue

            text = msg_text(messages[i])
            words = text.split()
            if len(words) <= 20:
                # genuinely tiny (assistant chatter) — nothing to keep, drop it
                keep.discard(i)
                freed += size
                evicted_desc.append(
                    f"message #{i} ({messages[i].get('role','?')}, {size} tokens) dropped"
                )
                evicted_chunks.append((i, messages[i].get("role", "?"), text))
                continue

            # keep the head, say how much went
            ratio = len(text) / max(size, 1)
            head = text[: int(head_tokens * ratio)]
            cut = size - count(head)
            messages[i] = {
                **messages[i],
                "content": head + f"\n\n[contextpaw: +{cut} tokens truncated here]",
            }
            freed += cut
            evicted_chunks.append((i, messages[i].get("role", "?"), text[len(head):]))
            evicted_desc.append(
                f"message #{i} ({messages[i].get('role','?')}) truncated to its head, "
                f"{cut} tokens cut"
            )
        strategy += " -> head-truncate"

    out = []
    marker_text = _marker(freed, "earlier conversation and tool output").strip()
    inserted = False
    for i in range(n):
        if i in keep:
            out.append(messages[i])
        elif not inserted:
            out.append({"role": "system", "content": marker_text})
            inserted = True

    # We may have just orphaned a tool result by evicting the assistant that called it.
    out, orphaned = rescue_orphan_tools(out)
    if orphaned:
        evicted_desc.append(
            f"{orphaned} tool result(s) re-parented to `user` — eviction orphaned them, "
            f"and the chat template would have deleted them without a word"
        )

    final = sum(count(msg_text(m)) for m in out)
    if final > budget:
        raise TooLongToCompact(
            f"messages still {final} tokens after collapsing, shrinking and evicting "
            f"everything evictable (budget {budget}); the system prompt alone does not fit"
        )

    evicted_chunks.sort()
    rep.evicted_text = "\n\n".join(f"[{role}]\n{txt}" for _, role, txt in evicted_chunks if txt)
    rep.marker = marker_text
    rep.marker_what = "earlier conversation and tool output"

    rep.compacted = True
    rep.strategy = strategy
    rep.final_tokens = final
    rep.evicted_tokens = freed
    rep.evicted = evicted_desc
    return out, rep
