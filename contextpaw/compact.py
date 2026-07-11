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


class TooLongToCompact(Exception):
    """Even after evicting everything evictable, it still does not fit."""


def compact_prompt(text: str, budget: int, count, *, head_frac=0.35, tail_frac=0.45):
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

    if count(out) > budget:
        raise TooLongToCompact(
            f"prompt still {count(out)} tokens after compaction, budget {budget}"
        )

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

    sizes = [count(msg_text(m)) for m in messages]
    total = sum(sizes)
    rep = CompactionReport(original_tokens=total, budget=budget, final_tokens=total)
    if total <= budget:
        return messages, rep

    n = len(messages)
    pinned = {i for i, m in enumerate(messages) if m.get("role") == "system"}
    recent = set(range(max(0, n - keep_recent), n))
    protected = pinned | recent

    evictable = sorted(
        (i for i in range(n) if i not in protected),
        key=lambda i: sizes[i],
        reverse=True,  # biggest first
    )

    keep = set(range(n))
    freed = 0
    need = total - budget
    evicted_desc = []
    evicted_chunks = []

    for i in evictable:
        if freed >= need:
            break
        keep.discard(i)
        freed += sizes[i]
        role = messages[i].get("role", "?")
        evicted_desc.append(f"message #{i} ({role}, {sizes[i]} tokens)")
        evicted_chunks.append((i, role, msg_text(messages[i])))

    out = []
    for i, m in enumerate(messages):
        if i in keep:
            out.append(m)
        elif not out or out[-1].get("_contextpaw_marker") is not True:
            # collapse a run of evicted messages into ONE marker, in place, so the
            # agent sees exactly where the hole is
            out.append(
                {
                    "role": "system",
                    "content": _marker(0, "earlier conversation and tool output").strip(),
                    "_contextpaw_marker": True,
                }
            )

    # fix up the marker text with the real number now that we know it
    marker_text = _marker(freed, "earlier conversation and tool output").strip()
    for m in out:
        if m.get("_contextpaw_marker"):
            m["content"] = marker_text
            m.pop("_contextpaw_marker", None)

    evicted_chunks.sort()
    rep.evicted_text = "\n\n".join(f"[{role}]\n{txt}" for _, role, txt in evicted_chunks)
    rep.marker = marker_text
    rep.marker_what = "earlier conversation and tool output"

    final = sum(count(msg_text(m)) for m in out)
    strategy = f"evict-biggest-first (system + last {keep_recent} preserved)"

    # Still over budget. In a real agent loop the LAST message is very often a huge
    # tool dump -- a file read, an HTTP body -- so it sits inside the protected
    # "recent" window and cannot be dropped. Protecting a message from DELETION must
    # not mean protecting it from SHRINKING, or one big recent tool result deadlocks
    # the whole compaction. So: shrink the biggest non-system messages in place,
    # middle-out, keeping their head and tail.
    if final > budget:
        over = final - budget
        order = sorted(
            (i for i, m in enumerate(out) if m.get("role") != "system"),
            key=lambda i: count(msg_text(out[i])),
            reverse=True,
        )
        for i in order:
            if over <= 0:
                break
            m = out[i]
            size = count(msg_text(m))
            if size < 64:
                continue
            target = max(size - over, 48)
            shrunk, sub = compact_prompt(msg_text(m), target, count)
            if not sub.compacted:
                continue
            out[i] = {**m, "content": shrunk}
            gained = size - count(shrunk)
            over -= gained
            freed += gained
            evicted_desc.append(
                f"message #{i} ({m.get('role','?')}) shrunk in place, "
                f"{sub.evicted_tokens} tokens elided from its middle"
            )
        strategy += " + shrink-in-place"
        final = sum(count(msg_text(m)) for m in out)

    if final > budget:
        raise TooLongToCompact(
            f"messages still {final} tokens after evicting and shrinking everything "
            f"evictable (budget {budget}); the system prompt alone does not fit"
        )

    rep.compacted = True
    rep.strategy = strategy
    rep.final_tokens = final
    rep.evicted_tokens = freed
    rep.evicted = evicted_desc
    return out, rep
