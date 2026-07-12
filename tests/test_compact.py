import pytest

from contextpaw.compact import (
    compact_messages,
    compact_prompt,
    TooLongToCompact,
)
from contextpaw.tokens import CalibratedCounter


def wc(text: str) -> int:
    """Deterministic fake tokenizer: 1 token per whitespace-separated word."""
    return len(text.split())


# --------------------------- raw prompts ---------------------------

def test_prompt_under_budget_is_untouched():
    p = "hello world " * 10
    out, rep = compact_prompt(p, budget=1000, count=wc)
    assert out == p
    assert rep.compacted is False
    assert rep.evicted_tokens == 0


def test_prompt_over_budget_keeps_the_head():
    """The whole point: the head survives. Ollama's truncation eats it."""
    secret = "PASSWORD-ANANAS-7734"
    p = secret + " " + ("filler " * 5000) + " what was the password?"
    out, rep = compact_prompt(p, budget=200, count=wc)

    assert rep.compacted is True
    assert secret in out, "head was evicted — this is the exact bug we exist to prevent"
    assert "what was the password?" in out, "tail was evicted"
    assert wc(out) <= 200
    assert rep.evicted_tokens > 0


def test_prompt_eviction_is_announced():
    p = "start " + ("filler " * 5000) + " end"
    out, rep = compact_prompt(p, budget=200, count=wc)
    assert "contextpaw" in out, "eviction must be visible to the model, never silent"
    assert "elided" in out


# --------------------------- chat messages ---------------------------

def _msgs():
    return [
        {"role": "system", "content": "SYSTEM you have tools " + ("t " * 50)},
        {"role": "user", "content": "GOAL find the bug"},
        {"role": "assistant", "content": "calling read_file"},
        {"role": "tool", "content": "HUGE " + ("x " * 4000)},   # stale file dump
        {"role": "assistant", "content": "calling read_file again"},
        {"role": "tool", "content": "HUGE2 " + ("y " * 4000)},  # stale file dump
        {"role": "user", "content": "RECENT so what is it?"},
    ]


def test_system_is_never_evicted():
    out, rep = compact_messages(_msgs(), budget=300, count=wc, keep_recent=2)
    assert rep.compacted is True
    assert any(m["role"] == "system" and "SYSTEM you have tools" in m["content"] for m in out)


def test_recent_messages_are_never_evicted():
    out, rep = compact_messages(_msgs(), budget=300, count=wc, keep_recent=2)
    texts = " ".join(m["content"] for m in out)
    assert "RECENT so what is it?" in texts


def test_shrink_before_evict():
    """
    SHRINK BEFORE DELETE. This ordering is the whole ballgame.

    Deleting a small tool result that happens to hold the answer frees a handful of tokens
    and loses everything. Shrinking a huge pile of filler middle-out frees thousands and
    loses almost nothing, because its head and tail survive.

    Measured with the old order (evict-biggest-first, shrink as a last resort): a big
    USELESS message protected by the `recent` window pushed the algorithm into deleting a
    small USEFUL one to make up the difference -- and the agent then answered a question
    from a log it no longer had.
    """
    out, rep = compact_messages(_msgs(), budget=300, count=wc, keep_recent=2)
    texts = " ".join(m["content"] for m in out)

    assert "shrink-in-place" in rep.strategy
    assert rep.evicted_tokens > 4000
    assert sum(wc(m["content"]) for m in out) <= 300
    # the big dumps are reduced, not vaporised -- their heads still say what they were
    assert "HUGE" in texts, "shrinking must preserve the head of a big tool output"


def test_huge_recent_tool_output_is_shrunk_not_deadlocked():
    """
    The common agent case: the newest message is a giant tool result, so it sits in
    the protected window. Protecting it from deletion must not block compaction --
    it gets shrunk in place instead.
    """
    out, rep = compact_messages(_msgs(), budget=300, count=wc, keep_recent=2)
    assert rep.compacted is True
    assert sum(wc(m["content"]) for m in out) <= 300
    assert "shrink-in-place" in rep.strategy
    # it is still THERE, just smaller -- head preserved
    texts = " ".join(m["content"] for m in out)
    assert "HUGE2" in texts, "the recent tool output's head must survive"


def test_eviction_leaves_a_marker_in_place():
    out, rep = compact_messages(_msgs(), budget=300, count=wc, keep_recent=2)
    texts = " ".join(m["content"] for m in out)
    assert "contextpaw" in texts, "the hole must be visible to the model"
    assert "fetch it again" in texts, "the model must be told how to recover"


def test_under_budget_is_identity():
    m = [{"role": "user", "content": "hi"}]
    out, rep = compact_messages(m, budget=100, count=wc)
    assert out == m and rep.compacted is False


def test_raises_when_protected_content_alone_overflows():
    """We refuse rather than quietly eat the system prompt."""
    m = [
        {"role": "system", "content": "s " * 5000},
        {"role": "user", "content": "u " * 5000},
    ]
    with pytest.raises(TooLongToCompact):
        compact_messages(m, budget=100, count=wc, keep_recent=1)


# --------------------------- calibration ---------------------------

def test_calibrated_counter_only_ratchets_upward():
    """
    Under-counting is the one failure we cannot afford: it is what lets a prompt
    slip past the budget and get silently truncated upstream. So the ratio may
    only ever grow.
    """
    c = CalibratedCounter()
    start = c.stats()["ratio"]

    c.observe("x" * 1000, 100)          # 0.10 tok/char — lower than initial
    assert c.stats()["ratio"] == start, "must not learn a lower (unsafe) ratio"

    c.observe("x" * 1000, 900)          # 0.90 tok/char — higher, must adopt
    assert c.stats()["ratio"] > start
    assert c.count("x" * 1000) >= 900, "must now over-count, never under-count"


def test_calibrated_counter_overestimates_by_design():
    c = CalibratedCounter()
    text = "hello world this is a test"
    assert c.count(text) >= len(text) * 0.34


# --------------------------- summarizer chunking ---------------------------

def test_summarizer_chunks_cover_the_whole_span():
    """
    The bug this locks down: trimming the span head+tail before summarizing drops the
    MIDDLE -- which is precisely the region compaction evicted and the summarizer exists
    to rescue. Measured, that scored 0/3 facts recovered. Chunking scored 3/3.
    Every character of the span must appear in some chunk.
    """
    from contextpaw.summarize import Summarizer

    s = Summarizer(session=None, max_input_chars=100, max_chunks=999)
    span = "".join(f"[{i:04d}]" for i in range(500))  # 3000 chars, unique markers
    chunks = s._chunks(span)

    assert "".join(chunks) == span, "chunks must reconstruct the span exactly — nothing skipped"
    assert "[0250]" in "".join(chunks), "the MIDDLE must survive"


def test_summarizer_strides_across_a_pathological_span():
    """When a span is too big to summarize whole, sample ACROSS it -- never truncate one end."""
    from contextpaw.summarize import Summarizer

    s = Summarizer(session=None, max_input_chars=10, max_chunks=5)
    span = "".join(f"{i:02d}" for i in range(100))  # 200 chars -> 20 chunks, capped to 5
    chunks = s._chunks(span)

    assert len(chunks) == 5
    joined = "".join(chunks)
    # must reach the far end, not just the first 5 chunks
    assert span[-10:] in joined or span[-2:] in joined, "must sample the tail, not truncate to the head"


def test_short_prompts_never_poison_the_calibration():
    """
    Found by dogfooding, in production. `prompt_eval_count` includes the chat template's
    fixed overhead. On a 26-char "say hello" prompt that overhead IS the whole count --
    ~19 tokens / 26 chars = 0.73 tok/char. Because the counter keeps the MAX ratio (for
    safety), that single short prompt permanently poisoned the estimate and made every
    subsequent request compact far earlier than it needed to.
    """
    c = CalibratedCounter()
    start = c.stats()["ratio"]

    c.observe("Dis bonjour en une phrase.", 19)     # 26 chars -> 0.73 tok/char, garbage
    assert c.stats()["ratio"] == start, "a short prompt must never move the ratio"

    long_text = "Ceci est du texte de remplissage. " * 200  # ~6800 chars
    c.observe(long_text, int(len(long_text) * 0.40))
    assert c.stats()["ratio"] > start, "a long, representative sample must still teach us"


def test_calibration_reads_both_backends_field_names():
    """
    Found in production: 55 real requests through llama.cpp, `samples: 0`. The counter
    never learned anything, because we only looked for Ollama's `prompt_eval_count`.
    llama.cpp reports the same number as `usage.prompt_tokens` (OpenAI shape). The
    self-calibration this tool advertises was simply not running on half its backends.
    """
    from contextpaw.proxy import ContextPaw

    paw = ContextPaw()
    long_text = "Ceci est du texte de remplissage. " * 200  # > MIN_SAMPLE_CHARS
    before = paw.counter.stats()["samples"]

    # Ollama shape
    paw._calibrate({"prompt": long_text}, {"prompt_eval_count": 9999})
    assert paw.counter.stats()["samples"] == before + 1, "Ollama's field must be read"

    # llama.cpp / OpenAI shape
    paw._calibrate({"prompt": long_text}, {"usage": {"prompt_tokens": 9999}})
    assert paw.counter.stats()["samples"] == before + 2, "llama.cpp's field must be read too"


def test_pin_survives_a_restart(tmp_path, monkeypatch):
    """
    The unit runs Restart=always. If the proxy crashes mid-session, a forgotten pin means
    the next background request evicts the model the pin existed to protect. Measured: a
    plain `systemctl restart` dropped the pin silently. The pin must outlive the process.
    """
    import contextpaw.runtime as rt

    monkeypatch.setattr(rt, "STATE", tmp_path)
    monkeypatch.setattr(rt, "PIN_FILE", tmp_path / "pin.json")

    rt.RuntimeManager._save_pin("llamacpp")

    # a brand-new manager, as if the process had just restarted
    fresh = rt.RuntimeManager(session=None)
    assert fresh.pinned == "llamacpp", "the pin must be restored on startup"

    rt.RuntimeManager._save_pin(None)
    assert rt.RuntimeManager(session=None).pinned is None, "unpinning must persist too"


# --------------------------- routing & API defaults ---------------------------

class _Req:
    def __init__(self, path, headers=None):
        self.path = path
        self.headers = headers or {}


def test_only_v1_paths_go_to_llamacpp():
    """
    Broke `ollama run` in production. An earlier version routed "anything not /api/*" to
    llama.cpp -- so the CLI's bare `HEAD /` liveness ping went to a llama-server that was
    not running, and came back ConnectionRefused. We sit on Ollama's port pretending to be
    Ollama: the DEFAULT destination must be Ollama.
    """
    from contextpaw.proxy import ContextPaw

    paw = ContextPaw(arbitrate=True, llamacpp_cmd=["/bin/true"])

    assert paw._upstream(_Req("/v1/chat/completions")) == paw.llamacpp
    assert paw._upstream(_Req("/v1/models")) == paw.llamacpp

    assert paw._upstream(_Req("/")) == paw.ollama, "the liveness ping must reach Ollama"
    assert paw._upstream(_Req("/api/chat")) == paw.ollama
    assert paw._upstream(_Req("/api/tags")) == paw.ollama


def test_stream_default_differs_between_the_two_apis():
    """
    Ollama defaults `stream` to TRUE when the field is absent (it returns ndjson); the
    OpenAI API defaults it to FALSE. Assuming "absent means false" made us parse an ndjson
    stream as a single JSON object -- and `ollama run` sends exactly such a request (a
    preload with no `stream` key), so the CLI died with a 500.
    """
    def streaming_for(path, body):
        if "stream" in body:
            return bool(body["stream"])
        return path.startswith("/api/")

    assert streaming_for("/api/generate", {}) is True, "Ollama streams by default"
    assert streaming_for("/v1/chat/completions", {}) is False, "OpenAI does not"
    assert streaming_for("/api/generate", {"stream": False}) is False, "explicit wins"
    assert streaming_for("/v1/chat/completions", {"stream": True}) is True


# --------------------------- orphan tool messages ---------------------------

def test_orphan_tool_message_is_rescued():
    """
    A `role: "tool"` message NOT preceded by an assistant carrying `tool_calls` is an
    orphan, and the chat template silently discards it. Measured on gemma-4-12b, same log,
    same question, only the role changed:

        orphan role=tool ............ 0/3 facts  ("Please provide the log you are
                                                   referring to!")
        assistant.tool_calls + tool .. 3/3 facts
        remapped to role=user ....... 3/3 facts

    Ollama returns 200 either way. The tool output simply never reaches the model.
    """
    from contextpaw.compact import rescue_orphan_tools

    out, n = rescue_orphan_tools([
        {"role": "user", "content": "find the bug"},
        {"role": "tool", "content": "ERROR on orion-7"},        # orphan
    ])
    assert n == 1
    assert out[1]["role"] == "user", "an orphan tool result must be re-parented, not lost"
    assert "orion-7" in out[1]["content"], "its content must survive intact"
    assert "TOOL OUTPUT" in out[1]["content"], "and be labelled as tool output"


def test_properly_parented_tool_message_is_left_alone():
    from contextpaw.compact import rescue_orphan_tools

    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read"}}]},
        {"role": "tool", "content": "ERROR on orion-7"},
    ]
    out, n = rescue_orphan_tools(msgs)
    assert n == 0 and out[1]["role"] == "tool", "a valid tool sequence must not be touched"


def test_compaction_does_not_orphan_a_tool_result():
    """
    The one that matters, and the one we could have shipped: OUR OWN eviction can create an
    orphan. Drop the assistant message that carried the tool_calls, keep the tool result
    that followed it, and we have handed the runtime something it will delete without a
    word. The tool output survives our budget and dies in the template.
    """
    msgs = [
        {"role": "system", "content": "sys " * 10},
        {"role": "user", "content": "GOAL find the bug"},
        {"role": "assistant", "content": "calling read_file",
         "tool_calls": [{"function": {"name": "read_file"}}]},
        {"role": "tool", "content": "NEEDLE-ANANAS " + ("pad " * 400)},
        {"role": "user", "content": "so what was it?"},
    ]
    out, rep = compact_messages(msgs, budget=120, count=wc, keep_recent=1)

    for i, m in enumerate(out):
        if m.get("role") == "tool":
            prev = out[i - 1] if i else None
            assert prev and prev.get("role") == "assistant" and prev.get("tool_calls"), (
                "a tool message survived without its parent — the template will eat it"
            )
    texts = " ".join(str(m.get("content", "")) for m in out)
    assert "NEEDLE-ANANAS" in texts, "the tool output's head must reach the model somehow"


def test_first_user_message_is_the_goal_and_is_pinned():
    """
    Measured: the task goal was evicted at 29 tokens, and the agent then confidently
    answered a question it no longer knew it had been asked. `system` did not protect it,
    and in a long loop it is nowhere near `recent`.
    """
    msgs = [
        {"role": "system", "content": "sys " * 5},
        {"role": "user", "content": "GOAL-SENTINEL find why the deploy failed"},
    ] + [
        {"role": "assistant", "content": "chatter " * 200} for _ in range(6)
    ] + [
        {"role": "user", "content": "well?"},
    ]
    out, rep = compact_messages(msgs, budget=300, count=wc, keep_recent=2)
    texts = " ".join(str(m.get("content", "")) for m in out)
    assert "GOAL-SENTINEL" in texts, "an agent that forgets its task is worse than one that fails"
    assert sum(wc(m.get("content", "")) for m in out) <= 300


def test_marker_overhead_is_a_real_structural_limit():
    """
    Honest limit, found while fixing the above and worth stating rather than hiding.

    Middle-out shrinking pays a ~30-token elision marker PER MESSAGE. Under a tight budget
    with many messages, the markers alone can exceed the budget: 8 messages x 30 tokens =
    240, against a budget of 150. There is no clever fix — you cannot annotate 8 holes in
    fewer tokens than it takes to describe 8 holes.

    So at extreme compression ratios ContextPaw raises rather than pretending. That is the
    correct behaviour: refusing is honest, and silently shipping a payload that lost the
    system prompt is exactly what we exist to prevent.
    """
    msgs = [{"role": "system", "content": "sys " * 5},
            {"role": "user", "content": "GOAL find it"}] \
         + [{"role": "assistant", "content": "chatter " * 200} for _ in range(6)] \
         + [{"role": "user", "content": "well?"}]

    with pytest.raises(TooLongToCompact):
        compact_messages(msgs, budget=150, count=wc, keep_recent=2)

    # give it enough room to pay for its own annotations and it succeeds
    out, rep = compact_messages(msgs, budget=400, count=wc, keep_recent=2)
    assert sum(wc(m.get("content", "")) for m in out) <= 400
