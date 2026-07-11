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


def test_biggest_evictable_messages_go_first():
    """Stale tool dumps are the cheapest thing to lose and the biggest win."""
    out, rep = compact_messages(_msgs(), budget=300, count=wc, keep_recent=2)
    texts = " ".join(m["content"] for m in out)
    assert "HUGE " not in texts, "the old (evictable) tool dump should be gone"
    assert rep.evicted_tokens > 4000


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
