"""Tests for the LLM judge: parsing (new + legacy shapes) and order-swapped pairwise mode.

The network is fully mocked via an injected fake OpenAI-compatible client, so these run offline.
"""

from evalgate.config import Settings
from evalgate.evaluators import LLMJudge


def _resp(content: str):
    msg = type("Msg", (), {"content": content})
    choice = type("Choice", (), {"message": msg})
    return type("Resp", (), {"choices": [choice]})


def _client(responder):
    completions = type("Completions", (), {
        "create": staticmethod(lambda **kw: _resp(responder(kw["messages"])))
    })
    chat = type("Chat", (), {"completions": completions()})
    return type("Client", (), {"chat": chat()})()


def _judge(responder) -> LLMJudge:
    return LLMJudge(settings=Settings(groq_api_key="x"), client=_client(responder))


# --- parsing --------------------------------------------------------------------------

def test_parse_new_shape_pass():
    r = LLMJudge._parse('{"reasoning": "all steps present", "verdict": "pass", "critique": "ok"}')
    assert r.passed is True
    assert r.critique == "ok"


def test_parse_new_shape_fail():
    r = LLMJudge._parse('{"reasoning": "missing sink", "verdict": "fail", "critique": "no output"}')
    assert r.passed is False


def test_parse_legacy_pass_boolean():
    r = LLMJudge._parse('{"pass": true, "critique": "legacy shape"}')
    assert r.passed is True
    assert r.critique == "legacy shape"


def test_parse_tolerates_code_fence_and_prose():
    raw = 'Sure!\n```json\n{"verdict": "pass", "critique": "fine"}\n```'
    r = LLMJudge._parse(raw)
    assert r.passed is True


def test_parse_critique_falls_back_to_reasoning():
    r = LLMJudge._parse('{"reasoning": "the plan matches", "verdict": "pass"}')
    assert r.passed is True
    assert "matches" in r.critique


def test_parse_ambiguous_fails_closed():
    r = LLMJudge._parse("I am not sure about this output.")
    assert r.passed is False
    assert "fail-closed" in r.critique.lower()


def test_parse_fail_token_fallback():
    assert LLMJudge._parse("this should fail").passed is False


# --- judge() --------------------------------------------------------------------------

def test_judge_calls_client_and_parses():
    j = _judge(lambda msgs: '{"verdict": "pass", "critique": "good"}')
    result = j.judge("in", "out", "rubric")
    assert result.passed is True


# --- pairwise (position-bias robust) --------------------------------------------------

def test_pairwise_position_bias_becomes_tie():
    # A judge that always picks the first slot (pure position bias) must resolve to a tie,
    # because the two orders disagree on the real winner.
    j = _judge(lambda msgs: '{"winner": "A"}')
    out = j.judge_pairwise("in", "cand_a", "cand_b", "rubric")
    assert out["winner"] == "tie"
    assert out["consistent"] is False


def test_pairwise_consistent_winner():
    # Responder that genuinely prefers whichever slot holds the GOOD candidate -> order-invariant.
    def responder(msgs):
        user = msgs[-1]["content"]
        a = user.split("OUTPUT A:")[1].split("OUTPUT B:")[0]
        return '{"winner": "A"}' if "GOOD" in a else '{"winner": "B"}'

    j = _judge(responder)
    out = j.judge_pairwise("in", "GOOD candidate", "weak candidate", "rubric")
    assert out["winner"] == "a"
    assert out["consistent"] is True


def test_judge_retries_then_raises():
    calls = {"n": 0}

    def boom(**kw):
        calls["n"] += 1
        raise RuntimeError("network down")

    completions = type("C", (), {"create": staticmethod(boom)})
    client = type("Cl", (), {"chat": type("Ch", (), {"completions": completions()})()})()
    j = LLMJudge(settings=Settings(groq_api_key="x"), client=client, max_retries=2)
    try:
        j.judge("i", "o", "r")
    except RuntimeError as e:
        assert "after 2 attempts" in str(e)
    else:
        raise AssertionError("expected RuntimeError after retries")
    assert calls["n"] == 2
