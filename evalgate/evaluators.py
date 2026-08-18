"""Evaluators: deterministic code checks + a binary LLM-judge with a written critique.

Design stance:
  * **Code checks first.** They are cheap, deterministic, and don't need calibration. Reach for
    the judge only for things code can't express (helpfulness, faithfulness, tone).
  * **Binary judge, chain-of-thought first.** The judge reasons, then commits to pass/fail +
    prose — never a 1-5 score (see README FAQ). Temperature 0 and JSON output make it
    reproducible; the critique is what makes a failure actionable and what a human anchors to.
  * **Bias mitigation.** LLM judges have known biases — verbosity (longer answers score higher,
    ~15-30pt), position (order effects in pairwise setups, ~10-15pt), and self-preference
    (favouring same-family outputs, ~10-25%). Mitigations wired in here: judge on an absolute
    rubric (not pairwise) for scoring; instruct to ignore length/style; for pairwise comparison
    run **both orders and call it a tie on disagreement**; and — the real safety net —
    calibrate against a human anchor set every run (see :mod:`evalgate.calibration`).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Protocol, runtime_checkable

from .config import Settings, get_settings
from .models import EvalResult, JudgeResult, Trace


@runtime_checkable
class Evaluator(Protocol):
    """Anything that turns a trace into a pass/fail EvalResult."""

    name: str

    def evaluate(self, trace: Trace) -> EvalResult: ...


# --------------------------------------------------------------------------------------
# Code checks — deterministic, fully implemented.
# --------------------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_CC_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")

_PII_PATTERNS = (_EMAIL_RE, _SSN_RE, _PHONE_RE, _CC_RE)


class CodeChecks:
    """Library of deterministic, judge-free checks."""

    @staticmethod
    def schema_valid(output: Any, schema: dict[str, type | tuple[type, ...] | None]) -> bool:
        """Validate ``output`` against a lightweight schema: ``{field: expected_type}``.

        ``output`` may be a dict or a JSON string. ``expected_type`` of ``None`` only requires
        the key to be present. Keeps a zero-dependency footprint (no jsonschema).
        """
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except (json.JSONDecodeError, ValueError):
                return False
        if not isinstance(output, dict):
            return False
        for key, expected in schema.items():
            if key not in output:
                return False
            if expected is not None and not isinstance(output[key], expected):
                return False
        return True

    @staticmethod
    def tool_call_succeeded(trace: Trace) -> bool:
        """True iff the trace contains at least one tool span and no tool span errored."""
        tool_spans = [
            s
            for s in trace.spans
            if (s.gen_ai_operation or "").lower() in {"tool", "execute_tool", "tool_call"}
        ]
        if not tool_spans:
            return False
        return all(s.status == "ok" for s in tool_spans)

    @staticmethod
    def no_pii(text: str) -> bool:
        """True iff no obvious PII (email / SSN / phone / card-like number) is present."""
        if not text:
            return True
        return not any(p.search(text) for p in _PII_PATTERNS)

    @staticmethod
    def latency_budget(trace: Trace, ms: float) -> bool:
        """True iff end-to-end trace latency is within ``ms`` milliseconds."""
        return trace.latency_ms <= ms


# --------------------------------------------------------------------------------------
# LLM judge — prompts + parsing fully implemented; network isolated behind ``_chat``.
# --------------------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator of an AI agent's output. "
    "Decide whether the OUTPUT satisfies the RUBRIC for the given INPUT.\n\n"
    "Rules:\n"
    "1. First reason briefly, THEN commit to a BINARY verdict: \"pass\" or \"fail\". "
    "Do NOT use a 1-5 or numeric score.\n"
    "2. Grade ONLY against the rubric. Ignore answer length, verbosity, and writing style — "
    "a longer answer is not a better answer.\n"
    "3. Do not favour any particular model or phrasing; judge substance, not surface form.\n"
    "4. The critique must be 1-3 sentences citing the specific rubric criterion met or "
    "violated.\n\n"
    "Respond with EXACTLY one JSON object and nothing else:\n"
    '{"reasoning": "<brief chain of thought>", "verdict": "pass" | "fail", '
    '"critique": "<text>"}'
)

_PAIRWISE_SYSTEM = (
    "You are comparing two candidate OUTPUTS (A and B) for the same INPUT against a RUBRIC. "
    "Decide which better satisfies the rubric. Ignore length, verbosity, style, and the order "
    "in which A and B are presented; judge substance only.\n\n"
    'Respond with EXACTLY one JSON object: {"reasoning": "<brief>", "winner": "A" | "B" | "tie"}'
)


class LLMJudge:
    """Binary LLM-judge over an OpenAI-compatible endpoint (Groq free tier by default)."""

    name = "llm_judge"

    def __init__(self, settings: Settings | None = None, client: Any | None = None,
                 max_retries: int = 3, timeout: float = 30.0) -> None:
        self.settings = settings or get_settings()
        self._client = client  # inject a client in tests; built lazily otherwise
        self.max_retries = max_retries
        self.timeout = timeout

    def _get_client(self) -> Any:
        if self._client is None:
            # Imported lazily so importing this module never requires the SDK or a network call.
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.settings.groq_api_key,
                base_url=self.settings.groq_base_url,
                timeout=self.timeout,
            )
        return self._client

    def _chat(self, messages: list[dict[str, str]]) -> str:
        """Single logical call to the judge model, with retry/backoff. All network lives here.

        Requests JSON output (OpenAI-compatible ``response_format``); silently retries without
        it if the endpoint rejects the parameter, so the judge works across providers.
        """
        client = self._get_client()
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                try:
                    resp = client.chat.completions.create(
                        model=self.settings.judge_model,
                        messages=messages,
                        temperature=0.0,  # deterministic judgments
                        response_format={"type": "json_object"},
                    )
                except TypeError:
                    # Injected/older client without response_format support.
                    resp = client.chat.completions.create(
                        model=self.settings.judge_model,
                        messages=messages,
                        temperature=0.0,
                    )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - broad by design; retry any transient error
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(0.5 * (2**attempt))  # 0.5s, 1s, 2s exponential backoff
        raise RuntimeError(f"judge call failed after {self.max_retries} attempts: {last_exc}")

    def _build_messages(
        self, input_text: str, output_text: str, rubric: str
    ) -> list[dict[str, str]]:
        user = (
            f"RUBRIC:\n{rubric}\n\n"
            f"INPUT:\n{input_text}\n\n"
            f"OUTPUT:\n{output_text}\n\n"
            "Return only the JSON object."
        )
        return [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any] | None:
        text = raw.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None

    @classmethod
    def _parse(cls, raw: str) -> JudgeResult:
        """Parse the judge's reply into a strict binary JudgeResult.

        Accepts the current ``{reasoning, verdict, critique}`` shape and the legacy
        ``{pass, critique}`` shape. Tolerant of code fences / stray prose; ambiguous replies
        fail closed (passed=False) with a note.
        """
        data = cls._extract_json(raw)
        if data is not None:
            critique = str(data.get("critique") or data.get("reasoning") or "").strip()
            if "verdict" in data:
                verdict = str(data["verdict"]).strip().lower()
                if verdict in {"pass", "fail"}:
                    return JudgeResult(passed=verdict == "pass", critique=critique)
            if "pass" in data:
                try:
                    return JudgeResult(passed=bool(data["pass"]), critique=critique)
                except (TypeError, ValueError):
                    pass
        # Fallback: sniff a clear pass/fail token.
        lowered = raw.strip().lower()
        if re.search(r"\bpass\b", lowered) and not re.search(r"\bfail\b", lowered):
            return JudgeResult(passed=True, critique=raw.strip()[:500])
        if re.search(r"\bfail\b", lowered):
            return JudgeResult(passed=False, critique=raw.strip()[:500])
        return JudgeResult(
            passed=False, critique=f"Unparseable judge reply (fail-closed): {raw.strip()[:300]}"
        )

    def judge(self, input_text: str, output_text: str, rubric: str) -> JudgeResult:
        raw = self._chat(self._build_messages(input_text, output_text, rubric))
        return self._parse(raw)

    def judge_pairwise(self, input_text: str, output_a: str, output_b: str,
                       rubric: str) -> dict[str, Any]:
        """Position-bias-robust pairwise comparison.

        Runs the comparison in BOTH orders (A vs B, then B vs A). Position bias swings pairwise
        win-rate by ~10-15 points, so we only trust a winner both orders agree on; a
        disagreement is reported as a position-determined ``tie``. This is the discipline that
        feeds a trustworthy McNemar decision for v1-vs-v2.
        """
        def _compare(first: str, second: str) -> str:
            user = (
                f"RUBRIC:\n{rubric}\n\nINPUT:\n{input_text}\n\n"
                f"OUTPUT A:\n{first}\n\nOUTPUT B:\n{second}\n\nReturn only the JSON object."
            )
            raw = self._chat([
                {"role": "system", "content": _PAIRWISE_SYSTEM},
                {"role": "user", "content": user},
            ])
            data = self._extract_json(raw) or {}
            return str(data.get("winner", "tie")).strip().upper()[:1]  # "A" | "B" | "T"

        fwd = _compare(output_a, output_b)          # A=a, B=b  -> winner in {A,B}
        rev = _compare(output_b, output_a)           # A=b, B=a  -> winner in {A,B}
        # Map both back to the real candidates (a / b / tie).
        fwd_real = {"A": "a", "B": "b"}.get(fwd, "tie")
        rev_real = {"A": "b", "B": "a"}.get(rev, "tie")
        consistent = fwd_real == rev_real and fwd_real != "tie"
        winner = fwd_real if consistent else "tie"
        return {"winner": winner, "consistent": consistent, "forward": fwd_real,
                "reverse": rev_real}

    def evaluate(
        self, trace: Trace, rubric: str = "The output correctly answers the input."
    ) -> EvalResult:
        """Evaluator-protocol adapter: judge a trace's root input/output."""
        result = self.judge(trace.root_input or "", trace.root_output or "", rubric)
        return EvalResult(evaluator=self.name, passed=result.passed, critique=result.critique)
