"""Evaluators: deterministic code checks + a binary LLM-judge with a written critique.

Design stance:
  * **Code checks first.** They are cheap, deterministic, and don't need calibration. Reach for
    the judge only for things code can't express (helpfulness, faithfulness, tone).
  * **Binary judge.** The judge returns pass/fail + prose, never a 1-5 score (see README FAQ).
  * **Bias mitigation.** LLM judges have known biases — verbosity (longer answers score higher),
    position (order effects in pairwise setups), and self-preference (favouring same-family
    outputs). Mitigations wired into the prompt/usage here: judge on an absolute rubric (not
    pairwise), explicitly instruct to ignore length/style and grade only the rubric, and — the
    real safety net — calibrate against a human anchor set every run (see evalgate.calibration).
"""

from __future__ import annotations

import json
import re
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
# LLM judge — prompt + parsing fully implemented; network isolated behind ``_chat``.
# --------------------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator of an AI agent's output. "
    "Decide whether the OUTPUT satisfies the RUBRIC for the given INPUT.\n\n"
    "Rules:\n"
    "1. Answer with a BINARY verdict only: pass or fail. Do NOT use a 1-5 or numeric score.\n"
    "2. Grade ONLY against the rubric. Ignore answer length, verbosity, and writing style — "
    "a longer answer is not a better answer.\n"
    "3. Do not favour any particular model or phrasing; judge substance, not surface form.\n"
    "4. Provide a concise written critique (1-3 sentences) justifying the verdict, citing the "
    "specific rubric criterion that was met or violated.\n\n"
    'Respond with EXACTLY one JSON object: {"pass": <true|false>, "critique": "<text>"}'
)


class LLMJudge:
    """Binary LLM-judge over an OpenAI-compatible endpoint (Groq free tier by default)."""

    name = "llm_judge"

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = client  # inject a client in tests; built lazily otherwise

    def _get_client(self) -> Any:
        if self._client is None:
            # Imported lazily so importing this module never requires the SDK or a network call.
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.settings.groq_api_key,
                base_url=self.settings.groq_base_url,
            )
        return self._client

    def _chat(self, messages: list[dict[str, str]]) -> str:
        """Single network hop to the judge model. Everything network lives here.

        TODO: add timeout, retry/backoff, and structured-output / JSON-mode enforcement.
        """
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.settings.judge_model,
            messages=messages,
            temperature=0.0,  # deterministic judgments
        )
        return resp.choices[0].message.content or ""

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
    def _parse(raw: str) -> JudgeResult:
        """Parse the judge's reply into a strict binary JudgeResult.

        Tolerant of code fences / stray prose: extracts the first JSON object, else falls back
        to keyword sniffing. Ambiguous replies fail closed (passed=False) with a note.
        """
        text = raw.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                passed = bool(data["pass"])
                critique = str(data.get("critique", "")).strip()
                return JudgeResult(passed=passed, critique=critique)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
        # Fallback: sniff a clear pass/fail token.
        lowered = text.lower()
        if re.search(r"\bpass\b", lowered) and not re.search(r"\bfail\b", lowered):
            return JudgeResult(passed=True, critique=text[:500])
        if re.search(r"\bfail\b", lowered):
            return JudgeResult(passed=False, critique=text[:500])
        return JudgeResult(
            passed=False, critique=f"Unparseable judge reply (fail-closed): {text[:300]}"
        )

    def judge(self, input_text: str, output_text: str, rubric: str) -> JudgeResult:
        raw = self._chat(self._build_messages(input_text, output_text, rubric))
        return self._parse(raw)

    def evaluate(
        self, trace: Trace, rubric: str = "The output correctly answers the input."
    ) -> EvalResult:
        """Evaluator-protocol adapter: judge a trace's root input/output."""
        result = self.judge(trace.root_input or "", trace.root_output or "", rubric)
        return EvalResult(evaluator=self.name, passed=result.passed, critique=result.critique)
