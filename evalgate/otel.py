"""OpenTelemetry GenAI ingestion adapter.

Maps decoded OTel **GenAI** spans onto EvalGate :class:`~evalgate.models.Trace` objects. The
GenAI semantic conventions are still ``Development`` and have churned (v1.37 renamed
``gen_ai.system`` -> ``gen_ai.provider.name`` and replaced per-message span events with the
aggregated ``gen_ai.input.messages`` / ``gen_ai.output.messages`` attributes), so all of that
version-sensitivity is isolated here — a future spec change is a one-file edit.

Accepts spans as plain dicts (however you decoded OTLP), tolerating both nested-``attributes``
and flat shapes, then groups them by ``trace_id`` into traces.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .models import Span, Trace

# Operation names that carry the request/response we treat as the trace root.
_ROOT_OPS = {"chat", "invoke_agent", "text_completion", "generate_content"}


def _attr(span: dict[str, Any], *keys: str) -> Any:
    """Look up the first present key in either the flat span or its ``attributes`` dict."""
    attrs = span.get("attributes", {}) or {}
    for k in keys:
        if k in span and span[k] is not None:
            return span[k]
        if k in attrs and attrs[k] is not None:
            return attrs[k]
    return None


def _text_of(value: Any) -> str | None:
    """Coerce a message value (string, list of message dicts, or dict) into flat text."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _text_of(value.get("content")) or str(value)
    if isinstance(value, list):
        parts = [t for m in value if (t := _text_of(m))]
        return "\n".join(parts) if parts else None
    return str(value)


def _status(span: dict[str, Any]) -> str:
    raw = _attr(span, "status", "status_code")
    if isinstance(raw, dict):
        raw = raw.get("status_code") or raw.get("code")
    return "error" if str(raw).upper() in {"ERROR", "STATUS_CODE_ERROR"} else "ok"


def _latency_ms(span: dict[str, Any]) -> float:
    ms = _attr(span, "latency_ms")
    if ms is not None:
        return float(ms)
    start = _attr(span, "start_time_unix_nano", "start_time")
    end = _attr(span, "end_time_unix_nano", "end_time")
    if start is not None and end is not None:
        return max(0.0, (float(end) - float(start)) / 1e6)
    return 0.0


def _trace_id(span: dict[str, Any]) -> str:
    tid = span.get("trace_id") or (span.get("context") or {}).get("trace_id") \
        or _attr(span, "gen_ai.conversation.id", "trace_id")
    return str(tid) if tid is not None else "unknown"


def span_to_span(span: dict[str, Any]) -> Span:
    """Map one OTel GenAI span dict onto a :class:`~evalgate.models.Span`."""
    in_tok = _attr(span, "gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens")
    out_tok = _attr(span, "gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens")
    tokens = None
    if in_tok is not None or out_tok is not None:
        tokens = int(in_tok or 0) + int(out_tok or 0)
    return Span(
        name=str(span.get("name", "span")),
        span_kind=str(span.get("kind", span.get("span_kind", "INTERNAL"))),
        gen_ai_operation=_attr(span, "gen_ai.operation.name"),
        input=_text_of(_attr(span, "gen_ai.input.messages", "input", "gen_ai.prompt")),
        output=_text_of(_attr(span, "gen_ai.output.messages", "output", "gen_ai.completion")),
        tokens=tokens,
        latency_ms=_latency_ms(span),
        status=_status(span),
    )


def spans_to_traces(spans: Iterable[dict[str, Any]]) -> list[Trace]:
    """Group OTel GenAI spans by ``trace_id`` into :class:`~evalgate.models.Trace` objects.

    The root input is the first root-operation span's input; the root output is the last
    root-operation span's output; the trace fails if any span errored.
    """
    grouped: dict[str, list[Span]] = {}
    order: list[str] = []
    for raw in spans:
        tid = _trace_id(raw)
        if tid not in grouped:
            grouped[tid] = []
            order.append(tid)
        grouped[tid].append(span_to_span(raw))

    traces: list[Trace] = []
    for tid in order:
        sp = grouped[tid]
        root_spans = [s for s in sp if (s.gen_ai_operation or "").lower() in _ROOT_OPS] or sp
        root_input = next((s.input for s in root_spans if s.input), None)
        root_output = next((s.output for s in reversed(root_spans) if s.output), None)
        status = "error" if any(s.status == "error" for s in sp) else "ok"
        traces.append(Trace(trace_id=tid, spans=sp, root_input=root_input,
                            root_output=root_output, status=status))
    return traces


def load_otel_jsonl(path: str) -> list[Trace]:
    """Load OTel GenAI spans from a JSONL file (one span per line) into traces."""
    import json

    spans: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                spans.append(json.loads(line))
    return spans_to_traces(spans)


def as_sequence(spans: Sequence[dict[str, Any]]) -> list[Trace]:
    """Convenience alias for :func:`spans_to_traces` over an in-memory sequence."""
    return spans_to_traces(spans)
