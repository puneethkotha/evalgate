"""Domain models for EvalGate.

Pydantic models are the in-memory contract shared across ingest, analysis, evaluation,
calibration, and the gate. They are shaped after the OpenTelemetry **GenAI** semantic
conventions (``gen_ai.*`` attributes) so ingesting real OTLP spans is a thin mapping.

The persisted/ORM representation lives in :mod:`evalgate.ingest` (``Base`` / ``TraceRecord``),
which owns the SQLAlchemy table and the pgvector embedding column. It is kept there so the
DB engine and pgvector types are only imported when storage is actually used.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Span(BaseModel):
    """A single OTel GenAI span.

    Maps to ``gen_ai.*`` semantic-convention attributes:
      * ``gen_ai_operation``  -> ``gen_ai.operation.name`` (chat | tool | embeddings | ...)
      * ``input``             -> ``gen_ai.prompt`` / input messages
      * ``output``            -> ``gen_ai.completion`` / output messages
      * ``tokens``            -> ``gen_ai.usage.total_tokens``
    """

    name: str
    span_kind: str = "INTERNAL"  # OTel SpanKind: INTERNAL | CLIENT | SERVER | ...
    gen_ai_operation: str | None = None
    input: str | None = None
    output: str | None = None
    tokens: int | None = None
    latency_ms: float = 0.0
    status: str = "ok"  # "ok" | "error"


class Trace(BaseModel):
    """A full agent trace: an ordered set of spans plus the root request/response."""

    trace_id: str
    spans: list[Span] = Field(default_factory=list)
    root_input: str | None = None
    root_output: str | None = None
    status: str = "ok"  # "ok" | "error"

    @property
    def latency_ms(self) -> float:
        """End-to-end latency as the sum of span latencies (cheap proxy for wall-clock)."""
        return sum(s.latency_ms for s in self.spans)

    @property
    def is_failure(self) -> bool:
        return self.status == "error" or any(s.status == "error" for s in self.spans)

    def text(self) -> str:
        """Flat text view of the trace, used for embedding / clustering."""
        return "\n".join(
            filter(None, [self.root_input, self.root_output, *[s.output for s in self.spans]])
        )


class EvalResult(BaseModel):
    """Outcome of a single evaluator (code check or judge) against one trace."""

    evaluator: str
    passed: bool
    critique: str = ""


class JudgeResult(BaseModel):
    """Binary verdict from the LLM-judge, with a required written critique."""

    passed: bool
    critique: str


class AnchorExample(BaseModel):
    """One human-labeled anchor: the ground truth the judge is calibrated against."""

    input: str
    output: str
    human_label: bool  # True = a human says this output passes


class CalibrationReport(BaseModel):
    """Judge-vs-human agreement on the anchor set for a single run."""

    kappa: float  # Cohen's kappa (chance-corrected agreement); NaN if degenerate
    tpr: float  # true-positive rate  (sensitivity)
    tnr: float  # true-negative rate  (specificity)
    n: int
    drifted: bool  # True => judge no longer trusted; block the run


class FailureCluster(BaseModel):
    """One node of the failure taxonomy produced by error analysis."""

    label: str
    size: int
    exemplars: list[str] = Field(default_factory=list)
