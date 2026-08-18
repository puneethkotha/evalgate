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
    """Judge-vs-human agreement on the anchor set for a single run.

    We report a *bundle* of agreement statistics, not one number, because any single
    coefficient can mislead: Cohen's kappa collapses under class imbalance (the prevalence
    paradox), while raw agreement ignores chance. Gwet's AC1 is prevalence-robust, so a large
    gap between kappa and AC1 flags an imbalanced anchor set (``paradox_flag``).
    """

    kappa: float  # Cohen's kappa (chance-corrected agreement); NaN if degenerate
    ac1: float  # Gwet's AC1 (prevalence-robust agreement)
    raw_agreement: float  # uncorrected fraction of matching labels
    tpr: float  # true-positive rate  (sensitivity)
    tnr: float  # true-negative rate  (specificity)
    fpr: float  # false-positive rate (1 - tnr); used for bias-correcting the pass-rate
    prevalence: float  # fraction of human "pass" labels in the anchor set
    band: str  # Landis-Koch qualitative band for kappa
    paradox_flag: bool  # kappa and AC1 diverge => anchor set likely imbalanced
    n: int
    min_kappa: float  # threshold this report was judged against
    drifted: bool  # True => judge no longer trusted; block the run


class FailureCluster(BaseModel):
    """One node of the failure taxonomy produced by error analysis."""

    label: str
    size: int
    exemplars: list[str] = Field(default_factory=list)
    terms: list[str] = Field(default_factory=list)  # top distinctive terms (auto-label)


class EvaluatorStat(BaseModel):
    """Per-evaluator pass breakdown, for the gate report + dashboard."""

    evaluator: str
    passed: int
    total: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else float("nan")


class PassRateReport(BaseModel):
    """Pass-rate point estimate with a confidence interval."""

    point: float
    lower: float
    upper: float
    method: str  # "wilson" | "bootstrap"
    confidence: float
    passed: int
    n: int
    corrected: float | None = None  # bias-corrected using judge FPR/TPR, when available


class VersionDelta(BaseModel):
    """Paired v1-vs-v2 comparison (McNemar) on the same eval set."""

    b: int  # regressions: v1 passed, v2 failed
    c: int  # fixes: v1 failed, v2 passed
    p_value: float
    verdict: str  # "improved" | "regressed" | "inconclusive"


class GateReport(BaseModel):
    """The full, structured outcome of a gate run — the one contract the CLI, API,
    dashboard, and GitHub Action all render from."""

    passed: bool
    reasons: list[str] = Field(default_factory=list)
    pass_rate: PassRateReport
    min_pass_rate: float
    calibration: CalibrationReport | None = None
    delta: VersionDelta | None = None
    by_evaluator: list[EvaluatorStat] = Field(default_factory=list)
