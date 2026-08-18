"""CI gate: turn a batch of eval results into a structured report + a pass/fail exit code.

Three independent ways to fail:
  1. **Agent regression** — the *lower bound* of a Wilson score interval on the pass-rate falls
     below ``min_pass_rate``. Using the CI lower bound (not the point estimate) means a small,
     lucky sample can't sneak past the gate.
  2. **Significant regression vs baseline** — when a baseline (e.g. ``main``) is supplied, a
     McNemar paired test on the same eval set says the change made things significantly worse.
  3. **Judge drift** — calibration says the judge no longer agrees with humans. A drifted judge
     invalidates every judgment this run, so we fail regardless of the pass-rate.

``evaluate_gate`` returns a :class:`~evalgate.models.GateReport` (pure, no I/O); ``render_gate``
turns it into the terminal readout; ``ci_gate`` wires both and returns a shell exit code.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np

from . import stats
from .calibration import corrected_pass_rate
from .models import (
    CalibrationReport,
    EvalResult,
    EvaluatorStat,
    GateReport,
    PassRateReport,
    VersionDelta,
)


def bootstrap_ci(
    passes: Sequence[float],
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a mean pass-rate.

    Returns ``(point_estimate, lower, upper)`` at confidence ``1 - alpha``. Kept for composite
    metrics; for a simple proportion the gate defaults to the Wilson interval (see
    :func:`evalgate.stats.wilson_ci`), which is better-behaved near 0/1 and at small n.
    """
    arr = np.asarray(passes, dtype=float)
    n = arr.size
    if n == 0:
        return 0.0, 0.0, 0.0
    point = float(arr.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    lower = float(np.quantile(boot_means, alpha / 2))
    upper = float(np.quantile(boot_means, 1 - alpha / 2))
    return point, lower, upper


def _pass_rate_report(
    passes: Sequence[bool],
    confidence: float,
    method: str,
    calibration: CalibrationReport | None,
    judge_derived: bool,
) -> PassRateReport:
    n = len(passes)
    passed = sum(1 for p in passes if p)
    if method == "bootstrap":
        alpha = 1 - confidence
        point, lower, upper = bootstrap_ci([1.0 if p else 0.0 for p in passes], alpha=alpha)
    else:
        point, lower, upper = stats.wilson_ci(passed, n, confidence=confidence)
    # Bias-correction only applies when the pass signal came from the judge; correcting a
    # deterministic code-check pass-rate by the judge's error profile would be a category error.
    corrected = None
    if judge_derived and calibration is not None:
        corrected = corrected_pass_rate(point, calibration.tpr, calibration.fpr)
    return PassRateReport(
        point=point, lower=lower, upper=upper, method=method,
        confidence=confidence, passed=passed, n=n, corrected=corrected,
    )


def _evaluator_stats(results: Sequence[EvalResult]) -> list[EvaluatorStat]:
    agg: dict[str, list[int]] = {}
    for r in results:
        bucket = agg.setdefault(r.evaluator, [0, 0])
        bucket[0] += 1 if r.passed else 0
        bucket[1] += 1
    return [EvaluatorStat(evaluator=k, passed=v[0], total=v[1]) for k, v in sorted(agg.items())]


def evaluate_gate(
    passes: Sequence[bool],
    min_pass_rate: float,
    calibration: CalibrationReport | None = None,
    baseline_passes: Sequence[bool] | None = None,
    by_evaluator: Sequence[EvaluatorStat] | None = None,
    confidence: float = 0.95,
    method: str = "wilson",
    judge_derived: bool = False,
) -> GateReport:
    """Evaluate the gate and return a structured :class:`~evalgate.models.GateReport`.

    Set ``judge_derived=True`` when ``passes`` came from the LLM judge's verdicts (not
    deterministic code checks); the report then includes a bias-corrected pass-rate that
    accounts for the judge's measured TPR/FPR.
    """
    pr = _pass_rate_report(passes, confidence, method, calibration, judge_derived)

    delta: VersionDelta | None = None
    if baseline_passes is not None and len(baseline_passes) == len(passes) and passes:
        m = stats.mcnemar_from_pairs(list(baseline_passes), list(passes))
        delta = VersionDelta(b=m.b, c=m.c, p_value=m.p_value, verdict=m.verdict)

    reasons: list[str] = []
    if not passes:
        reasons.append("no eval results to gate on")
    elif pr.lower < min_pass_rate:
        reasons.append(
            f"pass-rate CI lower bound {pr.lower:.3f} < min_pass_rate {min_pass_rate:.3f}"
        )
    if calibration is not None and calibration.drifted:
        reasons.append(
            f"judge drift: kappa {calibration.kappa:.3f} < min {calibration.min_kappa:.3f} "
            "(calibration failed)"
        )
    if delta is not None and delta.verdict == "regressed":
        reasons.append(
            f"significant regression vs baseline: {delta.b} newly failing "
            f"(McNemar p={delta.p_value:.3f})"
        )

    return GateReport(
        passed=not reasons,
        reasons=reasons,
        pass_rate=pr,
        min_pass_rate=min_pass_rate,
        calibration=calibration,
        delta=delta,
        by_evaluator=list(by_evaluator) if by_evaluator is not None else [],
    )


def _f(x: float) -> str:
    """Format a float, showing 'n/a' for NaN so a degenerate readout stays legible."""
    return "n/a" if x != x else f"{x:.3f}"


def render_gate(report: GateReport) -> str:
    """Render a GateReport as a terminal readout (box-drawing; safe for CI logs).

    The box is sized to its widest line so borders always align, whatever the numbers.
    """
    pr = report.pass_rate
    conf = int(round(pr.confidence * 100))
    verdict = "PASS" if report.passed else "FAIL"

    body: list[str] = [
        f"result       : {verdict}  (exit {0 if report.passed else 1})",
        f"pass-rate    : {pr.point:.3f}  {conf}% CI [{pr.lower:.3f}, {pr.upper:.3f}]  "
        f"({pr.method}, {pr.passed}/{pr.n})",
        f"min pass-rate: {report.min_pass_rate:.3f}",
    ]
    if pr.corrected is not None:
        body.append(f"bias-corrected pass-rate: {pr.corrected:.3f}")
    c = report.calibration
    if c is not None:
        body.append(f"judge kappa  : {_f(c.kappa)} ({c.band})  min {c.min_kappa:.2f}  "
                    f"drifted={c.drifted}")
        body.append(f"judge agree  : AC1 {_f(c.ac1)}  raw {_f(c.raw_agreement)}  "
                    f"TPR {_f(c.tpr)}  TNR {_f(c.tnr)}  prevalence {_f(c.prevalence)}")
        if c.paradox_flag:
            body.append("  ! kappa/AC1 diverge - anchor set likely imbalanced")
    if report.delta is not None:
        d = report.delta
        body.append(f"vs baseline  : +{d.c} fixed / -{d.b} regressed  "
                    f"({d.verdict}, McNemar p={d.p_value:.3f})")
    if report.by_evaluator:
        body.append("by evaluator :")
        for s in report.by_evaluator:
            body.append(f"  {s.evaluator:<24} {s.passed}/{s.total}  {s.pass_rate:.3f}")

    footer: list[str] = ["RESULT: PASS"] if report.passed else (
        ["RESULT: FAIL"] + [f"  - {r}" for r in report.reasons]
    )

    inner = max(len(s) for s in ["EVALGATE", *body, *footer]) + 1
    top = "┌" + "─" * (inner + 1) + "┐"
    mid = "├" + "─" * (inner + 1) + "┤"
    bot = "└" + "─" * (inner + 1) + "┘"

    def row(s: str) -> str:
        return "│ " + s.ljust(inner) + "│"

    out = [top, row("EVALGATE"), mid, *[row(s) for s in body],
           mid, *[row(s) for s in footer], bot]
    return "\n".join(out)


def ci_gate(
    results: Sequence[EvalResult],
    min_pass_rate: float,
    calibration: CalibrationReport | None,
    baseline_results: Sequence[EvalResult] | None = None,
    alpha: float = 0.05,
) -> int:
    """Evaluate the gate over a flat list of EvalResults and return an exit code (0/1).

    Prints the readout as a side effect. Each EvalResult is treated as one pass/fail sample;
    a per-evaluator breakdown is included. For trace-level gating, build per-trace booleans and
    call :func:`evaluate_gate` directly.
    """
    passes = [r.passed for r in results]
    baseline = [r.passed for r in baseline_results] if baseline_results is not None else None
    report = evaluate_gate(
        passes,
        min_pass_rate=min_pass_rate,
        calibration=calibration,
        baseline_passes=baseline,
        by_evaluator=_evaluator_stats(results),
        confidence=1 - alpha,
    )
    print(render_gate(report))
    return 0 if report.passed else 1


def main() -> int:
    """``make gate`` entrypoint. The real CLI lives in :mod:`evalgate.cli`."""
    print("evalgate.gate: run the CLI (`evalgate gate`) or examples/eval_flint_parser.py for "
          "the end-to-end pattern (load traces -> evaluators -> calibrate -> gate).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
