"""Reference integration: evaluating **Flint**, a natural-language -> DAG parser.

Runs the whole EvalGate pipeline against one agent, end to end and **offline** (no API key
required — set ``GROQ_API_KEY`` to swap in a real LLM judge for calibration):

    generate corpus -> error analysis (taxonomy) -> code checks -> calibrate judge -> gate

Run:  python -m examples.eval_flint_parser
"""

from __future__ import annotations

from evalgate.analysis import build_taxonomy
from evalgate.calibration import calibrate
from evalgate.config import get_settings
from evalgate.gate import evaluate_gate, render_gate
from evalgate.models import EvalResult
from evalgate.reference import (
    FLINT_RUBRIC,
    dag_code_checks,
    demo_judge_labels,
    generate_anchor_set,
    generate_flint_traces,
    trace_passes,
)


def main() -> int:
    settings = get_settings()
    traces = generate_flint_traces(n=200, n_fail=10, seed=0)

    # 1. Deterministic DAG code checks -> per-trace verdict + per-evaluator breakdown.
    all_results: list[EvalResult] = []
    passes: list[bool] = []
    for t in traces:
        results = dag_code_checks(t)
        all_results.extend(results)
        passes.append(all(r.passed for r in results))

    from evalgate.gate import _evaluator_stats

    by_evaluator = _evaluator_stats(all_results)

    # 2. Error analysis FIRST: cluster the failing traces into a taxonomy.
    failing = [t for t in traces if not trace_passes(t)]
    taxonomy = build_taxonomy([t.text() for t in failing], seed=0)
    print("failure taxonomy (error-analysis-first)")
    print("-" * 64)
    for i, c in enumerate(taxonomy, 1):
        print(f"  C{i}  n={c.size:<3} {c.label}")
    print()

    # 3. Calibrate the judge against a human anchor set (drift detection).
    anchors = generate_anchor_set(n=24, seed=1)
    human_labels = [a.human_label for a in anchors]
    if settings.groq_api_key:
        from evalgate.evaluators import LLMJudge

        judge = LLMJudge(settings=settings)
        judge_labels = [judge.judge(a.input, a.output, FLINT_RUBRIC).passed for a in anchors]
    else:
        # Offline stand-in so the pipeline runs with no key; see reference.demo_judge_labels.
        judge_labels = demo_judge_labels(anchors)
    calibration = calibrate(judge_labels, human_labels, min_kappa=settings.min_judge_kappa)

    # 4. A prior agent version (baseline) for a paired McNemar delta. Here: three traces that
    #    the previous version failed and this version fixed.
    baseline_passes = list(passes)
    fixed = 0
    for i, ok in enumerate(passes):
        if ok and fixed < 3:
            baseline_passes[i] = False
            fixed += 1

    # 5. Gate.
    report = evaluate_gate(
        passes,
        min_pass_rate=settings.min_pass_rate,
        calibration=calibration,
        baseline_passes=baseline_passes,
        by_evaluator=by_evaluator,
        confidence=0.95,
    )
    print(render_gate(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
