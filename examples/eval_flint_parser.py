"""Reference integration: evaluating **Flint**, a natural-language -> DAG parser.

Runs the whole EvalGate pipeline against one agent, end to end and **offline** (no API key
required — set ``GROQ_API_KEY`` to swap in a real LLM judge for calibration):

    generate corpus -> error analysis (taxonomy) -> code checks -> calibrate judge -> gate

Run:  python -m examples.eval_flint_parser   (or: evalgate demo)
"""

from __future__ import annotations

from evalgate.gate import render_gate
from evalgate.reference import run_demo


def main() -> int:
    taxonomy, report = run_demo(n=200, n_fail=10, seed=0)

    print("failure taxonomy (error-analysis-first)")
    print("-" * 64)
    for i, c in enumerate(taxonomy, 1):
        print(f"  C{i}  n={c.size:<3} {c.label}")
    print()
    print(render_gate(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
