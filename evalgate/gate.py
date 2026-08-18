"""CI gate: turn a batch of eval results into a single pass/fail exit code.

Two independent ways to fail:
  1. **Agent regression** — the *lower bound* of a bootstrap confidence interval on the
     pass-rate falls below ``min_pass_rate``. Using the CI lower bound (not the point estimate)
     means a small, lucky sample can't sneak past the gate.
  2. **Judge drift** — calibration says the judge no longer agrees with humans. A drifted judge
     invalidates every judgment this run, so we fail regardless of the pass-rate.

Bootstrap resampling is real (numpy). Run via ``python -m evalgate.gate``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np

from .models import CalibrationReport, EvalResult


def bootstrap_ci(
    passes: Sequence[float],
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a mean pass-rate.

    Returns ``(point_estimate, lower, upper)`` at confidence ``1 - alpha``.
    """
    arr = np.asarray(passes, dtype=float)
    n = arr.size
    if n == 0:
        return 0.0, 0.0, 0.0
    point = float(arr.mean())
    rng = np.random.default_rng(seed)
    # Resample indices n_boot times and take the mean of each resample.
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    lower = float(np.quantile(boot_means, alpha / 2))
    upper = float(np.quantile(boot_means, 1 - alpha / 2))
    return point, lower, upper


def ci_gate(
    results: Sequence[EvalResult],
    min_pass_rate: float,
    calibration: CalibrationReport | None,
    alpha: float = 0.05,
) -> int:
    """Evaluate the gate and return a shell exit code (0 = pass, 1 = fail).

    Prints a human-readable summary as a side effect.
    """
    passes = [1.0 if r.passed else 0.0 for r in results]
    point, lower, upper = bootstrap_ci(passes, alpha=alpha)
    conf = int(round((1 - alpha) * 100))

    reasons: list[str] = []
    if not results:
        reasons.append("no eval results to gate on")
    if lower < min_pass_rate:
        reasons.append(
            f"pass-rate CI lower bound {lower:.3f} < min_pass_rate {min_pass_rate:.3f}"
        )
    if calibration is not None and calibration.drifted:
        reasons.append(
            f"judge drift: kappa {calibration.kappa:.3f} below threshold (calibration failed)"
        )

    print("=" * 64)
    print("EVALGATE CI GATE")
    print("-" * 64)
    print(f"traces evaluated : {len(results)}")
    print(f"pass-rate        : {point:.3f}  ({conf}% CI [{lower:.3f}, {upper:.3f}])")
    print(f"min pass-rate    : {min_pass_rate:.3f}")
    if calibration is not None:
        print(
            f"judge calibration: kappa={calibration.kappa:.3f} "
            f"tpr={calibration.tpr:.3f} tnr={calibration.tnr:.3f} "
            f"n={calibration.n} drifted={calibration.drifted}"
        )
    else:
        print("judge calibration: (not run)")
    print("-" * 64)

    if reasons:
        print("RESULT: FAIL")
        for r in reasons:
            print(f"  - {r}")
        print("=" * 64)
        return 1

    print("RESULT: PASS")
    print("=" * 64)
    return 0


def main() -> int:
    """``make gate`` entrypoint.

    TODO: wire to real stored traces + a live judge + the anchor set. For now, exits cleanly
    with an explanatory message so ``python -m evalgate.gate`` never crashes in a fresh checkout.
    """
    print("evalgate.gate: no evaluation wired yet — see examples/eval_flint_parser.py for the "
          "end-to-end pattern (load traces -> run evaluators -> calibrate -> ci_gate).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
