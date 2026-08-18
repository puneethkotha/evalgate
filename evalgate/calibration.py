"""Judge calibration — the differentiator.

We never trust the LLM-judge blindly. Every run, we re-score a frozen, human-labeled **anchor
set** and measure agreement with **Cohen's kappa** (agreement corrected for chance) plus
true-positive / true-negative rates. If kappa falls below ``min_judge_kappa`` the judge is
considered *drifted* and the run is blocked (see :mod:`evalgate.gate`) — so a silently-degrading
judge can never green-light a regressing agent.

All math here is real (uses ``sklearn.metrics.cohen_kappa_score``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from .config import get_settings
from .models import AnchorExample, CalibrationReport


def load_anchor_set(path: str | Path) -> list[AnchorExample]:
    """Load a JSONL anchor set: one ``{input, output, human_label}`` object per line."""
    examples: list[AnchorExample] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(AnchorExample(**json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"bad anchor at {path}:{lineno}: {exc}") from exc
    return examples


def calibrate(
    judge_labels: Sequence[bool],
    human_labels: Sequence[bool],
    min_kappa: float | None = None,
) -> CalibrationReport:
    """Compute judge-vs-human agreement over the anchor set.

    Args:
        judge_labels: the judge's pass/fail verdict per anchor.
        human_labels: the ground-truth human pass/fail label per anchor.
        min_kappa: drift threshold; defaults to ``Settings.min_judge_kappa``.

    Returns a :class:`CalibrationReport`. ``drifted`` is True when kappa is below the threshold
    or undefined (degenerate: judge or humans gave a single class, so agreement can't be
    confirmed — fail closed).
    """
    if len(judge_labels) != len(human_labels):
        raise ValueError(
            f"label count mismatch: {len(judge_labels)} judge vs {len(human_labels)} human"
        )
    n = len(human_labels)
    if n == 0:
        raise ValueError("empty anchor set: nothing to calibrate against")

    if min_kappa is None:
        min_kappa = get_settings().min_judge_kappa

    jl = [bool(x) for x in judge_labels]
    hl = [bool(x) for x in human_labels]

    # Degenerate case: if either rater is single-class (no variance), Cohen's kappa is
    # undefined — chance-corrected agreement can't be estimated. sklearn is inconsistent here
    # (returns 0.0 when only the judge is constant, NaN when both are), so we detect it
    # explicitly and report NaN. A single-class judge is exactly the "rubber-stamp" failure the
    # calibration exists to catch, so we fail closed (drifted=True) regardless of version.
    degenerate = len(set(hl)) < 2 or len(set(jl)) < 2
    kappa = float("nan") if degenerate else float(cohen_kappa_score(hl, jl))

    # Confusion counts with the human label as ground truth (positive == "passes").
    tp = sum(1 for h, j in zip(hl, jl, strict=True) if h and j)
    fn = sum(1 for h, j in zip(hl, jl, strict=True) if h and not j)
    tn = sum(1 for h, j in zip(hl, jl, strict=True) if not h and not j)
    fp = sum(1 for h, j in zip(hl, jl, strict=True) if not h and j)

    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")

    drifted = math.isnan(kappa) or kappa < min_kappa

    return CalibrationReport(kappa=kappa, tpr=tpr, tnr=tnr, n=n, drifted=drifted)
