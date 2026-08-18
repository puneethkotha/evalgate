"""Judge calibration — the differentiator.

We never trust the LLM-judge blindly. Every run, we re-score a frozen, human-labeled **anchor
set** and measure agreement with a *bundle* of statistics — Cohen's kappa (chance-corrected),
Gwet's AC1 (prevalence-robust), raw agreement, and TPR/TNR. If kappa falls below
``min_judge_kappa`` the judge is considered *drifted* and the run is blocked (see
:mod:`evalgate.gate`) — so a silently-degrading judge can never green-light a regressing agent.

Two moves competitors don't make (verified against Langfuse, LangSmith, DeepEval, Braintrust,
Ragas, Arize, Opik, Weave, Patronus as of 2026):
  * we report a *chance-corrected* coefficient (most tools show raw % agreement), and
  * we cross-check kappa against AC1 to catch the prevalence paradox, then bias-correct the
    reported pass-rate using the judge's measured error rates.

All math is real (``sklearn.metrics.cohen_kappa_score`` + :mod:`evalgate.stats`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from . import stats
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
    """Compute the judge-vs-human agreement bundle over the anchor set.

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
    ac1 = stats.gwet_ac1(hl, jl)
    raw = stats.raw_agreement(hl, jl)
    prev = stats.prevalence(hl)

    # Confusion counts with the human label as ground truth (positive == "passes").
    tp = sum(1 for h, j in zip(hl, jl, strict=True) if h and j)
    fn = sum(1 for h, j in zip(hl, jl, strict=True) if h and not j)
    tn = sum(1 for h, j in zip(hl, jl, strict=True) if not h and not j)
    fp = sum(1 for h, j in zip(hl, jl, strict=True) if not h and j)

    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    drifted = math.isnan(kappa) or kappa < min_kappa

    return CalibrationReport(
        kappa=kappa,
        ac1=ac1,
        raw_agreement=raw,
        tpr=tpr,
        tnr=tnr,
        fpr=fpr,
        prevalence=prev,
        band=stats.kappa_band(kappa),
        paradox_flag=stats.kappa_paradox_flag(kappa, ac1),
        n=n,
        min_kappa=min_kappa,
        drifted=drifted,
    )


def calibrate_judge(judge, anchors: Sequence[AnchorExample], rubric: str,
                    min_kappa: float | None = None) -> CalibrationReport:
    """Run ``judge`` over a human-labeled anchor set and calibrate against the human labels.

    This is the wiring that makes the differentiator real: the same judge used to score the
    agent is re-scored against frozen human ground truth every run. ``judge`` must expose
    ``judge(input_text, output_text, rubric) -> JudgeResult`` (see
    :class:`evalgate.evaluators.LLMJudge`).
    """
    if not anchors:
        raise ValueError("empty anchor set: nothing to calibrate against")
    judge_labels = [judge.judge(a.input, a.output, rubric).passed for a in anchors]
    human_labels = [a.human_label for a in anchors]
    return calibrate(judge_labels, human_labels, min_kappa=min_kappa)


def corrected_pass_rate(observed: float, tpr: float, fpr: float) -> float | None:
    """De-bias an observed pass-rate using the judge's measured error rates.

    A judge with known TPR/FPR systematically mis-estimates the true pass-rate. The standard
    prevalence correction inverts that: ``true = (observed - fpr) / (tpr - fpr)``. Returns
    ``None`` when the judge is uninformative (``tpr == fpr``) or the rates are undefined.
    """
    if any(math.isnan(x) for x in (observed, tpr, fpr)):
        return None
    denom = tpr - fpr
    if abs(denom) < 1e-9:
        return None
    return min(1.0, max(0.0, (observed - fpr) / denom))
