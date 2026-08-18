"""Statistical primitives for gating on tiny, non-deterministic eval samples.

Everything here is real math with no hidden LLM calls. A CI gate must not flip on a two-sample
wobble, so EvalGate is deliberate about the statistics it reports:

  * **Wilson score interval** for the pass-rate. It inverts the score test instead of the Wald
    normal approximation, so it stays well-behaved at small ``n`` and near 0 % / 100 % — exactly
    the regime an eval gate lives in. We gate on the interval's *lower bound*, never the point
    estimate, so a lucky small sample can't sneak a regression past the gate.
  * **McNemar's exact paired test** to compare two agent versions on the *same* eval set. The
    two runs are paired (same inputs), so the correct question is only about the discordant
    pairs — v1-pass/v2-fail vs v1-fail/v2-pass — not two independent proportions.
  * **Cohen's kappa with Landis–Koch bands**, cross-checked against **Gwet's AC1**. Under class
    imbalance kappa collapses even when raw agreement is high (the "kappa prevalence paradox"),
    so we compute AC1 alongside it and flag when the two diverge.

References are cited in the module docstrings / README, not hard-coded here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Two-sided z critical values, keyed by confidence level, so callers don't pull in scipy for a
# constant. 0.95 -> 1.959964 covers the overwhelmingly common case.
_Z_BY_CONFIDENCE = {0.90: 1.6448536269, 0.95: 1.9599639845, 0.99: 2.5758293035}


def z_for_confidence(confidence: float = 0.95) -> float:
    """Return the two-sided normal critical value for a confidence level.

    Uses an exact table for the common levels and the rational Acklam inverse-normal
    approximation otherwise (max abs error < 1.2e-9 over the central region).
    """
    if confidence in _Z_BY_CONFIDENCE:
        return _Z_BY_CONFIDENCE[confidence]
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return _inv_norm_cdf(1.0 - (1.0 - confidence) / 2.0)


def _inv_norm_cdf(p: float) -> float:
    """Acklam's rational approximation to the standard-normal quantile function."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        successes: number of passing traces.
        n: total number of traces.
        confidence: two-sided confidence level (default 0.95).

    Returns ``(point_estimate, lower, upper)``, each clamped to ``[0, 1]``. Returns all-zero for
    ``n == 0`` (no evidence -> the gate should treat a zero-sample run as failing anyway).
    """
    if n < 0 or successes < 0 or successes > n:
        raise ValueError(f"invalid counts: successes={successes}, n={n}")
    if n == 0:
        return 0.0, 0.0, 0.0
    z = z_for_confidence(confidence)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    # Snap the analytically-exact boundaries (Wilson lower is exactly 0 at x=0, upper exactly 1
    # at x=n) so floating-point residue doesn't leave 1e-17 dust in a gate readout.
    if successes == 0:
        lower = 0.0
    if successes == n:
        upper = 1.0
    return p, lower, upper


# --------------------------------------------------------------------------------------
# McNemar's paired test — "is agent v2 actually different from v1 on the same eval set?"
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class McNemarResult:
    """Outcome of a paired v1-vs-v2 comparison on the same eval set."""

    b: int  # v1 passed, v2 failed (regressions)
    c: int  # v1 failed, v2 passed (fixes)
    n_discordant: int
    p_value: float
    verdict: str  # "improved" | "regressed" | "inconclusive"


def mcnemar_exact(b: int, c: int, mid_p: bool = True) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts.

    Under H0 the discordant pairs split 50/50, so ``b ~ Binomial(b + c, 0.5)``. We sum the exact
    binomial tail and double it (two-sided), capping at 1.0. ``mid_p=True`` applies the mid-p
    correction (subtract half the point mass), which is less conservative and preferred for the
    small samples an eval gate sees. Returns 1.0 when there are no discordant pairs.
    """
    if b < 0 or c < 0:
        raise ValueError(f"counts must be non-negative: b={b}, c={c}")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    point = math.comb(n, k) / (2.0**n)
    if mid_p:
        tail -= 0.5 * point
    return min(1.0, 2.0 * tail)


def mcnemar_from_pairs(
    v1: Sequence[bool], v2: Sequence[bool], alpha: float = 0.05, mid_p: bool = True
) -> McNemarResult:
    """Compare two agent versions' per-trace pass/fail over the *same* ordered eval set."""
    if len(v1) != len(v2):
        raise ValueError(f"paired inputs must match length: {len(v1)} vs {len(v2)}")
    b = sum(1 for a, d in zip(v1, v2, strict=True) if a and not d)
    c = sum(1 for a, d in zip(v1, v2, strict=True) if not a and d)
    p_value = mcnemar_exact(b, c, mid_p=mid_p)
    if p_value >= alpha:
        verdict = "inconclusive"
    else:
        verdict = "improved" if c > b else "regressed"
    return McNemarResult(b=b, c=c, n_discordant=b + c, p_value=p_value, verdict=verdict)


# --------------------------------------------------------------------------------------
# Agreement coefficients for judge calibration.
# --------------------------------------------------------------------------------------

# Landis & Koch (1977) interpretation bands for chance-corrected agreement coefficients.
_LANDIS_KOCH = (
    (0.0, "poor"),
    (0.20, "slight"),
    (0.40, "fair"),
    (0.60, "moderate"),
    (0.80, "substantial"),
    (1.01, "almost perfect"),
)


def kappa_band(kappa: float) -> str:
    """Landis–Koch qualitative band for a kappa / AC1 value (``nan`` -> 'undefined')."""
    if math.isnan(kappa):
        return "undefined"
    if kappa < 0:
        return "worse than chance"
    for upper, label in _LANDIS_KOCH:
        if kappa < upper:
            return label
    return "almost perfect"


def raw_agreement(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Fraction of items on which the two raters agree."""
    if len(a) != len(b):
        raise ValueError("rater sequences must match length")
    if not a:
        return float("nan")
    return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)


def prevalence(labels: Sequence[bool]) -> float:
    """Fraction of positive ('pass') labels — the base rate that drives the kappa paradox."""
    if not labels:
        return float("nan")
    return sum(1 for x in labels if x) / len(labels)


def gwet_ac1(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Gwet's AC1 for two raters on binary labels.

    AC1 anchors its chance-agreement term to the observed prevalence, so it does not collapse
    under class imbalance the way Cohen's kappa does. Returns ``nan`` for an empty input or when
    the raters agree perfectly *and* the categories are degenerate (``1 - p_e == 0``).
    """
    if len(a) != len(b):
        raise ValueError("rater sequences must match length")
    n = len(a)
    if n == 0:
        return float("nan")
    p_a = raw_agreement(a, b)
    # Mean proportion assigned to the positive category across both raters.
    pi_pos = (sum(1 for x in a if x) + sum(1 for x in b if x)) / (2 * n)
    p_e = 2 * pi_pos * (1 - pi_pos)  # q=2: (1/(q-1)) * sum_k pi_k(1-pi_k)
    if math.isclose(p_e, 1.0):
        return float("nan")
    return (p_a - p_e) / (1 - p_e)


def kappa_paradox_flag(kappa: float, ac1: float, gap: float = 0.2) -> bool:
    """True when kappa and AC1 diverge enough to suspect the prevalence paradox.

    A large gap (default 0.2) between a chance-corrected coefficient and the prevalence-robust
    AC1 is the signature of an imbalanced anchor set understating a good judge. Callers should
    surface this as a warning ("your anchor set is imbalanced"), not as a hard gate failure.
    """
    if math.isnan(kappa) or math.isnan(ac1):
        return False
    return abs(ac1 - kappa) >= gap
