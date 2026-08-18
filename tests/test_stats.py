"""Tests for the statistical primitives that keep the gate honest on tiny samples."""

import math

import pytest

from evalgate.stats import (
    gwet_ac1,
    kappa_band,
    kappa_paradox_flag,
    mcnemar_exact,
    mcnemar_from_pairs,
    prevalence,
    raw_agreement,
    wilson_ci,
    z_for_confidence,
)

# --- Wilson score interval ------------------------------------------------------------


def test_wilson_point_is_raw_proportion():
    point, lo, hi = wilson_ci(47, 50)
    assert math.isclose(point, 0.94)
    assert lo <= point <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_wilson_matches_known_value():
    # Classic textbook case: 1/10 successes, 95% Wilson CI ~= [0.0179, 0.4042].
    _, lo, hi = wilson_ci(1, 10)
    assert math.isclose(lo, 0.0179, abs_tol=1e-3)
    assert math.isclose(hi, 0.4042, abs_tol=1e-3)


def test_wilson_all_pass_upper_is_one_lower_below_one():
    # Unlike Wald, Wilson gives a sensible interval at the boundary p=1.
    point, lo, hi = wilson_ci(20, 20)
    assert point == 1.0
    assert math.isclose(hi, 1.0)
    assert lo < 1.0  # the gate still can't be *certain* from 20 samples


def test_wilson_zero_pass_lower_is_zero():
    point, lo, hi = wilson_ci(0, 20)
    assert point == 0.0
    assert lo == 0.0
    assert hi > 0.0


def test_wilson_empty_sample_is_all_zero():
    assert wilson_ci(0, 0) == (0.0, 0.0, 0.0)


def test_wilson_rejects_bad_counts():
    with pytest.raises(ValueError):
        wilson_ci(11, 10)


def test_higher_confidence_widens_interval():
    _, lo90, hi90 = wilson_ci(8, 10, confidence=0.90)
    _, lo99, hi99 = wilson_ci(8, 10, confidence=0.99)
    assert (hi99 - lo99) > (hi90 - lo90)


def test_z_for_confidence_table_and_approx():
    assert math.isclose(z_for_confidence(0.95), 1.959964, abs_tol=1e-5)
    # A non-tabulated level falls back to the inverse-normal approx.
    assert math.isclose(z_for_confidence(0.80), 1.281552, abs_tol=1e-4)


# --- McNemar paired test --------------------------------------------------------------


def test_mcnemar_all_regressions_is_significant():
    p = mcnemar_exact(b=10, c=0)
    assert p < 0.05


def test_mcnemar_balanced_discordant_is_inconclusive():
    p = mcnemar_exact(b=5, c=5)
    assert math.isclose(p, 1.0)


def test_mcnemar_no_discordant_pairs_is_one():
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_mid_p_less_conservative_than_exact():
    exact = mcnemar_exact(9, 1, mid_p=False)
    midp = mcnemar_exact(9, 1, mid_p=True)
    assert midp < exact  # mid-p shrinks the p-value


def test_mcnemar_from_pairs_detects_regression():
    # v2 fails everything v1 passed -> a clear regression.
    v1 = [True] * 8 + [False] * 2
    v2 = [False] * 8 + [False] * 2
    res = mcnemar_from_pairs(v1, v2)
    assert res.b == 8 and res.c == 0
    assert res.verdict == "regressed"


def test_mcnemar_from_pairs_detects_improvement():
    v1 = [False] * 8 + [True] * 2
    v2 = [True] * 8 + [True] * 2
    res = mcnemar_from_pairs(v1, v2)
    assert res.c == 8 and res.b == 0
    assert res.verdict == "improved"


def test_mcnemar_from_pairs_length_mismatch_raises():
    with pytest.raises(ValueError):
        mcnemar_from_pairs([True, False], [True])


# --- Agreement coefficients -----------------------------------------------------------


def test_kappa_band_landis_koch():
    assert kappa_band(0.85) == "almost perfect"
    assert kappa_band(0.7) == "substantial"
    assert kappa_band(0.5) == "moderate"
    assert kappa_band(-0.1) == "worse than chance"
    assert kappa_band(float("nan")) == "undefined"


def test_raw_agreement_and_prevalence():
    a = [True, True, False, False]
    b = [True, False, False, False]
    assert raw_agreement(a, b) == 0.75
    assert prevalence(a) == 0.5


def test_gwet_ac1_perfect_agreement_is_one():
    labels = [True, False, True, False]
    assert math.isclose(gwet_ac1(labels, labels), 1.0)


def test_gwet_ac1_defined_when_kappa_would_be_nan():
    # Both raters mark everything pass: Cohen's kappa is undefined, but AC1 is a clean 1.0
    # because it anchors chance-agreement to prevalence. This is the paradox AC1 exists to fix.
    allpass = [True, True, True, True]
    assert math.isclose(gwet_ac1(allpass, allpass), 1.0)


def test_kappa_paradox_flag_fires_on_divergence():
    # The documented paradox case: high raw agreement, kappa collapses (0.43) while AC1 stays high.
    assert kappa_paradox_flag(kappa=0.43, ac1=0.86) is True
    assert kappa_paradox_flag(kappa=0.80, ac1=0.82) is False
    assert kappa_paradox_flag(kappa=float("nan"), ac1=0.9) is False
