"""Tests for the pytest plugin's assert_gate helper."""

import pytest

from evalgate.calibration import calibrate
from evalgate.pytest_plugin import assert_gate


def test_assert_gate_passes_silently():
    assert_gate([True] * 100, min_pass_rate=0.9)  # should not raise


def test_assert_gate_raises_with_readout():
    with pytest.raises(AssertionError) as exc:
        assert_gate([True] * 9 + [False], min_pass_rate=0.9)
    assert "EvalGate blocked the build" in str(exc.value)
    assert "RESULT: FAIL" in str(exc.value)


def test_assert_gate_blocks_on_judge_drift():
    human = [True, True, True, False, False, False]
    drifted = [False, False, False, True, True, True]
    calib = calibrate(drifted, human, min_kappa=0.7)
    with pytest.raises(AssertionError):
        assert_gate([True] * 100, min_pass_rate=0.9, calibration=calib)
