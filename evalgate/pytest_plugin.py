"""pytest integration: assert the gate inside an existing test suite.

Registered as a pytest plugin via the ``pytest11`` entry point (see pyproject), so ``assert_gate``
is importable and eval tests fail the build like any other assertion:

    from evalgate.pytest_plugin import assert_gate

    def test_agent_quality(agent_passes, calibration):
        assert_gate(agent_passes, min_pass_rate=0.9, calibration=calibration)
"""

from __future__ import annotations

from collections.abc import Sequence

from .gate import evaluate_gate, render_gate
from .models import CalibrationReport


def assert_gate(
    passes: Sequence[bool],
    min_pass_rate: float = 0.9,
    calibration: CalibrationReport | None = None,
    baseline_passes: Sequence[bool] | None = None,
    confidence: float = 0.95,
    judge_derived: bool = False,
) -> None:
    """Fail the test (with the full gate readout) unless the gate passes."""
    report = evaluate_gate(
        passes,
        min_pass_rate=min_pass_rate,
        calibration=calibration,
        baseline_passes=baseline_passes,
        confidence=confidence,
        judge_derived=judge_derived,
    )
    if not report.passed:
        raise AssertionError("EvalGate blocked the build:\n" + render_gate(report))


def pytest_configure(config) -> None:  # noqa: ARG001 - pytest hook signature
    """Marker registration so ``@pytest.mark.evalgate`` doesn't warn."""
    config.addinivalue_line("markers", "evalgate: mark a test as an EvalGate quality gate")
