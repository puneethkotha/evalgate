"""Build the JSON payload the dashboard renders.

One function assembles everything the instrument-panel UI needs — the current gate report, the
failure taxonomy, and a run history for the judge-drift-over-time timeline — into a plain dict
the static dashboard fetches as ``report.json``. Deterministic (no timestamps/RNG at call time)
so the checked-in demo payload is stable.
"""

from __future__ import annotations

from typing import Any

from .reference import run_demo

# A short, illustrative run history for the drift timeline. Run #243 shows a judge-drift episode
# (kappa collapses below the 0.70 threshold) that later recovers — the exact failure EvalGate
# exists to catch. The final run matches the live `run_demo` output.
_HISTORY = [
    {"run": 237, "pass_rate": 0.93, "kappa": 0.82},
    {"run": 238, "pass_rate": 0.94, "kappa": 0.80},
    {"run": 239, "pass_rate": 0.92, "kappa": 0.83},
    {"run": 240, "pass_rate": 0.95, "kappa": 0.79},
    {"run": 241, "pass_rate": 0.94, "kappa": 0.81},
    {"run": 242, "pass_rate": 0.93, "kappa": 0.84},
    {"run": 243, "pass_rate": 0.94, "kappa": 0.55},
    {"run": 244, "pass_rate": 0.95, "kappa": 0.58},
    {"run": 245, "pass_rate": 0.94, "kappa": 0.80},
    {"run": 246, "pass_rate": 0.95, "kappa": 0.82},
    {"run": 247, "pass_rate": 0.95, "kappa": 0.83},
]


def build_dashboard_report(seed: int = 0, min_kappa: float = 0.70,
                           current_run: int = 248) -> dict[str, Any]:
    """Assemble the full dashboard payload from a fresh reference run."""
    taxonomy, gate = run_demo(seed=seed)
    history = [
        {**h, "drifted": h["kappa"] < min_kappa} for h in _HISTORY
    ]
    history.append({
        "run": current_run,
        "pass_rate": round(gate.pass_rate.point, 3),
        "kappa": round(gate.calibration.kappa, 3) if gate.calibration else None,
        "drifted": bool(gate.calibration.drifted) if gate.calibration else False,
    })
    return {
        "date": "2026-08-18",
        "agent": "nl→dag parser (llama-3.3-70b)",
        "run": current_run,
        "min_kappa": min_kappa,
        "gate": gate.model_dump(),
        "taxonomy": [c.model_dump() for c in taxonomy],
        "history": history,
    }
