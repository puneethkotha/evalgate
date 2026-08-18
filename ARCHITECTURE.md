# Architecture

A bird's-eye map of EvalGate for contributors. For *what* it does and why, see the
[README](README.md); this doc is about *how the code is laid out* and the invariants that keep it
honest.

## The one-sentence model

Traces come in → error analysis clusters the failures into a taxonomy → evaluators (code checks +
a calibrated LLM judge) score them → the gate turns pass-rate **and** judge-vs-human agreement into
a single exit code. Everything is a plain data contract (`evalgate.models`) that the CLI, the
GitHub Action, the pytest plugin, and the dashboard all render from.

```
OTel spans ─▶ otel ─▶ Trace ─┬─▶ analysis ─▶ FailureCluster[]  (taxonomy)
                             ├─▶ evaluators ─▶ EvalResult / JudgeResult
anchors.jsonl ─▶ calibration ─▶ CalibrationReport (κ + AC1 + drift)
                                         │
                    stats (Wilson, McNemar, AC1) │
                                         ▼
                                  gate ─▶ GateReport ─▶ exit 0 / 1
```

## Codemap

| Module | Responsibility |
|---|---|
| `evalgate/models.py` | The shared Pydantic contract: `Trace`/`Span`, `EvalResult`, `CalibrationReport`, `GateReport`, etc. Everything else speaks these. |
| `evalgate/stats.py` | Pure statistics, no I/O: Wilson score interval, McNemar's exact test, Gwet's AC1, Landis–Koch bands, kappa-paradox flag. |
| `evalgate/evaluators.py` | `CodeChecks` (deterministic) + `LLMJudge` (CoT→binary verdict→critique, JSON, retries, order-swapped pairwise). Network lives only in `LLMJudge._chat`. |
| `evalgate/calibration.py` | Scores the judge against the human anchor set → the agreement bundle; `calibrate_judge` wires the judge in; drift = κ below threshold. |
| `evalgate/analysis.py` | Error-analysis workbench: pluggable embedding encoder (default: offline TF-IDF+SVD) → KMeans → auto-labeled `build_taxonomy`. |
| `evalgate/gate.py` | Composes results + calibration into a `GateReport` (`evaluate_gate`, pure) and renders the terminal readout (`render_gate`). |
| `evalgate/otel.py` | Version-tolerant OpenTelemetry-GenAI span → `Trace` adapter (isolates semconv churn). |
| `evalgate/reference.py` | The offline NL→DAG reference agent + corpus + `run_demo` (shared by `evalgate demo` and the example). |
| `evalgate/report.py` | Assembles the dashboard `report.json` payload (gate + taxonomy + run history). |
| `evalgate/cli.py` | `evalgate` command (`demo`/`gate`/`analyze`/`calibrate`/`report`). Stdlib argparse only. |
| `evalgate/api.py` | FastAPI: serves the dashboard + a live `/report.json`. |
| `evalgate/pytest_plugin.py` | `assert_gate(...)` for eval-as-unit-tests. |
| `dashboard/` | Static instrument-panel UI (vanilla JS + SVG). Deploys as-is; reads `report.json`. |
| `action.yml` | Composite GitHub Action: run the gate, comment on the PR, fail the check. |

## Invariants

1. **Importing the package never touches the network or a DB.** The judge client and DB engine
   are built lazily; `evalgate.config.Settings` has a safe default for every field.
2. **`stats` and `evaluate_gate` are pure** (no I/O), so they are trivially testable and the gate
   decision is reproducible.
3. **The gate reads the CI *lower bound*, never the point estimate.** A lucky small sample must
   not sneak past.
4. **The judge is never trusted blindly.** Every run re-scores the frozen anchor set; if κ drops
   below `min_judge_kappa`, the run is blocked as *judge drift*, independent of the pass-rate.
5. **Degenerate calibration fails closed.** A single-class ("rubber-stamp") judge yields κ = NaN
   and is treated as drifted, regardless of the sklearn version's behavior.
6. **Binary verdicts only.** The judge returns pass/fail + a written critique — never a 1–5 score.
7. **$0 by default.** The embedding encoder needs no model download or network; the judge is
   optional and provider-agnostic.

## Testing

`python -m pytest` (94 tests). Network is always mocked (`LLMJudge` takes an injected client);
the reference corpus and stats give deterministic, offline coverage. `evalgate demo` is run in CI
so the gate gates EvalGate itself.
