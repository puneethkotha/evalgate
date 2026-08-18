"""Reference integration: evaluating **Flint**, a natural-language -> DAG parser.

End-to-end wiring of the whole EvalGate pipeline against one agent:

    load traces  ->  CodeChecks (DAG-specific)  +  LLMJudge  ->  calibrate judge  ->  ci_gate

The DAG code checks (is-a-DAG, legal node types, edges resolve) are fully implemented because
they're the domain-specific "crux" for this agent. The trace loading + judge call are shown as
skeleton with ``# TODO`` markers so the owner can point them at real Flint output.

Run:  python -m examples.eval_flint_parser
"""

from __future__ import annotations

import json
from typing import Any

from evalgate.calibration import calibrate
from evalgate.config import get_settings
from evalgate.evaluators import CodeChecks, LLMJudge
from evalgate.gate import ci_gate
from evalgate.models import EvalResult, Span, Trace

# Node types Flint is allowed to emit. Adjust to match the real grammar.
LEGAL_NODE_TYPES = {"source", "transform", "filter", "join", "sink"}


# --------------------------------------------------------------------------------------
# Domain-specific code checks for an NL -> DAG parser (fully implemented).
# A DAG is expected as: {"nodes": [{"id": str, "type": str}, ...],
#                        "edges": [{"from": str, "to": str}, ...]}
# --------------------------------------------------------------------------------------

def _parse_dag(output: Any) -> dict[str, Any] | None:
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(output, dict) and "nodes" in output and "edges" in output:
        return output
    return None


def legal_node_types(dag: dict[str, Any]) -> bool:
    """Every node has a type drawn from the legal grammar."""
    return all(n.get("type") in LEGAL_NODE_TYPES for n in dag.get("nodes", []))


def edges_resolve(dag: dict[str, Any]) -> bool:
    """Every edge endpoint references a declared node id."""
    ids = {n.get("id") for n in dag.get("nodes", [])}
    return all(e.get("from") in ids and e.get("to") in ids for e in dag.get("edges", []))


def is_a_dag(dag: dict[str, Any]) -> bool:
    """Directed + acyclic: a topological sort (Kahn's algorithm) must consume every node."""
    ids = [n.get("id") for n in dag.get("nodes", [])]
    indeg = {i: 0 for i in ids}
    adj: dict[Any, list[Any]] = {i: [] for i in ids}
    for e in dag.get("edges", []):
        src, dst = e.get("from"), e.get("to")
        if src not in indeg or dst not in indeg:
            return False  # dangling edge => not a well-formed DAG
        adj[src].append(dst)
        indeg[dst] += 1
    queue = [i for i, d in indeg.items() if d == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for nxt in adj[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    return visited == len(ids)


def run_dag_code_checks(trace: Trace) -> list[EvalResult]:
    dag = _parse_dag(trace.root_output)
    if dag is None:
        return [
            EvalResult(
                evaluator="dag_parses", passed=False, critique="output is not valid DAG JSON"
            )
        ]
    checks = {
        "is_a_dag": is_a_dag(dag),
        "legal_node_types": legal_node_types(dag),
        "edges_resolve": edges_resolve(dag),
        "latency_budget_2s": CodeChecks.latency_budget(trace, ms=2000),
    }
    return [
        EvalResult(evaluator=name, passed=passed, critique="" if passed else f"{name} failed")
        for name, passed in checks.items()
    ]


# --------------------------------------------------------------------------------------
# Sample traces (stand-ins for real Flint output).
# TODO: replace with traces loaded from evalgate.ingest.TraceStore.iter_failures() or a fixture.
# --------------------------------------------------------------------------------------

def load_sample_traces() -> list[Trace]:
    good = Trace(
        trace_id="flint-001",
        root_input="Read users.csv, drop nulls, write to warehouse.",
        root_output=json.dumps(
            {
                "nodes": [
                    {"id": "a", "type": "source"},
                    {"id": "b", "type": "filter"},
                    {"id": "c", "type": "sink"},
                ],
                "edges": [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}],
            }
        ),
        spans=[Span(name="parse", gen_ai_operation="chat", latency_ms=850.0)],
    )
    bad_cycle = Trace(
        trace_id="flint-002",
        status="error",
        root_input="Join orders and refunds, then feed back into orders.",
        root_output=json.dumps(
            {
                "nodes": [{"id": "x", "type": "join"}, {"id": "y", "type": "transform"}],
                "edges": [{"from": "x", "to": "y"}, {"from": "y", "to": "x"}],  # cycle!
            }
        ),
        spans=[Span(name="parse", gen_ai_operation="chat", latency_ms=1200.0, status="error")],
    )
    return [good, bad_cycle]


def main() -> int:
    settings = get_settings()
    traces = load_sample_traces()

    # 1. Deterministic DAG code checks.
    results: list[EvalResult] = []
    for trace in traces:
        results.extend(run_dag_code_checks(trace))

    # 2. LLM-judge for the fuzzy criterion ("does the DAG match the intent?").
    #    TODO: needs GROQ_API_KEY; wrapped so the example still runs offline.
    judge = LLMJudge(settings=settings)
    rubric = "The DAG faithfully implements the data pipeline described in the input."
    for trace in traces:
        try:
            results.append(judge.evaluate(trace, rubric=rubric))
        except Exception as exc:  # noqa: BLE001 - example resilience only
            print(f"[judge skipped for {trace.trace_id}: {exc}]")

    # 3. Calibrate the judge against a human anchor set.
    #    TODO: run the judge over the real anchor set; here we use placeholder labels.
    judge_labels = [True, True, False, True]
    human_labels = [True, True, False, False]
    calibration = calibrate(judge_labels, human_labels, min_kappa=settings.min_judge_kappa)

    # 4. Gate.
    return ci_gate(results, min_pass_rate=settings.min_pass_rate, calibration=calibration)


if __name__ == "__main__":
    raise SystemExit(main())
