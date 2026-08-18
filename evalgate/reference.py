"""Reference integration: an NL->DAG parser (the sibling "Flint" project).

This module exercises the whole EvalGate pipeline **offline and deterministically** — no network,
no API key — so the reference example and the tests both produce real, reproducible numbers:

  * ``generate_flint_traces`` builds a corpus of NL->DAG traces with a realistic, seeded mix of
    good outputs and known failure modes (cycles, illegal node types, dangling edges).
  * the DAG code checks (*is-a-DAG* via Kahn's algorithm, *legal node types*, *edges resolve*)
    are the deterministic "crux" for this agent.
  * ``generate_anchor_set`` + ``demo_judge_labels`` give a human anchor set and a stand-in judge
    so calibration + drift detection run without a live model (swap in a real ``LLMJudge`` by
    setting ``GROQ_API_KEY``).

A DAG is ``{"nodes": [{"id": str, "type": str}, ...], "edges": [{"from": str, "to": str}, ...]}``.
"""

from __future__ import annotations

import json
import random
from typing import Any

from .models import AnchorExample, EvalResult, Span, Trace

LEGAL_NODE_TYPES = {
    "source", "transform", "filter", "join", "sink", "sql", "http", "slack", "email", "shell",
}

FLINT_RUBRIC = (
    "The DAG faithfully implements the pipeline described in the input: every step named in the "
    "request is present, no steps are hallucinated, and the dependency order matches the request."
)


# --------------------------------------------------------------------------------------
# Deterministic DAG code checks.
# --------------------------------------------------------------------------------------

def parse_dag(output: Any) -> dict[str, Any] | None:
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
    """Directed + acyclic: Kahn's topological sort must consume every node."""
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


def dag_code_checks(trace: Trace) -> list[EvalResult]:
    """Run the DAG-specific deterministic checks on a trace's root output."""
    dag = parse_dag(trace.root_output)
    if dag is None:
        return [EvalResult(evaluator="dag_parses", passed=False,
                           critique="output is not valid DAG JSON")]
    checks = {
        "is_a_dag": is_a_dag(dag),
        "legal_node_types": legal_node_types(dag),
        "edges_resolve": edges_resolve(dag),
    }
    return [
        EvalResult(evaluator=name, passed=passed,
                   critique="" if passed else f"{name} failed")
        for name, passed in checks.items()
    ]


def trace_passes(trace: Trace) -> bool:
    """Trace-level verdict: all code checks pass."""
    return all(r.passed for r in dag_code_checks(trace))


# --------------------------------------------------------------------------------------
# Corpus generation.
# --------------------------------------------------------------------------------------

# (natural-language request, well-formed DAG) templates. Node ids are single letters; types are
# from the legal grammar.
_GOOD_TEMPLATES: list[tuple[str, dict[str, Any]]] = [
    ("Read {t}.csv, drop nulls, and write to the warehouse.",
     {"nodes": [{"id": "a", "type": "source"}, {"id": "b", "type": "filter"},
                {"id": "c", "type": "sink"}],
      "edges": [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}]}),
    ("Pull new signups from Postgres, enrich with Clearbit, then post a summary to Slack.",
     {"nodes": [{"id": "pull", "type": "sql"}, {"id": "enrich", "type": "http"},
                {"id": "post", "type": "slack"}],
      "edges": [{"from": "pull", "to": "enrich"}, {"from": "enrich", "to": "post"}]}),
    ("Join {t} and refunds, then load the result.",
     {"nodes": [{"id": "o", "type": "source"}, {"id": "r", "type": "source"},
                {"id": "j", "type": "join"}, {"id": "s", "type": "sink"}],
      "edges": [{"from": "o", "to": "j"}, {"from": "r", "to": "j"}, {"from": "j", "to": "s"}]}),
    ("Filter {t} over $100 and export them.",
     {"nodes": [{"id": "src", "type": "source"}, {"id": "flt", "type": "filter"},
                {"id": "out", "type": "sink"}],
      "edges": [{"from": "src", "to": "flt"}, {"from": "flt", "to": "out"}]}),
    ("Back up the {t} database nightly, then email me the log.",
     {"nodes": [{"id": "bk", "type": "shell"}, {"id": "em", "type": "email"}],
      "edges": [{"from": "bk", "to": "em"}]}),
]

_TABLES = ["users", "orders", "events", "transactions", "signups", "invoices", "sessions",
           "tickets", "payments", "leads"]


def _good_trace(tid: str, rng: random.Random) -> Trace:
    text, dag = rng.choice(_GOOD_TEMPLATES)
    table = rng.choice(_TABLES)
    latency = rng.uniform(600, 1600)
    return Trace(
        trace_id=tid,
        root_input=text.format(t=table),
        root_output=json.dumps(dag),
        status="ok",
        spans=[Span(name="parse", gen_ai_operation="chat", latency_ms=latency)],
    )


def _fail_trace(tid: str, mode: str, rng: random.Random) -> Trace:
    table = rng.choice(_TABLES)
    latency = rng.uniform(700, 1700)
    if mode == "cycle":
        inp = f"Join {table} and refunds, then feed the result back into {table}."
        dag = {"nodes": [{"id": "x", "type": "join"}, {"id": "y", "type": "transform"}],
               "edges": [{"from": "x", "to": "y"}, {"from": "y", "to": "x"}]}  # cycle
    elif mode == "hallucinated_type":
        inp = f"Pull {table} and push them to a webhook endpoint."
        dag = {"nodes": [{"id": "s", "type": "source"}, {"id": "w", "type": "webhook"}],
               "edges": [{"from": "s", "to": "w"}]}  # 'webhook' not in grammar
    else:  # dangling_edge
        inp = f"Read {table}, transform, and load — but wire the last step to a missing node."
        dag = {"nodes": [{"id": "a", "type": "source"}, {"id": "b", "type": "transform"}],
               "edges": [{"from": "a", "to": "b"}, {"from": "b", "to": "ghost"}]}  # dangling
    return Trace(
        trace_id=tid,
        root_input=inp,
        root_output=json.dumps(dag),
        status="error",
        spans=[Span(name="parse", gen_ai_operation="chat", latency_ms=latency, status="error")],
    )


def generate_flint_traces(n: int = 200, n_fail: int = 10, seed: int = 0) -> list[Trace]:
    """Deterministically generate a corpus of NL->DAG traces (mostly good, ``n_fail`` broken)."""
    rng = random.Random(seed)
    n_fail = min(n_fail, n)
    modes = ["cycle", "hallucinated_type", "dangling_edge"]
    traces: list[Trace] = [_good_trace(f"flint-{i:04d}", rng) for i in range(n - n_fail)]
    for i in range(n_fail):
        traces.append(_fail_trace(f"flint-{n - n_fail + i:04d}", modes[i % len(modes)], rng))
    rng.shuffle(traces)
    return traces


# --------------------------------------------------------------------------------------
# Anchor set + offline stand-in judge (for calibration without a live model).
# --------------------------------------------------------------------------------------

def generate_anchor_set(n: int = 24, seed: int = 1) -> list[AnchorExample]:
    """A balanced human-labeled anchor set: half genuinely-good DAGs, half genuine failures."""
    rng = random.Random(seed)
    anchors: list[AnchorExample] = []
    half = n // 2
    for _ in range(half):
        _text, dag = rng.choice(_GOOD_TEMPLATES)
        table = rng.choice(_TABLES)
        anchors.append(AnchorExample(input=_text.format(t=table), output=json.dumps(dag),
                                     human_label=True))
    for i in range(n - half):
        t = _fail_trace("anchor", ["cycle", "hallucinated_type", "dangling_edge"][i % 3], rng)
        anchors.append(AnchorExample(input=t.root_input or "", output=t.root_output or "",
                                     human_label=False))
    rng.shuffle(anchors)
    return anchors


def demo_judge_labels(anchors: list[AnchorExample], flips_per_class: int = 1,
                      seed: int = 2) -> list[bool]:
    """A deterministic stand-in for a real judge: agrees with the human label except on a few
    deterministically-chosen anchors, flipping an equal number of positives and negatives so the
    confusion matrix (and the bias-corrected pass-rate) stay realistic. Lands kappa in the
    'substantial' band. Swap in a real :class:`~evalgate.evaluators.LLMJudge` when a key is set.
    """
    labels = [a.human_label for a in anchors]
    rng = random.Random(seed)
    pos_idx = [i for i, v in enumerate(labels) if v]
    neg_idx = [i for i, v in enumerate(labels) if not v]
    for pool in (pos_idx, neg_idx):
        for idx in rng.sample(pool, min(flips_per_class, len(pool))):
            labels[idx] = not labels[idx]
    return labels


def run_demo(n: int = 200, n_fail: int = 10, seed: int = 0, settings=None):
    """Run the whole EvalGate pipeline over the reference corpus and return (taxonomy, report).

    Shared by ``evalgate demo`` and ``examples/eval_flint_parser.py`` so there is one code path.
    Uses a real :class:`~evalgate.evaluators.LLMJudge` when ``GROQ_API_KEY`` is set, else the
    deterministic offline stand-in.
    """
    from .analysis import build_taxonomy
    from .calibration import calibrate
    from .config import get_settings
    from .gate import _evaluator_stats, evaluate_gate

    settings = settings or get_settings()
    traces = generate_flint_traces(n=n, n_fail=n_fail, seed=seed)

    results: list[EvalResult] = []
    passes: list[bool] = []
    for t in traces:
        rs = dag_code_checks(t)
        results.extend(rs)
        passes.append(all(r.passed for r in rs))

    failing = [t for t in traces if not trace_passes(t)]
    taxonomy = build_taxonomy([t.text() for t in failing], seed=seed)

    anchors = generate_anchor_set(n=24, seed=1)
    human_labels = [a.human_label for a in anchors]
    if settings.groq_api_key:
        from .evaluators import LLMJudge

        judge = LLMJudge(settings=settings)
        judge_labels = [judge.judge(a.input, a.output, FLINT_RUBRIC).passed for a in anchors]
    else:
        judge_labels = demo_judge_labels(anchors)
    calibration = calibrate(judge_labels, human_labels, min_kappa=settings.min_judge_kappa)

    # A prior agent version: three traces this version fixed (for a paired McNemar delta).
    baseline_passes = list(passes)
    fixed = 0
    for i, ok in enumerate(passes):
        if ok and fixed < 3:
            baseline_passes[i] = False
            fixed += 1

    report = evaluate_gate(
        passes,
        min_pass_rate=settings.min_pass_rate,
        calibration=calibration,
        baseline_passes=baseline_passes,
        by_evaluator=_evaluator_stats(results),
        confidence=0.95,
    )
    return taxonomy, report
