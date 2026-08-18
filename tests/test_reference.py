"""Tests for the offline reference corpus (NL->DAG parser)."""

from evalgate.reference import (
    dag_code_checks,
    demo_judge_labels,
    generate_anchor_set,
    generate_flint_traces,
    is_a_dag,
    parse_dag,
    trace_passes,
)


def test_corpus_is_deterministic_and_sized():
    a = generate_flint_traces(n=50, n_fail=6, seed=0)
    b = generate_flint_traces(n=50, n_fail=6, seed=0)
    assert len(a) == 50
    assert [t.trace_id for t in a] == [t.trace_id for t in b]  # deterministic order
    assert [t.root_input for t in a] == [t.root_input for t in b]


def test_corpus_failure_count_matches_and_fails_checks():
    traces = generate_flint_traces(n=60, n_fail=9, seed=0)
    failing = [t for t in traces if t.is_failure]
    assert len(failing) == 9
    # Every injected failure trips at least one deterministic code check.
    for t in failing:
        assert not trace_passes(t)


def test_good_traces_pass_all_checks():
    traces = generate_flint_traces(n=40, n_fail=0, seed=3)
    assert all(trace_passes(t) for t in traces)


def test_is_a_dag_detects_cycle_and_accepts_chain():
    chain = {"nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
             "edges": [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}]}
    cycle = {"nodes": [{"id": "x"}, {"id": "y"}],
             "edges": [{"from": "x", "to": "y"}, {"from": "y", "to": "x"}]}
    assert is_a_dag(chain) is True
    assert is_a_dag(cycle) is False


def test_dag_code_checks_flags_each_mode():
    traces = generate_flint_traces(n=30, n_fail=9, seed=0)
    checks_seen = set()
    for t in traces:
        for r in dag_code_checks(t):
            if not r.passed:
                checks_seen.add(r.evaluator)
    # cycle -> is_a_dag; hallucinated -> legal_node_types; dangling -> is_a_dag/edges_resolve
    assert "is_a_dag" in checks_seen
    assert "legal_node_types" in checks_seen


def test_parse_dag_rejects_non_dag():
    assert parse_dag("not json") is None
    assert parse_dag('{"foo": 1}') is None
    assert parse_dag('{"nodes": [], "edges": []}') == {"nodes": [], "edges": []}


def test_anchor_set_balanced_and_labeled():
    anchors = generate_anchor_set(n=24, seed=1)
    assert len(anchors) == 24
    pos = sum(1 for a in anchors if a.human_label)
    assert pos == 12  # balanced


def test_demo_judge_labels_balanced_flips():
    anchors = generate_anchor_set(n=24, seed=1)
    human = [a.human_label for a in anchors]
    judge = demo_judge_labels(anchors, flips_per_class=1)
    disagreements = sum(1 for h, j in zip(human, judge, strict=True) if h != j)
    assert disagreements == 2  # one positive + one negative flipped
