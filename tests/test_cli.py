"""Tests for the evalgate CLI (offline paths)."""

import json

from evalgate.cli import _load_traces, main


def _write_traces(path, statuses):
    with open(path, "w", encoding="utf-8") as fh:
        for i, st in enumerate(statuses):
            fh.write(json.dumps({"input": f"q{i}", "output": f"a{i}", "status": st}) + "\n")


def test_load_traces_shorthand(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"input": "hi", "output": "bye", "status": "ok"}\n', encoding="utf-8")
    traces = _load_traces(p)
    assert traces[0].root_input == "hi"
    assert traces[0].root_output == "bye"


def test_demo_command_exits_zero():
    assert main(["demo"]) == 0


def test_demo_json_output(capsys):
    code = main(["demo", "--json"])
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "pass_rate" in parsed
    assert code == 0


def test_gate_status_eval_pass_and_fail(tmp_path):
    p = tmp_path / "t.jsonl"
    _write_traces(p, ["ok"] * 48 + ["error"] * 2)  # 0.96
    assert main(["gate", "--traces", str(p), "--eval", "status", "--min-pass-rate", "0.80"]) == 0
    assert main(["gate", "--traces", str(p), "--eval", "status", "--min-pass-rate", "0.99"]) == 1


def test_gate_writes_badge(tmp_path):
    p = tmp_path / "t.jsonl"
    _write_traces(p, ["ok"] * 50)
    badge = tmp_path / "badge.json"
    main(["gate", "--traces", str(p), "--eval", "status", "--min-pass-rate", "0.5",
          "--badge", str(badge)])
    data = json.loads(badge.read_text())
    assert data["schemaVersion"] == 1
    assert data["color"] == "brightgreen"


def test_calibrate_offline_demo(tmp_path, capsys):
    from evalgate.reference import generate_anchor_set

    p = tmp_path / "anchors.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        for a in generate_anchor_set(n=24, seed=1):
            fh.write(a.model_dump_json() + "\n")
    code = main(["calibrate", "--anchors", str(p)])
    out = capsys.readouterr().out
    assert "kappa" in out
    assert code == 0  # demo judge lands above the drift threshold


def test_analyze_command(tmp_path, capsys):
    from evalgate.reference import generate_flint_traces

    p = tmp_path / "t.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        for t in generate_flint_traces(n=60, n_fail=12, seed=0):
            fh.write(json.dumps({"trace_id": t.trace_id, "input": t.root_input,
                                 "output": t.root_output, "status": t.status}) + "\n")
    assert main(["analyze", "--traces", str(p)]) == 0
    assert "taxonomy" in capsys.readouterr().out
