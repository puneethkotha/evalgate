"""``evalgate`` command-line interface.

The adoption surface: a single binary that turns agent quality into a shell exit code, drops
into any CI job, and prints the instrument readout. Uses only the standard library's argparse
(no extra dependency) to stay $0 and light.

Subcommands:
  * ``demo``      — run the offline reference pipeline (no key, no data needed).
  * ``gate``      — evaluate a traces file and exit non-zero if the gate fails.
  * ``analyze``   — cluster failing traces into a failure taxonomy.
  * ``calibrate`` — score the judge against an anchor set and report agreement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .models import Span, Trace


def _load_traces(path: str | Path) -> list[Trace]:
    """Load traces from JSONL. Each line accepts ``root_input``/``root_output`` or the shorthand
    ``input``/``output``; ``status`` defaults to 'ok'; ``spans`` is optional."""
    traces: list[Trace] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"bad trace at {path}:{lineno}: {exc}") from exc
            spans = [Span(**s) for s in obj.get("spans", [])]
            traces.append(Trace(
                trace_id=obj.get("trace_id", f"trace-{lineno}"),
                root_input=obj.get("root_input", obj.get("input")),
                root_output=obj.get("root_output", obj.get("output")),
                status=obj.get("status", "ok"),
                spans=spans,
            ))
    return traces


def _judge_passes(traces: list[Trace], rubric: str, settings) -> list[bool]:
    from .evaluators import LLMJudge

    judge = LLMJudge(settings=settings)
    return [judge.judge(t.root_input or "", t.root_output or "", rubric).passed for t in traces]


def _write_badge(report, path: str) -> None:
    """Write a shields.io endpoint JSON so a README badge can show live gate status."""
    c = report.calibration
    kappa = f" · κ{c.kappa:.2f}" if c is not None and c.kappa == c.kappa else ""
    badge = {
        "schemaVersion": 1,
        "label": "evalgate",
        "message": f"{'pass' if report.passed else 'fail'} · {report.pass_rate.point:.0%}{kappa}",
        "color": "brightgreen" if report.passed else "red",
    }
    Path(path).write_text(json.dumps(badge), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Subcommand handlers.
# --------------------------------------------------------------------------------------

def cmd_demo(args: argparse.Namespace) -> int:
    from .gate import render_gate
    from .reference import run_demo

    taxonomy, report = run_demo(seed=args.seed)
    if args.json:
        print(report.model_dump_json(indent=2))
        return 0 if report.passed else 1
    print("failure taxonomy (error-analysis-first)")
    print("-" * 64)
    for i, c in enumerate(taxonomy, 1):
        print(f"  C{i}  n={c.size:<3} {c.label}")
    print()
    print(render_gate(report))
    return 0 if report.passed else 1


def cmd_gate(args: argparse.Namespace) -> int:
    from .calibration import calibrate, load_anchor_set
    from .config import get_settings
    from .gate import evaluate_gate, render_gate

    settings = get_settings()
    traces = _load_traces(args.traces)

    calibration = None
    judge_derived = False
    if args.eval == "judge" or (args.eval == "auto" and settings.groq_api_key):
        if not settings.groq_api_key:
            raise SystemExit("gate --eval judge needs GROQ_API_KEY (or use --eval status)")
        passes = _judge_passes(traces, args.rubric, settings)
        judge_derived = True
        if args.anchors:
            anchors = load_anchor_set(args.anchors)
            judge_labels = _judge_passes(
                [Trace(trace_id=f"a{i}", root_input=a.input, root_output=a.output)
                 for i, a in enumerate(anchors)], args.rubric, settings)
            calibration = calibrate(judge_labels, [a.human_label for a in anchors],
                                    min_kappa=args.min_kappa)
    else:
        passes = [t.status == "ok" for t in traces]

    baseline_passes = None
    if args.baseline:
        base_traces = _load_traces(args.baseline)
        if args.eval in ("judge",) or (args.eval == "auto" and settings.groq_api_key):
            baseline_passes = _judge_passes(base_traces, args.rubric, settings)
        else:
            baseline_passes = [t.status == "ok" for t in base_traces]

    report = evaluate_gate(
        passes,
        min_pass_rate=args.min_pass_rate,
        calibration=calibration,
        baseline_passes=baseline_passes,
        confidence=args.confidence,
        judge_derived=judge_derived,
    )
    if args.badge:
        _write_badge(report, args.badge)
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(render_gate(report))
    return 0 if report.passed else 1


def cmd_analyze(args: argparse.Namespace) -> int:
    from .analysis import build_taxonomy

    traces = _load_traces(args.traces)
    failing = [t for t in traces if t.is_failure] or traces
    taxonomy = build_taxonomy([t.text() for t in failing], k=args.k)
    if args.json:
        print(json.dumps([c.model_dump() for c in taxonomy], indent=2))
        return 0
    print(f"failure taxonomy — {len(failing)} traces, {len(taxonomy)} clusters")
    print("-" * 64)
    for i, c in enumerate(taxonomy, 1):
        print(f"  C{i}  n={c.size:<3} {c.label}")
        for ex in c.exemplars[:1]:
            print(f"        e.g. {ex[:80]}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibration import calibrate, load_anchor_set
    from .config import get_settings

    settings = get_settings()
    anchors = load_anchor_set(args.anchors)
    human = [a.human_label for a in anchors]
    if settings.groq_api_key:
        judge = _judge_passes(
            [Trace(trace_id=f"a{i}", root_input=a.input, root_output=a.output)
             for i, a in enumerate(anchors)], args.rubric, settings)
    else:
        from .reference import demo_judge_labels
        print("(no GROQ_API_KEY — using the offline demo judge)", file=sys.stderr)
        judge = demo_judge_labels(anchors)
    report = calibrate(judge, human, min_kappa=args.min_kappa)
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(f"judge calibration — n={report.n}")
        print(f"  kappa {report.kappa:.3f} ({report.band})   AC1 {report.ac1:.3f}   "
              f"raw {report.raw_agreement:.3f}")
        print(f"  TPR {report.tpr:.3f}   TNR {report.tnr:.3f}   prevalence {report.prevalence:.3f}")
        print(f"  drifted = {report.drifted}  (min kappa {report.min_kappa:.2f})")
    return 1 if report.drifted else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalgate", description=(
        "A CI gate for LLM agents: error-analysis-first evals, a calibrated judge, and a "
        "pass/fail exit code."))
    p.add_argument("--version", action="version", version=f"evalgate {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run the offline reference pipeline (no key needed)")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_demo)

    g = sub.add_parser("gate", help="evaluate a traces file and exit non-zero on failure")
    g.add_argument("--traces", required=True, help="JSONL of traces")
    g.add_argument("--anchors", help="JSONL anchor set for judge calibration")
    g.add_argument("--baseline", help="JSONL of a prior version's traces (paired McNemar)")
    g.add_argument("--eval", choices=["auto", "judge", "status"], default="auto",
                   help="per-trace pass signal: judge verdict, the trace's own status, or auto")
    g.add_argument("--rubric", default="The output correctly and faithfully answers the input.")
    g.add_argument("--min-pass-rate", type=float, default=0.9)
    g.add_argument("--min-kappa", type=float, default=0.7)
    g.add_argument("--confidence", type=float, default=0.95)
    g.add_argument("--badge", help="write a shields.io endpoint JSON to this path")
    g.add_argument("--json", action="store_true")
    g.set_defaults(func=cmd_gate)

    a = sub.add_parser("analyze", help="cluster failing traces into a failure taxonomy")
    a.add_argument("--traces", required=True)
    a.add_argument("--k", type=int, default=None, help="cluster count (default: auto)")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("calibrate", help="score the judge against an anchor set")
    c.add_argument("--anchors", required=True)
    c.add_argument("--rubric", default="The output correctly and faithfully answers the input.")
    c.add_argument("--min-kappa", type=float, default=0.7)
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_calibrate)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
