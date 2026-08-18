"""FastAPI surface for EvalGate.

Endpoints are stubs (typed shapes + ``# TODO``) so the app boots and serves docs immediately,
but doesn't pretend to do work it can't yet. ``main()`` is the ``evalgate`` console script.
"""

from __future__ import annotations

from fastapi import FastAPI

from . import __version__
from .models import CalibrationReport, FailureCluster, Trace

app = FastAPI(
    title="EvalGate",
    version=__version__,
    summary="Error-analysis-first eval + observability for LLM agents, with judge drift detection.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/traces")
def post_traces(trace: Trace) -> dict[str, str]:
    """Ingest a single trace (JSON body) into the store.

    TODO: embed + persist via evalgate.ingest.TraceStore.store(...).
    """
    # TODO: wire to TraceStore; embed the trace text for clustering.
    return {"status": "accepted", "trace_id": trace.trace_id}


@app.get("/taxonomy")
def get_taxonomy() -> list[FailureCluster]:
    """Return the current failure taxonomy (clustered failing traces)."""
    # TODO: pull failure embeddings from the store and run evalgate.analysis.cluster_failures.
    return []


@app.get("/report")
def get_report() -> dict[str, object]:
    """Return the latest eval summary (pass-rate + CI + per-check breakdown)."""
    # TODO: run evaluators over stored traces and return evalgate.gate outputs as JSON.
    return {"detail": "not implemented"}


@app.get("/calibration")
def get_calibration() -> dict[str, object] | CalibrationReport:
    """Return the latest judge calibration report."""
    # TODO: load anchor set, score with the judge, return evalgate.calibration.calibrate(...).
    return {"detail": "not implemented"}


def main() -> None:
    """Console-script entrypoint: ``evalgate`` -> serve the API."""
    import uvicorn

    uvicorn.run("evalgate.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
