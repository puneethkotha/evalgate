"""FastAPI surface for EvalGate.

Serves the instrument-panel dashboard (static files under ``dashboard/``) and a live
``/report.json`` endpoint that regenerates the dashboard payload on demand. The same dashboard
deploys as a fully static site (Cloudflare Pages / GitHub Pages) by shipping a pre-built
``dashboard/report.json`` — no backend required.

``main()`` is the ``evalgate-serve`` console script.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from . import __version__

app = FastAPI(
    title="EvalGate",
    version=__version__,
    summary="Error-analysis-first eval + CI gate for LLM agents, with judge drift detection.",
)

_DASHBOARD = Path(__file__).resolve().parent.parent / "dashboard"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/report.json")
def report() -> JSONResponse:
    """Regenerate the dashboard payload live (the static deploy uses the checked-in file)."""
    from .report import build_dashboard_report

    return JSONResponse(build_dashboard_report())


# Mount the static dashboard last so explicit routes (/health, /report.json) win. When running
# from an installed wheel without the dashboard dir, this is simply skipped.
if _DASHBOARD.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_DASHBOARD), html=True), name="dashboard")


def main() -> None:
    """Console-script entrypoint: serve the dashboard + API on :8000."""
    import uvicorn

    uvicorn.run("evalgate.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
