.PHONY: install dev ingest gate test lint

# Install package + dev tooling in editable mode.
install:
	pip install -e ".[dev]"

# Run the API (auto-reload) for local development.
dev:
	uvicorn evalgate.api:app --reload --port 8000

# Ingest OTel GenAI spans into Postgres + pgvector.
# TODO: point at your OTLP source; today this runs the placeholder receiver.
ingest:
	python -m evalgate.ingest

# Run the CI gate: code checks + judge + calibration -> exit 0/1.
gate:
	python -m evalgate.gate

test:
	pytest -q

lint:
	ruff check evalgate tests examples
