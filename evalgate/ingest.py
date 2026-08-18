"""Ingest layer: OTLP GenAI span receiver + Postgres/pgvector trace store.

The SQLAlchemy 2.0 model and the ``TraceStore`` CRUD are real. The OTLP wiring
(``OTLPReceiver``) that maps raw ``opentelemetry`` protobuf spans onto
:class:`~evalgate.models.Trace` is left as a clearly-marked stub — plumbing, not the crux.

No network / DB connection happens at import time; the engine is created lazily.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, Float, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import get_settings
from .models import Span, Trace

# Embedding dimension for the pgvector column. Match this to your encoder in evalgate.analysis.
EMBED_DIM = 384


class Base(DeclarativeBase):
    """Declarative base for EvalGate's ORM models (owns the SQLAlchemy metadata)."""


class TraceRecord(Base):
    """Persisted trace. The ``embedding`` column is a pgvector vector used for
    failure clustering + similarity search inside the error-analysis workbench."""

    __tablename__ = "traces"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    root_input: Mapped[str | None] = mapped_column(String, nullable=True)
    root_output: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String(16), index=True, default="ok")
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    spans: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.UTC)
    )

    def to_trace(self) -> Trace:
        return Trace(
            trace_id=self.trace_id,
            root_input=self.root_input,
            root_output=self.root_output,
            status=self.status,
            spans=[Span(**s) for s in (self.spans or [])],
        )


class TraceStore:
    """Thin persistence wrapper around Postgres + pgvector."""

    def __init__(self, database_url: str | None = None) -> None:
        self._url = database_url or get_settings().database_url
        self._engine = create_engine(self._url, future=True)
        self._Session: sessionmaker[Session] = sessionmaker(self._engine, expire_on_commit=False)

    def create_all(self) -> None:
        """Create the pgvector extension + tables. Call once on setup.

        TODO: run ``CREATE EXTENSION IF NOT EXISTS vector;`` before ``create_all`` on a fresh DB
        (the pgvector/pgvector image ships the extension but it must be enabled per-database).
        """
        Base.metadata.create_all(self._engine)

    def store(self, trace: Trace, embedding: list[float] | None = None) -> None:
        """Upsert a trace (+ optional embedding) into Postgres."""
        record = TraceRecord(
            trace_id=trace.trace_id,
            root_input=trace.root_input,
            root_output=trace.root_output,
            status=trace.status,
            latency_ms=trace.latency_ms,
            spans=[s.model_dump() for s in trace.spans],
            embedding=embedding,
        )
        with self._Session.begin() as session:
            session.merge(record)  # merge => idempotent on trace_id

    def iter_failures(self) -> Iterator[Trace]:
        """Stream stored failing traces for error analysis / evaluation."""
        with self._Session() as session:
            stmt = select(TraceRecord).where(TraceRecord.status == "error")
            for record in session.scalars(stmt):
                yield record.to_trace()


class OTLPReceiver:
    """Accepts OpenTelemetry GenAI spans and persists them as traces.

    TODO: wire the actual OTLP ingest path. Options:
      * run an OTLP/HTTP endpoint and decode ``ExportTraceServiceRequest`` protobufs, or
      * plug a custom ``SpanExporter`` that forwards to :meth:`ingest`.
    Then group spans by ``trace_id``, map ``gen_ai.*`` attributes onto
    :class:`~evalgate.models.Span`, derive root input/output + status, embed, and store.
    """

    def __init__(self, store: TraceStore) -> None:
        self.store = store

    def ingest(self, otlp_spans: Any) -> None:  # noqa: ARG002
        # TODO: decode OTLP spans -> group by trace_id -> build Trace -> embed -> store.
        raise NotImplementedError("OTLP receiver wiring not implemented yet")


def main() -> int:
    """``make ingest`` entrypoint (placeholder)."""
    print("evalgate.ingest: OTLP receiver not wired yet. Implement OTLPReceiver.ingest, or use "
          "TraceStore.store(trace) directly. See docstring for the mapping steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
