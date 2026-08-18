"""Tests for the OTel GenAI ingestion adapter (version-tolerant span -> trace mapping)."""

from evalgate.otel import span_to_span, spans_to_traces


def test_flat_span_shape_from_brief():
    span = {
        "name": "invoke_agent parser",
        "trace_id": "t1",
        "gen_ai.operation.name": "chat",
        "input": "pull signups, enrich, post to slack",
        "output": '{"nodes": [], "edges": []}',
        "gen_ai.usage.input_tokens": 240,
        "gen_ai.usage.output_tokens": 96,
        "latency_ms": 1180,
        "status": "OK",
    }
    s = span_to_span(span)
    assert s.gen_ai_operation == "chat"
    assert s.tokens == 336
    assert s.latency_ms == 1180.0
    assert s.status == "ok"
    assert "signups" in s.input


def test_nested_attributes_and_message_list():
    span = {
        "name": "chat",
        "attributes": {
            "gen_ai.operation.name": "chat",
            "gen_ai.input.messages": [{"role": "user", "content": "hello"}],
            "gen_ai.output.messages": [{"role": "assistant", "content": "hi there"}],
        },
    }
    s = span_to_span(span)
    assert s.input == "hello"
    assert s.output == "hi there"


def test_error_status_variants():
    assert span_to_span({"name": "x", "status": "ERROR"}).status == "error"
    assert span_to_span({"name": "x", "status": {"status_code": "ERROR"}}).status == "error"
    assert span_to_span({"name": "x", "status": "OK"}).status == "ok"


def test_latency_from_nanos():
    s = span_to_span({"name": "x", "start_time_unix_nano": 1_000_000_000,
                      "end_time_unix_nano": 1_500_000_000})
    assert s.latency_ms == 500.0


def test_grouping_into_traces():
    spans = [
        {"name": "invoke_agent", "trace_id": "a", "gen_ai.operation.name": "invoke_agent",
         "input": "do the thing"},
        {"name": "execute_tool", "trace_id": "a", "gen_ai.operation.name": "execute_tool",
         "status": "ERROR"},
        {"name": "chat", "trace_id": "a", "gen_ai.operation.name": "chat",
         "output": "final answer"},
        {"name": "chat", "trace_id": "b", "gen_ai.operation.name": "chat",
         "input": "another", "output": "ok"},
    ]
    traces = spans_to_traces(spans)
    assert len(traces) == 2
    a = next(t for t in traces if t.trace_id == "a")
    assert a.root_input == "do the thing"
    assert a.root_output == "final answer"
    assert a.status == "error"  # a tool span errored
    assert len(a.spans) == 3
