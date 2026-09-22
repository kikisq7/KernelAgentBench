from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

RunType = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]


def tracing_enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"} and bool(
        os.environ.get("LANGSMITH_API_KEY")
    )


def trace_payload(
    value: dict[str, Any],
    *,
    include_content: bool,
) -> dict[str, Any]:
    if include_content:
        return value
    redacted = dict(value)
    for key in ("prompt", "response", "source", "patch"):
        if key in redacted:
            redacted[key] = "[redacted]"
    return redacted


@contextmanager
def trace_span(
    name: str,
    *,
    inputs: dict[str, Any],
    metadata: dict[str, Any],
    include_content: bool,
    run_type: RunType = "chain",
) -> Iterator[Any | None]:
    if not tracing_enabled():
        yield None
        return

    from langsmith import trace

    with trace(
        name,
        run_type=run_type,
        inputs=trace_payload(inputs, include_content=include_content),
        metadata=metadata,
        project_name=os.environ.get("LANGSMITH_PROJECT", "kernel-agent-bench"),
    ) as run_tree:
        yield run_tree


def add_trace_outputs(run_tree: Any | None, outputs: dict[str, Any], include_content: bool) -> None:
    if run_tree is not None:
        run_tree.add_outputs(trace_payload(outputs, include_content=include_content))
