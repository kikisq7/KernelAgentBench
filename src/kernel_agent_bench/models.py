from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import ModelSelector, TokenRecord


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)
    )


def resolve_model_ids(
    selectors: Iterable[ModelSelector],
    available_ids: Iterable[str],
) -> dict[str, str]:
    ids = list(available_ids)
    resolved: dict[str, str] = {}
    for selector in selectors:
        pattern = re.compile(selector.match, re.IGNORECASE)
        matches = [model_id for model_id in ids if pattern.search(model_id)]
        if not matches:
            raise ValueError(
                f"no Cursor model matches {selector.alias!r} / {selector.match!r}; "
                f"available: {', '.join(sorted(ids))}"
            )
        resolved[selector.alias] = sorted(matches, key=_natural_key, reverse=True)[0]
    if len(set(resolved.values())) != len(resolved):
        raise ValueError(f"model selectors resolved to duplicate IDs: {resolved}")
    return resolved


def discover_models(selectors: Iterable[ModelSelector]) -> dict[str, str]:
    from cursor_sdk import Cursor

    catalog = Cursor.models.list()
    return resolve_model_ids(selectors, (entry.id for entry in catalog))


@dataclass(frozen=True)
class AgentRun:
    status: str
    agent_id: str
    run_id: str
    response: str
    usage: TokenRecord
    cost: dict[str, Any] | None


def _token_record(usage: Any | None) -> TokenRecord:
    if usage is None:
        return TokenRecord(available=False)
    return TokenRecord(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        total_tokens=usage.total_tokens,
        reasoning_tokens=usage.reasoning_tokens,
    )


def run_cursor_agent(model_id: str, workspace: Path, prompt: str) -> AgentRun:
    from cursor_sdk import Agent, LocalAgentOptions

    api_key = os.environ.get("CURSOR_API_KEY")
    if not api_key:
        raise RuntimeError("CURSOR_API_KEY is required for a real model run")

    with Agent.create(
        model=model_id,
        api_key=api_key,
        local=LocalAgentOptions(cwd=str(workspace)),
    ) as agent:
        run = agent.send(prompt)
        result = run.wait()
        response = result.result or ""
        cost: dict[str, Any] | None = None
        try:
            billed = agent.get_usage()
            cost = {
                "total_tokens": billed.usage.total_tokens,
                "raw_cost_cents": billed.cost.raw_cost_cents if billed.cost else None,
                "charged_cents": billed.cost.charged_cents if billed.cost else None,
            }
        except Exception:
            # Per-run billed usage may lag or be unavailable for an account.
            cost = None
        return AgentRun(
            status=str(result.status),
            agent_id=agent.agent_id,
            run_id=run.id,
            response=response,
            usage=_token_record(result.usage),
            cost=cost,
        )
