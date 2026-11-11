from __future__ import annotations

import os
import re
import shutil
import tempfile
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


def resolve_configured_models(
    selectors: Iterable[ModelSelector],
    available_ids: Iterable[str],
) -> dict[str, str]:
    """Resolve Cursor aliases from a catalog. Direct API aliases use their literal model id."""
    selected = list(selectors)
    resolved = {
        selector.alias: selector.model_id
        for selector in selected
        if selector.provider in {"anthropic", "openai"}
    }
    cursor_selectors = [selector for selector in selected if selector.provider == "cursor"]
    if cursor_selectors:
        resolved.update(resolve_model_ids(cursor_selectors, available_ids))
    if len(resolved) != len(selected):
        raise ValueError(f"model selectors were not all resolved: {resolved}")
    if len(set(resolved.values())) != len(resolved):
        raise ValueError(f"model selectors resolved to duplicate IDs: {resolved}")
    return resolved


def discover_models(selectors: Iterable[ModelSelector]) -> dict[str, str]:
    selected = list(selectors)
    available: list[str] = []
    if any(selector.provider == "cursor" for selector in selected):
        from cursor_sdk import Cursor

        available = [entry.id for entry in Cursor.models.list()]
    return resolve_configured_models(selected, available)


def uses_anthropic(model_id: str) -> bool:
    return model_id.startswith("claude-")


def uses_openai(model_id: str) -> bool:
    return model_id.startswith("gpt-")


@dataclass(frozen=True)
class AgentRun:
    status: str
    agent_id: str
    run_id: str
    response: str
    usage: TokenRecord
    cost: dict[str, Any] | None


def _token_record(usage: Any | None) -> TokenRecord:
    """Cursor's input_tokens already includes cache reads; store the uncached part."""
    if usage is None:
        return TokenRecord(available=False, disjoint=True)
    fresh = max(usage.input_tokens - usage.cache_read_tokens, 0)
    return TokenRecord(
        input_tokens=fresh,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        total_tokens=fresh
        + usage.output_tokens
        + usage.cache_read_tokens
        + usage.cache_write_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        disjoint=True,
    )


def disjoint_token_usage(usage: dict[str, Any], resolved_model_id: str) -> dict[str, Any]:
    """Apply the Cursor correction to a ledger row written before `disjoint` existed."""
    if usage.get("disjoint") or uses_anthropic(resolved_model_id) or uses_openai(
        resolved_model_id
    ):
        return {**usage, "disjoint": True}
    cache_read = int(usage.get("cache_read_tokens") or 0)
    fresh = max(int(usage.get("input_tokens") or 0) - cache_read, 0)
    output = int(usage.get("output_tokens") or 0)
    cache_write = int(usage.get("cache_write_tokens") or 0)
    return {
        **usage,
        "input_tokens": fresh,
        "total_tokens": fresh + output + cache_read + cache_write,
        "disjoint": True,
    }


def _int_field(payload: dict[str, Any], *names: str) -> int:
    for name in names:
        value = payload.get(name)
        if value is not None:
            return int(value)
    return 0


def anthropic_token_record(
    usage: dict[str, Any] | None,
    model_usage: dict[str, Any] | None,
) -> TokenRecord:
    """Prefer per-model totals so subagent and thinking tokens are included."""
    if model_usage:
        rows = [row for row in model_usage.values() if isinstance(row, dict)]
        input_tokens = sum(_int_field(row, "inputTokens", "input_tokens") for row in rows)
        output_tokens = sum(_int_field(row, "outputTokens", "output_tokens") for row in rows)
        cache_read = sum(
            _int_field(row, "cacheReadInputTokens", "cache_read_input_tokens") for row in rows
        )
        cache_write = sum(
            _int_field(row, "cacheCreationInputTokens", "cache_creation_input_tokens")
            for row in rows
        )
        reasoning = sum(_int_field(row, "thinkingTokens", "thinking_tokens") for row in rows)
        return TokenRecord(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            total_tokens=input_tokens + output_tokens + cache_read + cache_write,
            reasoning_tokens=reasoning,
            disjoint=True,
        )
    if not usage:
        return TokenRecord(available=False)
    input_tokens = _int_field(usage, "input_tokens")
    output_tokens = _int_field(usage, "output_tokens")
    cache_read = _int_field(usage, "cache_read_input_tokens")
    cache_write = _int_field(usage, "cache_creation_input_tokens")
    return TokenRecord(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        total_tokens=input_tokens + output_tokens + cache_read + cache_write,
        reasoning_tokens=None,
        disjoint=True,
    )


def run_anthropic_agent(
    model_id: str,
    workspace: Path,
    prompt: str,
    *,
    effort: str = "high",
) -> AgentRun:
    """Run one coding-agent turn against the Anthropic API, not Cursor."""
    import asyncio

    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing. Add it to KernelAgentBench/.env "
            "(see .env.example). Claude trials do not use CURSOR_API_KEY."
        )
    selected_effort = (os.environ.get("KAB_ANTHROPIC_EFFORT") or effort).strip()
    return asyncio.run(
        _anthropic_query(
            model_id,
            workspace,
            prompt,
            api_key=api_key,
            effort=selected_effort,
        )
    )


async def _anthropic_query(
    model_id: str,
    workspace: Path,
    prompt: str,
    *,
    api_key: str,
    effort: str,
) -> AgentRun:
    from claude_agent_sdk import ClaudeAgentOptions, ResultError, ResultMessage, query

    stderr_lines: list[str] = []

    def _stderr(line: str) -> None:
        stderr_lines.append(line.rstrip())
        del stderr_lines[:-40]

    # Empty config dir so a local Claude login cannot replace the API key.
    config_dir = tempfile.mkdtemp(prefix="kab-claude-")
    try:
        options = ClaudeAgentOptions(
            model=model_id,
            cwd=str(workspace),
            effort=effort,  # type: ignore[arg-type]
            # One unattended edit-and-test turn inside the temp workspace.
            permission_mode="bypassPermissions",
            max_turns=40,
            tools=["Read", "Edit", "Write", "Bash", "Glob", "Grep"],
            disallowed_tools=["WebSearch", "WebFetch", "Task", "Agent"],
            setting_sources=[],
            strict_mcp_config=True,
            fallback_model=None,
            env={
                "ANTHROPIC_API_KEY": api_key,
                "ANTHROPIC_MODEL": model_id,
                "CLAUDE_CONFIG_DIR": config_dir,
            },
            stderr=_stderr,
        )
        result: Any = None
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    result = message
        except ResultError as error:
            # The CLI yields the result (with usage) and then exits non-zero.
            # Hitting max_turns ends the turn; other errors still fail it.
            if result is None or error.subtype != "error_max_turns":
                raise
    finally:
        shutil.rmtree(config_dir, ignore_errors=True)
    if result is None:
        tail = "\n".join(stderr_lines[-10:])
        raise RuntimeError(f"Anthropic agent returned no result. {tail}".strip())
    tokens = anthropic_token_record(result.usage, result.model_usage)
    cents = None
    if result.total_cost_usd is not None:
        cents = int(round(float(result.total_cost_usd) * 100))
    succeeded = result.subtype == "success" and not result.is_error
    detail = result.result or ""
    if result.errors:
        detail = (detail + "\n" + "\n".join(result.errors)).strip()
    if not succeeded and stderr_lines:
        detail = (detail + "\n" + "\n".join(stderr_lines[-10:])).strip()
    return AgentRun(
        status="finished" if succeeded else str(result.subtype),
        agent_id=str(result.session_id),
        run_id=str(result.uuid or result.session_id),
        response=detail,
        usage=tokens,
        cost={
            "provider": "anthropic",
            "model_id": model_id,
            "effort": effort,
            "total_tokens": tokens.total_tokens,
            "raw_cost_cents": cents,
            "charged_cents": cents,
            "total_cost_usd": result.total_cost_usd,
        },
    )


def openai_token_record(usage: dict[str, Any] | None) -> TokenRecord:
    """OpenAI input_tokens includes cached input; output_tokens includes reasoning."""
    if not usage:
        return TokenRecord(available=False)
    input_tokens = _int_field(usage, "input_tokens")
    cached = _int_field(usage, "cached_input_tokens")
    output_tokens = _int_field(usage, "output_tokens")
    return TokenRecord(
        input_tokens=input_tokens - cached,
        output_tokens=output_tokens,
        cache_read_tokens=cached,
        cache_write_tokens=0,
        total_tokens=input_tokens + output_tokens,
        reasoning_tokens=_int_field(usage, "reasoning_output_tokens"),
        disjoint=True,
    )


@dataclass
class CodexTurn:
    thread_id: str = ""
    response: str = ""
    usage: dict[str, int] | None = None
    failure: str = ""
    errors: list[str] | None = None


def parse_codex_events(lines: Iterable[str]) -> CodexTurn:
    """Fold `codex exec --json` JSONL into one turn. Usage is summed over turns."""
    import json

    turn = CodexTurn(errors=[])
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "thread.started":
            turn.thread_id = str(event.get("thread_id") or "")
        elif kind == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message":
                turn.response = str(item.get("text") or "")
        elif kind == "turn.completed":
            usage = event.get("usage") or {}
            totals = dict(turn.usage or {})
            for name, value in usage.items():
                if isinstance(value, int):
                    totals[name] = totals.get(name, 0) + value
            turn.usage = totals
        elif kind == "turn.failed":
            turn.failure = str((event.get("error") or {}).get("message") or "turn failed")
        elif kind == "error":
            turn.errors = [*(turn.errors or []), str(event.get("message") or "")][-10:]
    return turn


def codex_binary() -> str:
    override = (os.environ.get("KAB_CODEX_BIN") or "").strip()
    if override:
        return override
    local = (
        Path(__file__).resolve().parents[2]
        / ".kernel-agent-bench/tools/node_modules/.bin/codex"
    )
    if local.is_file():
        return str(local)
    found = shutil.which("codex")
    if not found:
        raise RuntimeError(
            "Codex CLI not found. Install it with "
            "`npm install --prefix .kernel-agent-bench/tools @openai/codex@0.167.0` "
            "or set KAB_CODEX_BIN."
        )
    return found


def run_openai_agent(
    model_id: str,
    workspace: Path,
    prompt: str,
    *,
    effort: str = "high",
) -> AgentRun:
    """Run one coding-agent turn against the OpenAI API via the Codex CLI, not Cursor."""
    import subprocess

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to KernelAgentBench/.env "
            "(see .env.example). GPT trials do not use CURSOR_API_KEY."
        )
    selected_effort = (os.environ.get("KAB_OPENAI_EFFORT") or effort).strip()
    binary = codex_binary()
    version = subprocess.run(
        [binary, "--version"], text=True, capture_output=True, check=False
    ).stdout.strip()
    command = [
        binary,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        # One unattended edit-and-test turn inside the temp workspace. The Julia
        # tests write to ~/.julia, which the Codex sandbox would block.
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd",
        str(workspace),
        "--model",
        model_id,
        "-c",
        f'model_reasoning_effort="{selected_effort}"',
        "-c",
        'web_search="disabled"',
        "--disable",
        "multi_agent",
        "--disable",
        "plugins",
        "--disable",
        "apps",
        "-",
    ]
    # Empty CODEX_HOME so a local ChatGPT login cannot replace the API key.
    codex_home = tempfile.mkdtemp(prefix="kab-codex-")
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"OPENAI_API_KEY", "CODEX_API_KEY", "CURSOR_API_KEY", "ANTHROPIC_API_KEY"}
    }
    env.update({"CODEX_API_KEY": api_key, "CODEX_HOME": codex_home})
    try:
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )
    finally:
        shutil.rmtree(codex_home, ignore_errors=True)
    turn = parse_codex_events(completed.stdout.splitlines())
    stderr_tail = "\n".join(completed.stderr.splitlines()[-10:])
    if turn.usage is None:
        detail = "\n".join([turn.failure, *(turn.errors or [])[-3:], stderr_tail]).strip()
        raise RuntimeError(
            f"OpenAI agent returned no completed turn (exit {completed.returncode}). {detail}"
        )
    tokens = openai_token_record(turn.usage)
    succeeded = completed.returncode == 0 and not turn.failure
    detail = turn.response
    if not succeeded:
        detail = "\n".join([detail, turn.failure, stderr_tail]).strip()
    return AgentRun(
        status="finished" if succeeded else "failed",
        agent_id=turn.thread_id,
        run_id=turn.thread_id,
        response=detail,
        usage=tokens,
        cost={
            "provider": "openai",
            "model_id": model_id,
            "effort": selected_effort,
            "codex_version": version,
            "total_tokens": tokens.total_tokens,
            # Codex does not report a price; derive it from tokens offline.
            "raw_cost_cents": None,
            "charged_cents": None,
        },
    )


def run_coding_agent(model_id: str, workspace: Path, prompt: str) -> AgentRun:
    if uses_anthropic(model_id):
        return run_anthropic_agent(model_id, workspace, prompt)
    if uses_openai(model_id):
        return run_openai_agent(model_id, workspace, prompt)
    return run_cursor_agent(model_id, workspace, prompt)


def run_cursor_agent(model_id: str, workspace: Path, prompt: str) -> AgentRun:
    from cursor_sdk import Agent, LocalAgentOptions

    api_key = (os.environ.get("CURSOR_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "CURSOR_API_KEY is missing. Add it to KernelAgentBench/.env "
            "(see .env.example) or export it in your shell."
        )

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
