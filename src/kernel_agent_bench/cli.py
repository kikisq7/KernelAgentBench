from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Annotated, Any

import typer

from .config import canonical_hash, load_orcd, load_project_env, load_study, project_root
from .graph import TrialGraph
from .models import discover_models, disjoint_token_usage
from .orcd import (
    candidate_id,
    ensure_held_salloc,
    probe_ssh,
    release_held_salloc,
    render_slurm,
    ssh_handoff_commands,
)
from .records import read_jsonl
from .schemas import StudyConfig, json_schema_bundle
from .tracing import configure_tracing

app = typer.Typer(no_args_is_help=True, help="Julia kernel coding-agent benchmark.")


@app.callback()
def _load_env() -> None:
    load_project_env()


def _study(path: Path | None) -> tuple[Path, StudyConfig]:
    selected = path or project_root() / "config" / "study.yaml"
    return selected, load_study(selected)


@app.command("validate-config")
def validate_config(
    study_path: Annotated[Path | None, typer.Option("--study")] = None,
    orcd_path: Annotated[Path | None, typer.Option("--orcd")] = None,
) -> None:
    selected, study = _study(study_path)
    typer.echo(f"study: {selected} ({canonical_hash(study)})")
    if orcd_path:
        orcd = load_orcd(orcd_path)
        typer.echo(f"orcd: {orcd_path} ({canonical_hash(orcd)})")


@app.command("write-schemas")
def write_schemas(
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    output = output or project_root() / "schemas.json"
    output.write_text(json.dumps(json_schema_bundle(), indent=2, sort_keys=True) + "\n")
    typer.echo(output)


@app.command("models")
def models(study_path: Annotated[Path | None, typer.Option("--study")] = None) -> None:
    _, study = _study(study_path)
    typer.echo(json.dumps(discover_models(study.models), indent=2, sort_keys=True))


@app.command("start")
def start(
    model_alias: Annotated[str, typer.Option("--model")],
    language: Annotated[str, typer.Option("--language")],
    replicate: Annotated[int, typer.Option("--replicate", min=1)] = 1,
    study_path: Annotated[Path | None, typer.Option("--study")] = None,
    orcd_path: Annotated[Path | None, typer.Option("--orcd")] = None,
    baseline_commit: Annotated[str, typer.Option("--baseline")] = "HEAD",
) -> None:
    configure_tracing()
    _, study = _study(study_path)
    selectors = [model for model in study.models if model.alias == model_alias]
    if not selectors:
        aliases = ", ".join(model.alias for model in study.models)
        raise typer.BadParameter(f"unknown model alias; choose from {aliases}")
    if selectors[0].provider == "anthropic":
        if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
            raise typer.BadParameter(
                "ANTHROPIC_API_KEY is missing from KernelAgentBench/.env. "
                "Claude trials call the Anthropic API and do not use CURSOR_API_KEY."
            )
        os.environ["KAB_ANTHROPIC_EFFORT"] = selectors[0].effort
    if selectors[0].provider == "openai":
        if not (os.environ.get("OPENAI_API_KEY") or "").strip():
            raise typer.BadParameter(
                "OPENAI_API_KEY is missing from KernelAgentBench/.env. "
                "GPT trials call the OpenAI API and do not use CURSOR_API_KEY."
            )
        os.environ["KAB_OPENAI_EFFORT"] = selectors[0].effort
    resolved = discover_models(selectors)
    orcd = load_orcd(orcd_path) if orcd_path else None
    if orcd and orcd.auto_evaluate:
        probe_ssh(orcd)
    language_ids = {target.id for target in study.languages}
    if language not in language_ids:
        raise typer.BadParameter(f"unknown language; choose from {', '.join(sorted(language_ids))}")
    trial_id = f"{study.study_id}-{language}-{model_alias}-r{replicate}"
    runner = TrialGraph(project_root())
    try:
        state = runner.start(
            study=study,
            study_hash=canonical_hash(study),
            trial_id=trial_id,
            model_alias=model_alias,
            resolved_model_id=resolved[model_alias],
            language=language,
            replicate=replicate,
            baseline_commit=baseline_commit,
            orcd=orcd,
        )
    finally:
        runner.close()
    typer.echo(json.dumps(state, indent=2, sort_keys=True))


@app.command("matrix")
def matrix(study_path: Annotated[Path | None, typer.Option("--study")] = None) -> None:
    _, study = _study(study_path)
    rows = [
        {
            "trial_id": f"{study.study_id}-{language.id}-{model.alias}-r{replicate}",
            "language": language.id,
            "backend": language.backend,
            "model": model.alias,
            "replicate": replicate,
        }
        for language in study.languages
        for model in study.models
        for replicate in range(1, study.replicates + 1)
    ]
    typer.echo(json.dumps(rows, indent=2))


@app.command("resume")
def resume(
    trial_id: Annotated[str, typer.Option("--trial")],
    result: Annotated[Path, typer.Option("--result", exists=True, dir_okay=False)],
) -> None:
    configure_tracing()
    runner = TrialGraph(project_root())
    try:
        state = runner.resume(trial_id, result)
    finally:
        runner.close()
    typer.echo(json.dumps(state, indent=2, sort_keys=True))


@app.command("continue")
def continue_trial(trial_id: Annotated[str, typer.Option("--trial")]) -> None:
    """Re-run an interrupted step, such as an H200 eval, without redoing agent turns."""
    configure_tracing()
    runner = TrialGraph(project_root())
    try:
        state = runner.continue_trial(trial_id)
    finally:
        runner.close()
    typer.echo(json.dumps(state, indent=2, sort_keys=True))


@app.command("interrupted")
def interrupted(trial_id: Annotated[str, typer.Option("--trial")]) -> None:
    """Exit 0 if the trial has an interrupted step that `continue` can resume."""
    runner = TrialGraph(project_root())
    try:
        nodes = runner.pending_nodes(trial_id)
    finally:
        runner.close()
    if not nodes:
        raise typer.Exit(1)
    typer.echo(",".join(nodes))


@app.command("hold-h200")
def hold_h200(
    orcd_path: Annotated[Path, typer.Option("--orcd", exists=True, dir_okay=False)],
) -> None:
    """Request (or keep) one H200 allocation that every eval reuses via srun."""
    config = ensure_held_salloc(load_orcd(orcd_path), project_root())
    typer.echo(config.salloc_job_id)


@app.command("release-h200")
def release_h200(
    orcd_path: Annotated[Path, typer.Option("--orcd", exists=True, dir_okay=False)],
) -> None:
    job_id = release_held_salloc(load_orcd(orcd_path), project_root())
    typer.echo(f"released {job_id}" if job_id else "no held allocation")


@app.command("status")
def status(trial_id: Annotated[str, typer.Option("--trial")]) -> None:
    runner = TrialGraph(project_root())
    try:
        state = runner.state(trial_id)
    finally:
        runner.close()
    typer.echo(json.dumps(state, indent=2, sort_keys=True))


@app.command("pending")
def pending() -> None:
    evaluated: set[str] = set()
    evaluation_log = project_root() / "results" / "evaluation_events.jsonl"
    if evaluation_log.exists():
        for row in read_jsonl([evaluation_log]):
            evaluation = row.get("evaluation")
            if isinstance(evaluation, dict) and evaluation.get("candidate_id"):
                evaluated.add(evaluation["candidate_id"])
            elif row.get("event") == "evaluation_ingested" and row.get("candidate_id"):
                evaluated.add(row["candidate_id"])
    for manifest_path in sorted((project_root() / "candidates").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["candidate_id"] not in evaluated:
            typer.echo(str(manifest_path.parent))


@app.command("render-slurm")
def render_slurm_command(
    bundle: Annotated[Path, typer.Option("--bundle", exists=True, file_okay=False)],
    orcd_path: Annotated[Path, typer.Option("--orcd", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    result: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    config = load_orcd(orcd_path)
    selected_result = result or project_root() / "results" / f"{candidate_id(bundle)}.json"
    output.write_text(
        render_slurm(config, project=project_root(), bundle=bundle, result=selected_result),
        encoding="utf-8",
    )
    output.chmod(0o750)
    typer.echo(output)
    typer.echo("sbatch --parsable " + str(output))


@app.command("ssh-check")
def ssh_check(
    orcd_path: Annotated[Path, typer.Option("--orcd", exists=True, dir_okay=False)],
) -> None:
    config = load_orcd(orcd_path)
    probe_ssh(config)
    typer.echo(f"ssh ok: {config.ssh_user}@{config.ssh_host}")


@app.command("ssh-handoff")
def ssh_handoff(
    bundle: Annotated[Path, typer.Option("--bundle", exists=True, file_okay=False)],
    orcd_path: Annotated[Path, typer.Option("--orcd", exists=True, dir_okay=False)],
) -> None:
    config = load_orcd(orcd_path)
    typer.echo(
        "\n".join(
            ssh_handoff_commands(
                config,
                bundle=bundle,
                local_project=project_root(),
            )
        )
    )


@app.command("summary")
def summary(
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    source = project_root() / "results" / "trials.jsonl"
    if not source.exists():
        raise typer.BadParameter("no completed trials")
    rows = read_jsonl([source])
    runs_path = source.parent / "agent_runs.jsonl"
    turns: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    if runs_path.exists():
        for run in read_jsonl([runs_path]):
            if run.get("event") == "agent_run" and run.get("token_usage"):
                usage = disjoint_token_usage(run["token_usage"], run["resolved_model_id"])
                turns.setdefault(run["trial_id"], []).append((int(run["iteration"]), usage))

    def token_columns(row: dict[str, Any]) -> dict[str, Any]:
        usages = turns.get(row["trial_id"])
        per_turn = bool(usages)
        if not usages:
            usage = disjoint_token_usage(row["token_usage"], row["resolved_model_id"])
            usages = [(0, usage)]
        first_correct = row.get("first_correct_iteration")
        columns: dict[str, Any] = {
            field: sum(int(usage.get(field) or 0) for _, usage in usages)
            for field in (
                "input_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "output_tokens",
                "reasoning_tokens",
                "total_tokens",
            )
        }
        columns["recorded_total_tokens"] = row["token_usage"]["total_tokens"]
        columns["tokens_to_correctness"] = (
            sum(
                int(usage["total_tokens"])
                for iteration, usage in usages
                if iteration <= int(first_correct)
            )
            if first_correct is not None and per_turn
            else row.get("tokens_to_correctness")
        )
        return columns

    flattened = [
        {
            "study_id": row["study_id"],
            "trial_id": row["trial_id"],
            "model_alias": row["model_alias"],
            "resolved_model_id": row["resolved_model_id"],
            "language": row["language"],
            "replicate": row["replicate"],
            "speedup": row.get("incumbent_speedup"),
            "h200_evaluations": row["h200_evaluations"],
            **token_columns(row),
            "first_correct_iteration": row.get("first_correct_iteration"),
            "termination_reason": row.get("termination_reason"),
        }
        for row in rows
    ]
    if output is None or output.suffix == ".json":
        payload = json.dumps(flattened, indent=2, sort_keys=True)
        if output:
            output.write_text(payload + "\n", encoding="utf-8")
        else:
            typer.echo(payload)
        return
    if output.suffix != ".csv":
        raise typer.BadParameter("summary output must end in .json or .csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


if __name__ == "__main__":
    app()
