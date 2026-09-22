from __future__ import annotations

import json
import math
import shlex
import sqlite3
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict, cast

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .candidates import CandidateManager, Worktree
from .models import AgentRun, run_cursor_agent
from .orcd import evaluate_candidate_remote
from .records import append_jsonl, sha256_file
from .schemas import (
    EvaluationResult,
    LanguageTarget,
    OrcdConfig,
    StudyConfig,
    TokenRecord,
    TrialRecord,
)
from .tracing import add_trace_outputs, trace_span

ModelRunner = Callable[[str, Path, str], AgentRun]
EvaluateRunner = Callable[[OrcdConfig, Path, Path], Path]


class TrialState(TypedDict, total=False):
    study: dict[str, Any]
    study_hash: str
    graph_hash: str
    trial_id: str
    model_alias: str
    resolved_model_id: str
    language: str
    replicate: int
    seed: int
    started_monotonic: float
    baseline_commit: str
    worktree_root: str
    workspace: str
    status: str
    iteration: int
    h200_evaluations: int
    total_tokens: int
    token_usage: dict[str, Any]
    prompt: str
    response: str
    agent_id: str
    cursor_run_id: str
    billed_cost: dict[str, Any] | None
    public_test: dict[str, Any]
    candidate_id: str
    candidate_bundle: str
    parent_candidate_id: str | None
    evaluation_result_path: str
    evaluation: dict[str, Any]
    incumbent_candidate_id: str | None
    incumbent_speedup: float | None
    termination_reason: str | None
    last_feedback: str
    orcd: dict[str, Any] | None
    auto_evaluate: bool


def _sum_tokens(current: dict[str, Any], new: TokenRecord) -> TokenRecord:
    previous = TokenRecord.model_validate(current) if current else TokenRecord(available=True)
    return TokenRecord(
        input_tokens=previous.input_tokens + new.input_tokens,
        output_tokens=previous.output_tokens + new.output_tokens,
        cache_read_tokens=previous.cache_read_tokens + new.cache_read_tokens,
        cache_write_tokens=previous.cache_write_tokens + new.cache_write_tokens,
        total_tokens=previous.total_tokens + new.total_tokens,
        reasoning_tokens=(previous.reasoning_tokens or 0) + (new.reasoning_tokens or 0),
        available=previous.available and new.available,
    )


class TrialGraph:
    def __init__(
        self,
        project: Path,
        *,
        model_runner: ModelRunner = run_cursor_agent,
        evaluate_runner: EvaluateRunner = evaluate_candidate_remote,
        checkpoint_path: Path | None = None,
    ) -> None:
        self.project = project.resolve()
        self.manager = CandidateManager(self.project)
        self.model_runner = model_runner
        self.evaluate_runner = evaluate_runner
        checkpoint = checkpoint_path or self.project / ".kernel-agent-bench" / "checkpoints.sqlite"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(checkpoint, check_same_thread=False)
        self.checkpointer = SqliteSaver(self.connection)
        self.graph = self._build()

    def close(self) -> None:
        self.connection.close()

    def _build(self) -> Any:
        builder = StateGraph(TrialState)
        builder.add_node("run_agent", self._run_agent)
        builder.add_node("public_checks", self._public_checks)
        builder.add_node("package_candidate", self._package_candidate)
        builder.add_node("evaluate_h200", self._evaluate_h200)
        builder.add_node("ingest_result", self._ingest_result)
        builder.add_node("select_candidate", self._select_candidate)
        builder.add_node("finalize", self._finalize)
        builder.add_conditional_edges(
            START,
            self._route_start,
            {
                "run_agent": "run_agent",
                "ingest_result": "ingest_result",
                "finalize": "finalize",
            },
        )
        builder.add_edge("run_agent", "public_checks")
        builder.add_conditional_edges(
            "public_checks",
            self._route_after_checks,
            {
                "package_candidate": "package_candidate",
                "run_agent": "run_agent",
                "finalize": "finalize",
            },
        )
        builder.add_conditional_edges(
            "package_candidate",
            self._route_after_package,
            {"evaluate_h200": "evaluate_h200", "wait": END},
        )
        builder.add_edge("evaluate_h200", "ingest_result")
        builder.add_edge("ingest_result", "select_candidate")
        builder.add_conditional_edges(
            "select_candidate",
            self._route_after_selection,
            {"run_agent": "run_agent", "finalize": "finalize"},
        )
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _route_start(state: TrialState) -> str:
        if state.get("status") == "awaiting_h200_result":
            if not state.get("evaluation_result_path"):
                raise ValueError("trial is waiting for --result")
            return "ingest_result"
        if state.get("status") == "finished":
            return "finalize"
        return "run_agent"

    @staticmethod
    def _route_after_selection(state: TrialState) -> str:
        return "finalize" if state.get("termination_reason") else "run_agent"

    @staticmethod
    def _route_after_checks(state: TrialState) -> str:
        if state.get("termination_reason"):
            return "finalize"
        return "package_candidate" if state.get("status") == "packaging" else "run_agent"

    @staticmethod
    def _route_after_package(state: TrialState) -> str:
        return "evaluate_h200" if state.get("auto_evaluate") else "wait"

    def _metadata(self, state: TrialState) -> dict[str, Any]:
        return {
            "study_id": state["study"]["study_id"],
            "task_version": state["study"]["task_version"],
            "trial_id": state["trial_id"],
            "model_id": state["resolved_model_id"],
            "language": state["language"],
            "iteration": state["iteration"],
            "baseline_commit": state["baseline_commit"],
            "graph_hash": state["graph_hash"],
        }

    @staticmethod
    def _target(study: StudyConfig, language: str) -> LanguageTarget:
        for target in study.languages:
            if target.id == language:
                return target
        raise ValueError(f"unknown language {language!r}")

    def _run_agent(self, state: TrialState) -> dict[str, Any]:
        study = StudyConfig.model_validate(state["study"])
        feedback = ""
        if state.get("evaluation"):
            feedback = "\n\nPrevious H200 evaluation:\n" + json.dumps(
                state["evaluation"], sort_keys=True, indent=2
            )
        if state.get("last_feedback"):
            feedback += "\n\nLocal validation feedback:\n" + state["last_feedback"]
        prompt = state["prompt"] + feedback
        with trace_span(
            "cursor_coding_agent",
            inputs={"prompt": prompt, "workspace": state["workspace"]},
            metadata=self._metadata(state),
            include_content=study.trace_content,
            run_type="llm",
        ) as span:
            run = self.model_runner(
                state["resolved_model_id"],
                Path(state["workspace"]),
                prompt,
            )
            cumulative = _sum_tokens(state.get("token_usage", {}), run.usage)
            outputs = {
                "status": run.status,
                "response": run.response,
                "agent_id": run.agent_id,
                "run_id": run.run_id,
                "token_usage": run.usage.model_dump(),
                "cost": run.cost,
            }
            if span is not None:
                span.add_metadata(
                    {
                        "token_input": run.usage.input_tokens,
                        "token_output": run.usage.output_tokens,
                        "token_cache_read": run.usage.cache_read_tokens,
                        "token_cache_write": run.usage.cache_write_tokens,
                        "token_total": run.usage.total_tokens,
                        "token_reasoning": run.usage.reasoning_tokens,
                        "token_usage_available": run.usage.available,
                    }
                )
            add_trace_outputs(span, outputs, study.trace_content)
        if run.status not in {"finished", "success"}:
            raise RuntimeError(f"Cursor run failed with status {run.status}: {run.response}")
        return {
            "status": "checking",
            "response": run.response,
            "agent_id": run.agent_id,
            "cursor_run_id": run.run_id,
            "billed_cost": run.cost,
            "token_usage": cumulative.model_dump(),
            "total_tokens": cumulative.total_tokens,
        }

    def _public_checks(self, state: TrialState) -> dict[str, Any]:
        study = StudyConfig.model_validate(state["study"])
        target = self._target(study, state["language"])
        command = shlex.split(target.public_test_command)
        validation_error = ""
        try:
            self.manager.validate_changes(self._worktree(state), target.editable_paths)
        except ValueError as exception:
            validation_error = str(exception)
        with trace_span(
            "public_checks",
            inputs={"command": command},
            metadata=self._metadata(state),
            include_content=study.trace_content,
            run_type="tool",
        ) as span:
            completed = subprocess.run(
                command,
                cwd=state["workspace"],
                text=True,
                capture_output=True,
                timeout=min(study.budgets.max_wall_seconds, 1800),
            )
            result = {
                "command": command,
                "exit_code": completed.returncode or (2 if validation_error else 0),
                "stdout": completed.stdout[-50_000:],
                "stderr": (completed.stderr + "\n" + validation_error).strip()[-50_000:],
            }
            add_trace_outputs(span, result, study.trace_content)
        if result["exit_code"]:
            next_iteration = state["iteration"] + 1
            reason = self._budget_reason(state, next_iteration, study)
            return {
                "status": "finishing" if reason else "optimizing",
                "public_test": result,
                "iteration": next_iteration,
                "last_feedback": str(result["stderr"])[-4000:],
                "termination_reason": reason,
            }
        return {"status": "packaging", "public_test": result, "last_feedback": ""}

    def _worktree(self, state: TrialState) -> Worktree:
        return Worktree(
            repository=self.manager.repository,
            root=Path(state["worktree_root"]),
            project=Path(state["workspace"]),
            baseline_commit=state["baseline_commit"],
        )

    def _package_candidate(self, state: TrialState) -> dict[str, Any]:
        study = StudyConfig.model_validate(state["study"])
        target = self._target(study, state["language"])
        with trace_span(
            "package_candidate",
            inputs={"iteration": state["iteration"]},
            metadata=self._metadata(state),
            include_content=study.trace_content,
            run_type="tool",
        ) as span:
            manifest, bundle = self.manager.package(
                self._worktree(state),
                state["trial_id"],
                state["iteration"],
                target.editable_paths,
                state.get("parent_candidate_id"),
                state["language"],
                study.task_version,
                state["study_hash"],
                state["prompt"],
                study.evaluation,
            )
            add_trace_outputs(
                span,
                {
                    "candidate_id": manifest.candidate_id,
                    "patch_sha256": manifest.patch_sha256,
                },
                study.trace_content,
            )
        append_jsonl(
            self.project / "results" / "candidate_events.jsonl",
            {
                "event": "candidate_packaged",
                "manifest": manifest.model_dump(mode="json"),
                "agent_id": state["agent_id"],
                "cursor_run_id": state["cursor_run_id"],
                "token_usage": state["token_usage"],
                "billed_cost": state.get("billed_cost"),
            },
        )
        return {
            "status": "awaiting_h200_result",
            "candidate_id": manifest.candidate_id,
            "candidate_bundle": str(bundle),
            "parent_candidate_id": manifest.candidate_id,
            "evaluation_result_path": "",
            "last_feedback": "",
        }

    def _evaluate_h200(self, state: TrialState) -> dict[str, Any]:
        if not state.get("orcd"):
            raise ValueError("auto H200 evaluation requires an ORCD config")
        orcd = OrcdConfig.model_validate(state["orcd"])
        study = StudyConfig.model_validate(state["study"])
        with trace_span(
            "evaluate_h200",
            inputs={
                "candidate_id": state["candidate_id"],
                "salloc": " ".join(
                    [
                        "salloc",
                        "-N",
                        str(orcd.nodes),
                        "-n",
                        str(orcd.ntasks),
                        "--cpus-per-task",
                        str(orcd.cpus_per_task),
                        "--mem",
                        orcd.memory,
                        "--time",
                        orcd.wall_time,
                        "-G",
                        orcd.gpu_request,
                    ]
                ),
            },
            metadata=self._metadata(state),
            include_content=study.trace_content,
            run_type="tool",
        ) as span:
            result_path = self.evaluate_runner(
                orcd,
                Path(state["candidate_bundle"]),
                self.project,
            )
            add_trace_outputs(
                span,
                {"result_path": str(result_path)},
                study.trace_content,
            )
        return {
            "status": "awaiting_h200_result",
            "evaluation_result_path": str(result_path),
        }

    def _ingest_result(self, state: TrialState) -> dict[str, Any]:
        study = StudyConfig.model_validate(state["study"])
        path = Path(state["evaluation_result_path"])
        with trace_span(
            "ingest_h200_result",
            inputs={"result_path": str(path), "candidate_id": state["candidate_id"]},
            metadata=self._metadata(state),
            include_content=study.trace_content,
            run_type="tool",
        ) as span:
            data = json.loads(path.read_text(encoding="utf-8"))
            result = EvaluationResult.model_validate(data)
            if result.candidate_id != state["candidate_id"]:
                raise ValueError(
                    f"result candidate {result.candidate_id} does not match {state['candidate_id']}"
                )
            if result.language != state["language"]:
                raise ValueError(
                    f"result language {result.language} does not match {state['language']}"
                )
            result_hash = sha256_file(path)
            evaluation = result.model_dump(mode="json")
            evaluation["result_sha256"] = result_hash
            add_trace_outputs(
                span,
                {
                    "correct": result.correct,
                    "speedup": result.geometric_mean_speedup,
                    "result_sha256": result_hash,
                },
                study.trace_content,
            )
        append_jsonl(
            self.project / "results" / "evaluation_events.jsonl",
            {"event": "evaluation_ingested", "evaluation": evaluation},
        )
        return {
            "status": "selecting",
            "evaluation": evaluation,
            "h200_evaluations": state["h200_evaluations"] + 1,
            "evaluation_result_path": "",
        }

    def _select_candidate(self, state: TrialState) -> dict[str, Any]:
        study = StudyConfig.model_validate(state["study"])
        target = self._target(study, state["language"])
        evaluation = EvaluationResult.model_validate(state["evaluation"])
        incumbent_speedup = state.get("incumbent_speedup")
        incumbent_candidate = state.get("incumbent_candidate_id")
        improved = False
        if evaluation.correct and evaluation.geometric_mean_speedup is not None:
            reference = incumbent_speedup if incumbent_speedup is not None else 1.0
            required = reference * (1 + study.evaluation.min_relative_improvement)
            if incumbent_speedup is None:
                if evaluation.geometric_mean_speedup >= 1.0:
                    incumbent_speedup = evaluation.geometric_mean_speedup
                    incumbent_candidate = evaluation.candidate_id
            elif evaluation.geometric_mean_speedup > incumbent_speedup:
                incumbent_speedup = evaluation.geometric_mean_speedup
                incumbent_candidate = evaluation.candidate_id
            improved = evaluation.geometric_mean_speedup >= required

        if incumbent_candidate != evaluation.candidate_id:
            self.manager.restore_candidate(
                self._worktree(state),
                target.editable_paths,
                incumbent_candidate,
            )

        next_iteration = state["iteration"] + 1
        reason = self._budget_reason(state, next_iteration, study)
        if reason is None and evaluation.correct and not improved:
            reason = "no_significant_improvement"
        selected = {
            "status": "finishing" if reason else "optimizing",
            "iteration": next_iteration,
            "incumbent_candidate_id": incumbent_candidate,
            "incumbent_speedup": incumbent_speedup,
            "termination_reason": reason,
        }
        with trace_span(
            "select_candidate",
            inputs={
                "candidate_id": evaluation.candidate_id,
                "correct": evaluation.correct,
                "speedup": evaluation.geometric_mean_speedup,
            },
            metadata=self._metadata(state),
            include_content=study.trace_content,
        ) as span:
            add_trace_outputs(span, selected, study.trace_content)
        return selected

    @staticmethod
    def _budget_reason(
        state: TrialState,
        next_iteration: int,
        study: StudyConfig,
    ) -> str | None:
        elapsed = time.monotonic() - state["started_monotonic"]
        if next_iteration > study.budgets.max_agent_iterations:
            return "agent_iteration_budget"
        if state["h200_evaluations"] >= study.budgets.max_h200_evaluations:
            return "h200_evaluation_budget"
        if state["total_tokens"] >= study.budgets.max_total_tokens:
            return "token_budget"
        if elapsed >= study.budgets.max_wall_seconds:
            return "wall_time_budget"
        return None

    def _finalize(self, state: TrialState) -> dict[str, Any]:
        token_usage = TokenRecord.model_validate(state.get("token_usage", {}))
        record = TrialRecord(
            study_id=state["study"]["study_id"],
            trial_id=state["trial_id"],
            model_alias=state["model_alias"],
            resolved_model_id=state["resolved_model_id"],
            language=state["language"],
            replicate=state["replicate"],
            status="finished",
            iteration=state["iteration"],
            h200_evaluations=state["h200_evaluations"],
            token_usage=token_usage,
            incumbent_candidate_id=state.get("incumbent_candidate_id"),
            incumbent_speedup=state.get("incumbent_speedup"),
            termination_reason=state.get("termination_reason") or "already_finished",
            metadata={"study_hash": state["study_hash"], "graph_hash": state["graph_hash"]},
        )
        append_jsonl(self.project / "results" / "trials.jsonl", record)
        self.manager.cleanup_worktree(Path(state["worktree_root"]))
        return {"status": "finished"}

    def start(
        self,
        *,
        study: StudyConfig,
        study_hash: str,
        trial_id: str,
        model_alias: str,
        resolved_model_id: str,
        language: str,
        replicate: int,
        baseline_commit: str = "HEAD",
        orcd: OrcdConfig | None = None,
    ) -> TrialState:
        worktree = self.manager.create_worktree(trial_id, baseline_commit)
        target = self._target(study, language)
        prompt = (worktree.project / target.prompt_path).read_text(encoding="utf-8")
        initial: TrialState = {
            "study": study.model_dump(mode="json"),
            "study_hash": study_hash,
            "graph_hash": sha256_file(Path(__file__)),
            "trial_id": trial_id,
            "model_alias": model_alias,
            "resolved_model_id": resolved_model_id,
            "language": language,
            "replicate": replicate,
            "seed": study.seed + replicate,
            "started_monotonic": time.monotonic(),
            "baseline_commit": worktree.baseline_commit,
            "worktree_root": str(worktree.root),
            "workspace": str(worktree.project),
            "status": "new",
            "iteration": 1,
            "h200_evaluations": 0,
            "total_tokens": 0,
            "token_usage": TokenRecord().model_dump(),
            "prompt": prompt,
            "parent_candidate_id": None,
            "incumbent_candidate_id": None,
            "incumbent_speedup": None,
            "termination_reason": None,
            "last_feedback": "",
            "orcd": orcd.model_dump(mode="json") if orcd else None,
            "auto_evaluate": bool(orcd and orcd.auto_evaluate),
        }
        metadata: dict[str, Any] = {
            "study_id": study.study_id,
            "task_version": study.task_version,
            "trial_id": trial_id,
            "model_id": resolved_model_id,
            "language": language,
            "baseline_commit": worktree.baseline_commit,
            "study_hash": study_hash,
        }
        config: dict[str, Any] = {
            "configurable": {"thread_id": trial_id},
            # Node spans are emitted explicitly so trace_content=false cannot
            # leak the full LangGraph state through automatic callbacks.
            "callbacks": [],
            "metadata": metadata,
            "run_name": "kernel_agent_trial",
        }
        with trace_span(
            "kernel_agent_trial",
            inputs={
                "trial_id": trial_id,
                "model_id": resolved_model_id,
                "baseline_commit": worktree.baseline_commit,
            },
            metadata=metadata,
            include_content=study.trace_content,
        ) as span:
            result = cast(TrialState, self.graph.invoke(initial, config=config))
            add_trace_outputs(
                span,
                {"status": result["status"], "candidate_id": result.get("candidate_id")},
                study.trace_content,
            )
            return result

    def resume(self, trial_id: str, result_path: Path) -> TrialState:
        config = {"configurable": {"thread_id": trial_id}}
        snapshot = self.graph.get_state(config)
        if not snapshot.values:
            raise KeyError(f"unknown trial: {trial_id}")
        if snapshot.values.get("status") != "awaiting_h200_result":
            raise ValueError(f"trial is not awaiting a result: {snapshot.values.get('status')}")
        values = cast(TrialState, snapshot.values)
        study = StudyConfig.model_validate(values["study"])
        metadata = self._metadata(values)
        with trace_span(
            "kernel_agent_trial_resume",
            inputs={"trial_id": trial_id, "result_path": str(result_path)},
            metadata=metadata,
            include_content=study.trace_content,
        ) as span:
            result = cast(
                TrialState,
                self.graph.invoke(
                    {"evaluation_result_path": str(result_path.resolve())},
                    config={
                        **config,
                        "callbacks": [],
                        "metadata": metadata,
                        "run_name": "kernel_agent_trial_resume",
                    },
                ),
            )
            add_trace_outputs(
                span,
                {"status": result["status"], "candidate_id": result.get("candidate_id")},
                study.trace_content,
            )
            return result

    def state(self, trial_id: str) -> TrialState:
        snapshot = self.graph.get_state({"configurable": {"thread_id": trial_id}})
        if not snapshot.values:
            raise KeyError(f"unknown trial: {trial_id}")
        return cast(TrialState, dict(snapshot.values))


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))
