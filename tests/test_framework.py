from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kernel_agent_bench.config import canonical_hash
from kernel_agent_bench.graph import TrialGraph, geometric_mean
from kernel_agent_bench.models import AgentRun, resolve_model_ids
from kernel_agent_bench.orcd import (
    remote_transfer,
    render_slurm,
    rsync_download_command,
    rsync_upload_command,
    salloc_command,
    ssh_handoff_commands,
)
from kernel_agent_bench.schemas import (
    Budgets,
    CandidateManifest,
    EvaluationConfig,
    EvaluationResult,
    LanguageTarget,
    ModelSelector,
    OrcdConfig,
    StudyConfig,
    TokenRecord,
    WorkloadResult,
)
from kernel_agent_bench.tracing import trace_payload


def test_model_resolution_uses_newest_natural_version() -> None:
    selectors = [
        ModelSelector(alias="composer", match="^composer"),
        ModelSelector(alias="anthropic", match="^claude"),
    ]
    available = ["composer-2.5", "composer-2.12", "claude-4.5", "claude-4.6"]
    assert resolve_model_ids(selectors, available) == {
        "composer": "composer-2.12",
        "anthropic": "claude-4.6",
    }


def test_model_resolution_rejects_duplicates_and_missing() -> None:
    with pytest.raises(ValueError, match="no Cursor model"):
        resolve_model_ids([ModelSelector(alias="gpt", match="^gpt-5.6")], ["gpt-5.5"])
    with pytest.raises(ValueError, match="duplicate"):
        resolve_model_ids(
            [
                ModelSelector(alias="one", match="composer"),
                ModelSelector(alias="two", match="composer"),
            ],
            ["composer-2.5"],
        )


def test_orcd_required_fields_and_salloc_flags() -> None:
    base = {
        "partition": "mit_normal_gpu",
        "gpu_request": "h200:1",
        "scratch_root": "REQUIRED",
        "ssh_host": "orcd-login.mit.edu",
        "ssh_user": "user",
        "remote_project": "/home/user/UROPJulia/KernelAgentBench",
    }
    with pytest.raises(ValueError, match="scratch_root"):
        OrcdConfig.model_validate(base)
    with pytest.raises(ValueError, match="KernelAgentBench"):
        OrcdConfig.model_validate(
            {**base, "scratch_root": "/scratch/user", "remote_project": "/home/user"}
        )
    config = OrcdConfig.model_validate({**base, "scratch_root": "/scratch/user"})
    assert salloc_command(config)[:14] == [
        "salloc",
        "-N",
        "1",
        "-n",
        "32",
        "--cpus-per-task",
        "1",
        "--mem",
        "128GB",
        "--time",
        "06:00:00",
        "-G",
        "h200:1",
        "-p",
    ]


@pytest.mark.parametrize(
    ("language", "expected_command"),
    [
        ("julia", "julia/evaluate_candidate.jl"),
        ("python", "python/evaluate_candidate.py"),
        ("cpp", "nvcc -O3"),
    ],
)
def test_slurm_renderer_freezes_candidate_evaluation(
    tmp_path: Path,
    language: str,
    expected_command: str,
) -> None:
    project = tmp_path / "repo" / "KernelAgentBench"
    bundle = project / "candidates" / "candidate-1"
    bundle.mkdir(parents=True)
    patch = bundle / "candidate.patch"
    patch.write_text("patch", encoding="utf-8")
    evaluation = _study().evaluation
    manifest = CandidateManifest(
        candidate_id="candidate-1",
        trial_id="trial-1",
        baseline_commit="abc123",
        patch_sha256="0" * 64,
        patch_file=patch.name,
        iteration=1,
        language=language,
        task_version="test",
        study_hash="1" * 64,
        prompt_sha256="2" * 64,
        evaluation=evaluation,
    )
    (bundle / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    config = OrcdConfig(
        partition="mit_normal_gpu",
        gpu_request="h200:1",
        nodes=1,
        ntasks=32,
        cpus_per_task=1,
        memory="128GB",
        wall_time="06:00:00",
        account="",
        scratch_root="/scratch/user",
        ssh_host="orcd-login.mit.edu",
        ssh_user="user",
        remote_project="/home/user/UROPJulia/KernelAgentBench",
        auto_evaluate=True,
    )
    script = render_slurm(
        config,
        project=project,
        bundle=bundle,
        result=project / "results" / "candidate-1.json",
    )
    assert "#SBATCH -G h200:1" in script
    assert "--correctness-shapes 5x5" in script
    assert "--steps 1" in script
    assert expected_command in script
    commands = ssh_handoff_commands(config, bundle=bundle, local_project=project)
    assert any(
        "salloc -N 1 -n 32 --cpus-per-task 1 --mem 128GB --time 06:00:00 -G h200:1" in command
        for command in commands
    )
    assert any("kernel-agent-bench resume --trial trial-1" in command for command in commands)
    rsync_lines = [command for command in commands if command.startswith("rsync ")]
    assert rsync_lines
    assert all(" --delete " not in f" {command} " for command in rsync_lines)


def test_rsync_stays_inside_one_candidate_directory(tmp_path: Path) -> None:
    project = tmp_path / "repo" / "KernelAgentBench"
    bundle = project / "candidates" / "candidate-1"
    bundle.mkdir(parents=True)
    (bundle / "candidate.patch").write_text("patch", encoding="utf-8")
    manifest = CandidateManifest(
        candidate_id="candidate-1",
        trial_id="trial-1",
        baseline_commit="abc123",
        patch_sha256="0" * 64,
        patch_file="candidate.patch",
        iteration=1,
        language="julia",
        task_version="test",
        study_hash="1" * 64,
        prompt_sha256="2" * 64,
        evaluation=_study().evaluation,
    )
    (bundle / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    config = OrcdConfig(
        scratch_root="/scratch/user",
        ssh_host="orcd-login.mit.edu",
        ssh_user="user",
        remote_project="/home/user/UROPJulia/KernelAgentBench",
    )
    transfer = remote_transfer(config, bundle, project)
    upload = rsync_upload_command(config, bundle, transfer)
    download = rsync_download_command(config, transfer)
    assert "--delete" not in upload
    assert upload[-2] == str(bundle.resolve()) + "/"
    assert upload[-1].endswith("/KernelAgentBench/candidates/candidate-1/")
    assert "/KernelAgentBench/candidates/candidate-1/" in upload[-1]
    assert upload[-1].count("/") >= 5
    assert not upload[-1].endswith("/KernelAgentBench/")
    assert download[-2].endswith("/KernelAgentBench/results/candidate-1.json")
    assert download[-1] == str(project.resolve() / "results" / "candidate-1.json")
    with pytest.raises(ValueError, match="candidate_id"):
        CandidateManifest.model_validate(
            {**manifest.model_dump(mode="json"), "candidate_id": "../escape"}
        )


def test_content_redaction() -> None:
    value = {"prompt": "secret source", "token_usage": {"total_tokens": 42}}
    assert trace_payload(value, include_content=False) == {
        "prompt": "[redacted]",
        "token_usage": {"total_tokens": 42},
    }
    assert trace_payload(value, include_content=True) == value


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, text=True)


def _study(
    language: str = "julia",
    editable_path: str = "kernels/gray_scott/candidate.jl",
    prompt_path: str = "prompts/task.md",
) -> StudyConfig:
    return StudyConfig(
        study_id="test-study",
        task_version="test-task-v1",
        seed=7,
        replicates=1,
        trace_content=False,
        models=[ModelSelector(alias="mock", match="mock")],
        budgets=Budgets(
            max_agent_iterations=1,
            max_h200_evaluations=1,
            max_total_tokens=100,
            max_wall_seconds=60,
        ),
        evaluation=EvaluationConfig(
            correctness_shapes=[(5, 5)],
            performance_shapes=[(8, 8)],
            steps=1,
            warmups=0,
            repetitions=1,
            atol=1e-5,
            rtol=1e-4,
            max_regression_fraction=0.03,
            min_relative_improvement=0.05,
        ),
        languages=[
            LanguageTarget(
                id=language,
                editable_paths=[editable_path],
                public_test_command="python3 -c pass",
                prompt_path=prompt_path,
            )
        ],
    )


@pytest.mark.parametrize(
    ("language", "editable_path"),
    [
        ("julia", "kernels/gray_scott/candidate.jl"),
        ("python", "python/gray_scott/candidate.py"),
        ("cpp", "cpp/gray_scott/candidate.cuh"),
    ],
)
def test_mocked_checkpointed_trial(
    tmp_path: Path,
    language: str,
    editable_path: str,
) -> None:
    repository = tmp_path / "repo"
    project = repository / "KernelAgentBench"
    candidate = project / editable_path
    prompt = project / "prompts" / "task.md"
    candidate.parent.mkdir(parents=True)
    prompt.parent.mkdir(parents=True)
    candidate.write_text("baseline = true\n", encoding="utf-8")
    prompt.write_text("optimize", encoding="utf-8")
    (project / ".gitignore").write_text(
        ".kernel-agent-bench/\ncandidates/\nresults/\n", encoding="utf-8"
    )
    _git(repository, "init")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "baseline")

    def fake_runner(model_id: str, workspace: Path, agent_prompt: str) -> AgentRun:
        assert model_id == "mock-1"
        assert agent_prompt.startswith("optimize")
        path = workspace / editable_path
        path.write_text("baseline = false\n", encoding="utf-8")
        return AgentRun(
            status="finished",
            agent_id="agent-test",
            run_id="run-test",
            response="changed candidate",
            usage=TokenRecord(input_tokens=3, output_tokens=2, total_tokens=5),
            cost={"charged_cents": 1.0},
        )

    study = _study(language, editable_path)
    graph = TrialGraph(project, model_runner=fake_runner)
    try:
        waiting = graph.start(
            study=study,
            study_hash=canonical_hash(study),
            trial_id="trial-1",
            model_alias="mock",
            resolved_model_id="mock-1",
            language=language,
            replicate=1,
        )
        assert waiting["status"] == "awaiting_h200_result"
        assert Path(waiting["candidate_bundle"], "candidate.patch").is_file()

        result = EvaluationResult(
            candidate_id=waiting["candidate_id"],
            language=language,
            evaluator_version="test",
            status="passed",
            correct=True,
            workloads=[
                WorkloadResult(
                    shape=(8, 8),
                    baseline_samples_ms=[2.0],
                    candidate_samples_ms=[1.0],
                    baseline_median_ms=2.0,
                    candidate_median_ms=1.0,
                    speedup=2.0,
                    correct=True,
                    max_abs_error=0,
                    relative_l2_error=0,
                )
            ],
            geometric_mean_speedup=2.0,
        )
        result_path = tmp_path / "result.json"
        result_path.write_text(result.model_dump_json(), encoding="utf-8")
        finished = graph.resume("trial-1", result_path)
        assert finished["status"] == "finished"
        assert finished["incumbent_speedup"] == 2.0
        assert finished["h200_evaluations"] == 1
    finally:
        graph.close()


def test_auto_salloc_evaluation_closes_the_loop(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    project = repository / "KernelAgentBench"
    candidate = project / "kernels" / "gray_scott" / "candidate.jl"
    prompt = project / "prompts" / "task.md"
    candidate.parent.mkdir(parents=True)
    prompt.parent.mkdir(parents=True)
    candidate.write_text("baseline = true\n", encoding="utf-8")
    prompt.write_text("optimize", encoding="utf-8")
    (project / ".gitignore").write_text(
        ".kernel-agent-bench/\ncandidates/\nresults/\n", encoding="utf-8"
    )
    _git(repository, "init")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "baseline")

    def fake_runner(model_id: str, workspace: Path, agent_prompt: str) -> AgentRun:
        del model_id, agent_prompt
        (workspace / "kernels" / "gray_scott" / "candidate.jl").write_text(
            "baseline = false\n", encoding="utf-8"
        )
        return AgentRun(
            status="finished",
            agent_id="agent-test",
            run_id="run-test",
            response="changed candidate",
            usage=TokenRecord(input_tokens=3, output_tokens=2, total_tokens=5),
            cost=None,
        )

    def fake_evaluate(orcd: OrcdConfig, bundle: Path, local_project: Path) -> Path:
        assert orcd.gpu_request == "h200:1"
        assert orcd.ntasks == 32
        manifest = CandidateManifest.model_validate_json(
            (bundle / "manifest.json").read_text(encoding="utf-8")
        )
        result = EvaluationResult(
            candidate_id=manifest.candidate_id,
            language="julia",
            evaluator_version="test",
            status="passed",
            correct=True,
            workloads=[],
            geometric_mean_speedup=1.02,
        )
        path = local_project / "results" / f"{manifest.candidate_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(), encoding="utf-8")
        return path

    orcd = OrcdConfig(
        scratch_root="/scratch/user",
        ssh_host="orcd-login.mit.edu",
        ssh_user="user",
        remote_project="/home/user/UROPJulia/KernelAgentBench",
        auto_evaluate=True,
    )
    graph = TrialGraph(project, model_runner=fake_runner, evaluate_runner=fake_evaluate)
    try:
        finished = graph.start(
            study=_study(),
            study_hash=canonical_hash(_study()),
            trial_id="trial-auto",
            model_alias="mock",
            resolved_model_id="mock-1",
            language="julia",
            replicate=1,
            orcd=orcd,
        )
        assert finished["status"] == "finished"
        assert finished["h200_evaluations"] == 1
        assert finished["auto_evaluate"] is True
    finally:
        graph.close()


def test_stops_early_when_speedup_is_not_significant(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    project = repository / "KernelAgentBench"
    candidate = project / "kernels" / "gray_scott" / "candidate.jl"
    prompt = project / "prompts" / "task.md"
    candidate.parent.mkdir(parents=True)
    prompt.parent.mkdir(parents=True)
    candidate.write_text("baseline = true\n", encoding="utf-8")
    prompt.write_text("optimize", encoding="utf-8")
    (project / ".gitignore").write_text(
        ".kernel-agent-bench/\ncandidates/\nresults/\n", encoding="utf-8"
    )
    _git(repository, "init")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "baseline")

    def fake_runner(model_id: str, workspace: Path, agent_prompt: str) -> AgentRun:
        del model_id, agent_prompt
        path = workspace / "kernels" / "gray_scott" / "candidate.jl"
        path.write_text("baseline = false\n", encoding="utf-8")
        return AgentRun(
            status="finished",
            agent_id="agent-test",
            run_id="run-test",
            response="changed candidate",
            usage=TokenRecord(input_tokens=3, output_tokens=2, total_tokens=5),
            cost=None,
        )

    study = _study()
    study = study.model_copy(
        update={
            "budgets": Budgets(
                max_agent_iterations=5,
                max_h200_evaluations=5,
                max_total_tokens=100000,
                max_wall_seconds=3600,
            )
        }
    )
    graph = TrialGraph(project, model_runner=fake_runner)
    try:
        waiting = graph.start(
            study=study,
            study_hash=canonical_hash(study),
            trial_id="trial-early-stop",
            model_alias="mock",
            resolved_model_id="mock-1",
            language="julia",
            replicate=1,
        )
        result = EvaluationResult(
            candidate_id=waiting["candidate_id"],
            language="julia",
            evaluator_version="test",
            status="passed",
            correct=True,
            workloads=[],
            geometric_mean_speedup=1.02,
        )
        result_path = tmp_path / "result.json"
        result_path.write_text(result.model_dump_json(), encoding="utf-8")
        finished = graph.resume("trial-early-stop", result_path)
        assert finished["status"] == "finished"
        assert finished["h200_evaluations"] == 1
        assert finished["termination_reason"] == "no_significant_improvement"
        assert finished["incumbent_speedup"] == 1.02
    finally:
        graph.close()


def test_budgets_are_capped_at_five_tries() -> None:
    with pytest.raises(ValueError, match="less than or equal to 5"):
        Budgets(
            max_agent_iterations=6,
            max_h200_evaluations=5,
            max_total_tokens=100,
            max_wall_seconds=60,
        )
    assert geometric_mean([2.0, 8.0]) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        geometric_mean([])
