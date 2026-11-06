from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from kernel_agent_bench.candidates import CandidateManager
from kernel_agent_bench.config import canonical_hash
from kernel_agent_bench.graph import TrialGraph, geometric_mean
from kernel_agent_bench.models import AgentRun, resolve_model_ids
from kernel_agent_bench.orcd import (
    _parse_salloc_job_id,
    _parse_sbatch_job_id,
    persist_salloc_job_id,
    remote_commit_command,
    remote_eval_command,
    remote_mkdir_command,
    remote_transfer,
    render_slurm,
    rsync_download_command,
    rsync_ssh,
    rsync_upload_command,
    salloc_command,
    ssh_control_path,
    ssh_handoff_commands,
)
from kernel_agent_bench.records import read_jsonl
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
from kernel_agent_bench.tracing import configure_tracing, trace_payload

JULIA_CORRECTNESS_CASES = [{"dtype": "Float32"}, {"dtype": "Float64"}]


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


def test_anthropic_models_skip_the_cursor_catalog() -> None:
    from kernel_agent_bench.models import anthropic_token_record, resolve_configured_models

    opus = ModelSelector(
        alias="opus-5.5",
        match="^claude-opus-5-5$",
        provider="anthropic",
        model_id="claude-opus-5-5",
        effort="high",
    )
    composer = ModelSelector(alias="composer-2.5", match=r"^composer-2\.5$")
    resolved = resolve_configured_models([opus, composer], ["composer-2.5"])
    assert resolved == {"opus-5.5": "claude-opus-5-5", "composer-2.5": "composer-2.5"}
    with pytest.raises(ValueError, match="literal model_id"):
        ModelSelector(alias="opus", match="^claude", provider="anthropic", effort="high")
    record = anthropic_token_record(
        None,
        {
            "claude-opus-5-5": {
                "inputTokens": 10,
                "outputTokens": 4,
                "cacheReadInputTokens": 3,
                "cacheCreationInputTokens": 2,
                "thinkingTokens": 7,
            }
        },
    )
    assert record.total_tokens == 19
    assert record.reasoning_tokens == 7


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
        "1",
        "--cpus-per-task",
        "1",
        "--mem",
        "8GB",
        "--time",
        "00:20:00",
        "-G",
        "h200:1",
        "-p",
    ]
    assert salloc_command(config, wall_time="04:00:00")[10] == "04:00:00"


def test_persist_and_parse_held_salloc(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "orcd.yaml").write_text('salloc_job_id: ""\n', encoding="utf-8")
    persist_salloc_job_id(tmp_path, "24592061")
    assert 'salloc_job_id: "24592061"' in (config_dir / "orcd.yaml").read_text(encoding="utf-8")
    assert _parse_salloc_job_id("salloc: Pending job allocation 24592061\n") == "24592061"
    assert _parse_salloc_job_id("salloc: Granted job allocation 99\n") == "99"


@pytest.mark.parametrize(
    ("language", "expected_command"),
    [
        ("julia-ka", "--backend ka"),
        ("julia-cpu", "--backend cpu"),
        ("julia-cuda", "--backend cuda"),
        ("python", "python/evaluate_candidate.py"),
        ("cpp", "g++ -O3 -std=c++17"),
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
        ntasks=1,
        cpus_per_task=1,
        memory="8GB",
        wall_time="00:20:00",
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
    assert "/System/" not in script
    assert "#SBATCH -G h200:1" in script
    assert "--correctness-shapes 5x5" in script
    assert "--steps 1" in script
    assert expected_command in script
    if language == "cpp":
        assert "--correctness-dtypes Float32,Float64" in script
    remote_script = render_slurm(
        config,
        project=Path("/home/user/UROPJulia/KernelAgentBench"),
        bundle=bundle,
        result=Path("/home/user/UROPJulia/KernelAgentBench/results/candidate-1.json"),
        repository=Path("/home/user/UROPJulia/KernelAgentBench"),
        patch_path=Path(
            "/home/user/UROPJulia/KernelAgentBench/candidates/candidate-1/candidate.patch"
        ),
    )
    assert "/System/" not in remote_script
    assert "REPOSITORY=/home/user/UROPJulia/KernelAgentBench" in remote_script
    commands = ssh_handoff_commands(config, bundle=bundle, local_project=project)
    assert any("sbatch --parsable" in command for command in commands)
    assert all("salloc " not in command for command in commands)
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
        language="julia-ka",
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
    assert "ControlMaster=auto" in rsync_ssh(config)
    assert ssh_control_path(config) in rsync_ssh(config)
    assert "%C" not in rsync_ssh(config)
    assert ssh_control_path(config).endswith("kab-user@orcd-login.mit.edu:22")
    assert _parse_sbatch_job_id("23595210\n") == "23595210"
    assert _parse_sbatch_job_id("23595210;mit_normal_gpu") == "23595210"
    assert remote_eval_command(config, "/tmp/job.sbatch") == "sbatch --parsable /tmp/job.sbatch"
    held = config.model_copy(update={"salloc_job_id": "23595210"})
    assert "--jobid=23595210" in remote_eval_command(held, "/tmp/job.sbatch")
    assert "salloc " not in remote_eval_command(held, "/tmp/job.sbatch")
    assert "/candidates/" in remote_mkdir_command(transfer)
    assert remote_mkdir_command(transfer).startswith("mkdir -p ")
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


def test_empty_langsmith_key_disables_hosted_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    configure_tracing()
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_content_redaction() -> None:
    value = {"prompt": "secret source", "token_usage": {"total_tokens": 42}}
    assert trace_payload(value, include_content=False) == {
        "prompt": "[redacted]",
        "token_usage": {"total_tokens": 42},
    }
    assert trace_payload(value, include_content=True) == value


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, text=True)


def test_trials_refuse_uncommitted_cpp_changes(tmp_path: Path) -> None:
    kernel = tmp_path / "cpp" / "gray_scott" / "candidate_kernel.cl"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("__kernel void step_kernel(){}\n", encoding="utf-8")
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "baseline")
    manager = CandidateManager(tmp_path)
    manager.require_committed()
    kernel.write_text("__kernel void step_kernel(){ /* edit */ }\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cpp/gray_scott/candidate_kernel.cl"):
        manager.create_worktree("trial-cpp")


def test_trials_refuse_uncommitted_kernel_changes(tmp_path: Path) -> None:
    kernel = tmp_path / "kernels" / "baseline.jl"
    kernel.parent.mkdir()
    kernel.write_text("old\n", encoding="utf-8")
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "baseline")
    manager = CandidateManager(tmp_path)
    manager.require_committed()
    kernel.write_text("new\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="kernels/baseline.jl"):
        manager.create_worktree("trial-1")


def test_remote_commit_check_fetches_then_rechecks(tmp_path: Path) -> None:
    project = tmp_path / "KernelAgentBench"
    bundle = project / "candidates" / "candidate-1"
    bundle.mkdir(parents=True)
    manifest = CandidateManifest(
        candidate_id="candidate-1",
        trial_id="trial-1",
        baseline_commit="abc1234",
        patch_sha256="0" * 64,
        patch_file="candidate.patch",
        iteration=1,
        language="julia-cuda",
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
        remote_project="/home/user/KernelAgentBench",
    )
    transfer = remote_transfer(config, bundle, project)
    command = remote_commit_command(transfer, "abc1234")
    assert command == (
        "git -C /home/user/KernelAgentBench cat-file -e abc1234^{commit} 2>/dev/null || "
        "{ git -C /home/user/KernelAgentBench fetch --quiet origin && "
        "git -C /home/user/KernelAgentBench cat-file -e abc1234^{commit}; }"
    )
    with pytest.raises(ValueError):
        remote_commit_command(transfer, "HEAD; rm -rf ~")


def _study(
    language: str = "julia-ka",
    editable_path: str | None = None,
    prompt_path: str = "prompts/task.md",
    *,
    evaluate_locally: bool | None = None,
) -> StudyConfig:
    backend = {
        "julia-cpu": "cpu",
        "julia-cuda": "cuda",
        "julia-ka": "ka",
        "julia": "cuda",
        "python": "cuda",
        "cpp": "opencl",
    }.get(language, "ka")
    default_editable = {
        "julia-cpu": "kernels/gray_scott/cpu/candidate.jl",
        "julia-cuda": "kernels/gray_scott/cuda/candidate.jl",
        "julia-ka": "kernels/gray_scott/ka/candidate.jl",
        "julia": "kernels/gray_scott/cuda/candidate.jl",
        "python": "python/gray_scott/candidate.py",
        "cpp": "cpp/gray_scott/candidate_kernel.cl",
    }.get(language, "kernels/gray_scott/ka/candidate.jl")
    local = backend == "cpu" if evaluate_locally is None else evaluate_locally
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
            correctness_dtypes=["Float32", "Float64"],
            float64_atol=1e-12,
            float64_rtol=1e-10,
        ),
        languages=[
            LanguageTarget(
                id=language,
                backend=backend,  # type: ignore[arg-type]
                requires_gpu=not local,
                evaluate_locally=local,
                editable_paths=[editable_path or default_editable],
                public_test_command="python3 -c pass",
                prompt_path=prompt_path,
            )
        ],
    )


@pytest.mark.parametrize(
    ("language", "editable_path"),
    [
        ("julia-ka", "kernels/gray_scott/ka/candidate.jl"),
        ("julia-cuda", "kernels/gray_scott/cuda/candidate.jl"),
        ("python", "python/gray_scott/candidate.py"),
        ("cpp", "cpp/gray_scott/candidate_kernel.cl"),
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
            correctness_cases=JULIA_CORRECTNESS_CASES,
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
        assert finished["first_correct_iteration"] == 1
        assert finished["tokens_to_correctness"] == 5
    finally:
        graph.close()


def test_auto_salloc_evaluation_closes_the_loop(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    project = repository / "KernelAgentBench"
    candidate = project / "kernels" / "gray_scott" / "ka" / "candidate.jl"
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
        (workspace / "kernels" / "gray_scott" / "ka" / "candidate.jl").write_text(
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
        assert orcd.ntasks == 1
        manifest = CandidateManifest.model_validate_json(
            (bundle / "manifest.json").read_text(encoding="utf-8")
        )
        result = EvaluationResult(
            candidate_id=manifest.candidate_id,
            language="julia-ka",
            evaluator_version="test",
            correctness_cases=JULIA_CORRECTNESS_CASES,
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
            language="julia-ka",
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
    candidate = project / "kernels" / "gray_scott" / "ka" / "candidate.jl"
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
        path = workspace / "kernels" / "gray_scott" / "ka" / "candidate.jl"
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
            language="julia-ka",
            replicate=1,
        )
        result = EvaluationResult(
            candidate_id=waiting["candidate_id"],
            language="julia-ka",
            evaluator_version="test",
            correctness_cases=JULIA_CORRECTNESS_CASES,
            status="passed",
            correct=True,
            workloads=[],
            geometric_mean_speedup=1.02,
        )
        result_path = tmp_path / "result.json"
        result_path.write_text(result.model_dump_json(), encoding="utf-8")
        waiting = graph.resume("trial-early-stop", result_path)
        assert waiting["status"] == "awaiting_h200_result"
        assert waiting["h200_evaluations"] == 1
        assert waiting.get("no_improvement_streak") == 1
        assert waiting.get("termination_reason") is None

        result2 = EvaluationResult(
            candidate_id=waiting["candidate_id"],
            language="julia-ka",
            evaluator_version="test",
            correctness_cases=JULIA_CORRECTNESS_CASES,
            status="passed",
            correct=True,
            workloads=[],
            geometric_mean_speedup=1.02,
        )
        result_path2 = tmp_path / "result2.json"
        result_path2.write_text(result2.model_dump_json(), encoding="utf-8")
        finished = graph.resume("trial-early-stop", result_path2)
        assert finished["status"] == "finished"
        assert finished["h200_evaluations"] == 2
        assert finished["termination_reason"] == "no_significant_improvement"
        assert finished["incumbent_speedup"] == 1.02
        assert finished["first_correct_iteration"] == 1
        assert finished["tokens_to_correctness"] == 5
    finally:
        graph.close()


def test_workspace_paths_must_cover_editable_and_prompt() -> None:
    with pytest.raises(ValueError, match="workspace_paths must include"):
        LanguageTarget(
            id="julia-ka",
            backend="ka",
            editable_paths=["kernels/gray_scott/ka/candidate.jl"],
            workspace_paths=["julia"],
            public_test_command="true",
            prompt_path="prompts/optimize_gray_scott_ka.md",
        )


def test_agent_workspace_excludes_other_backends(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    project = repository / "KernelAgentBench"
    candidate = project / "kernels" / "gray_scott" / "ka" / "candidate.jl"
    prompt = project / "prompts" / "optimize_gray_scott_ka.md"
    decoy = project / "kernels" / "gray_scott" / "cuda" / "candidate.jl"
    harness = project / "src" / "kernel_agent_bench" / "graph.py"
    candidate.parent.mkdir(parents=True)
    prompt.parent.mkdir(parents=True)
    decoy.parent.mkdir(parents=True)
    harness.parent.mkdir(parents=True)
    candidate.write_text("baseline = true\n", encoding="utf-8")
    prompt.write_text("optimize", encoding="utf-8")
    decoy.write_text("do not read me\n", encoding="utf-8")
    harness.write_text("harness\n", encoding="utf-8")
    (project / "julia").mkdir()
    (project / "julia" / "evaluate_candidate.jl").write_text("eval\n", encoding="utf-8")
    (project / ".gitignore").write_text(
        ".kernel-agent-bench/\ncandidates/\nresults/\n", encoding="utf-8"
    )
    _git(repository, "init")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "baseline")

    seen: list[Path] = []

    def fake_runner(model_id: str, workspace: Path, agent_prompt: str) -> AgentRun:
        del model_id, agent_prompt
        seen.append(workspace)
        assert (workspace / "kernels" / "gray_scott" / "ka" / "candidate.jl").is_file()
        assert (workspace / "julia" / "evaluate_candidate.jl").is_file()
        assert not (workspace / "kernels" / "gray_scott" / "cuda").exists()
        assert not (workspace / "src").exists()
        (workspace / "kernels" / "gray_scott" / "ka" / "candidate.jl").write_text(
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

    study = _study()
    study = study.model_copy(
        update={
            "languages": [
                LanguageTarget(
                    id="julia-ka",
                    backend="ka",
                    editable_paths=["kernels/gray_scott/ka/candidate.jl"],
                    workspace_paths=[
                        "kernels/gray_scott/ka",
                        "julia",
                        "prompts/optimize_gray_scott_ka.md",
                    ],
                    public_test_command="python3 -c pass",
                    prompt_path="prompts/optimize_gray_scott_ka.md",
                )
            ]
        }
    )
    graph = TrialGraph(project, model_runner=fake_runner)
    try:
        waiting = graph.start(
            study=study,
            study_hash=canonical_hash(study),
            trial_id="trial-slim",
            model_alias="mock",
            resolved_model_id="mock-1",
            language="julia-ka",
            replicate=1,
        )
        assert waiting["status"] == "awaiting_h200_result"
        assert seen
        assert seen[0] != Path(waiting["workspace"])
        worktree_candidate = (
            Path(waiting["workspace"]) / "kernels" / "gray_scott" / "ka" / "candidate.jl"
        )
        assert worktree_candidate.read_text(encoding="utf-8") == "baseline = false\n"
        assert CandidateManager(project).bundles.is_dir()
    finally:
        graph.close()


def test_cpu_backend_evaluates_locally(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    project = repository / "KernelAgentBench"
    candidate = project / "kernels" / "gray_scott" / "cpu" / "candidate.jl"
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
        (workspace / "kernels" / "gray_scott" / "cpu" / "candidate.jl").write_text(
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

    def fake_local(local_project: Path, workspace: Path, bundle: Path) -> Path:
        del workspace
        manifest = CandidateManifest.model_validate_json(
            (bundle / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest.language == "julia-cpu"
        result = EvaluationResult(
            candidate_id=manifest.candidate_id,
            language="julia-cpu",
            backend="cpu",
            evaluator_version="test",
            correctness_cases=JULIA_CORRECTNESS_CASES,
            status="passed",
            correct=True,
            workloads=[],
            geometric_mean_speedup=1.1,
        )
        path = local_project / "results" / f"{manifest.candidate_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(), encoding="utf-8")
        return path

    graph = TrialGraph(
        project,
        model_runner=fake_runner,
        local_evaluate_runner=fake_local,
    )
    try:
        finished = graph.start(
            study=_study("julia-cpu"),
            study_hash=canonical_hash(_study("julia-cpu")),
            trial_id="trial-cpu",
            model_alias="mock",
            resolved_model_id="mock-1",
            language="julia-cpu",
            replicate=1,
        )
        assert finished["status"] == "finished"
        assert finished["h200_evaluations"] == 1
        assert finished["incumbent_speedup"] == 1.1
    finally:
        graph.close()

    results = project / "results"
    runs = read_jsonl([results / "agent_runs.jsonl"])
    assert [row["event"] for row in runs] == ["agent_run", "public_check"]
    assert runs[0]["trial_id"] == "trial-cpu"
    assert runs[0]["token_usage"]["total_tokens"] == 5
    assert runs[0]["wall_seconds"] >= 0
    (evaluation,) = read_jsonl([results / "evaluation_events.jsonl"])
    assert evaluation["geometric_mean_speedup"] == 1.1
    assert evaluation["wall_seconds"] >= 0
    (trial,) = read_jsonl([results / "trials.jsonl"])
    assert trial["token_usage"]["total_tokens"] == 5
    assert trial["first_correct_iteration"] == 1
    assert trial["tokens_to_correctness"] == 5
    assert trial["metadata"]["wall_seconds"] >= trial["metadata"]["evaluation_seconds"] >= 0


def test_tokens_to_correctness_after_failed_h200_eval(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    project = repository / "KernelAgentBench"
    candidate = project / "kernels" / "gray_scott" / "ka" / "candidate.jl"
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
        (workspace / "kernels" / "gray_scott" / "ka" / "candidate.jl").write_text(
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

    study = _study()
    study = study.model_copy(
        update={
            "budgets": Budgets(
                max_agent_iterations=2,
                max_h200_evaluations=2,
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
            trial_id="trial-correctness-metrics",
            model_alias="mock",
            resolved_model_id="mock-1",
            language="julia-ka",
            replicate=1,
        )
        fail = EvaluationResult(
            candidate_id=waiting["candidate_id"],
            language="julia-ka",
            evaluator_version="test",
            correctness_cases=JULIA_CORRECTNESS_CASES,
            status="failed",
            correct=False,
            workloads=[],
            geometric_mean_speedup=None,
        )
        fail_path = tmp_path / "fail.json"
        fail_path.write_text(fail.model_dump_json(), encoding="utf-8")
        waiting = graph.resume("trial-correctness-metrics", fail_path)
        assert waiting["status"] == "awaiting_h200_result"
        assert waiting.get("first_correct_iteration") is None
        assert waiting.get("tokens_to_correctness") is None

        pass_result = EvaluationResult(
            candidate_id=waiting["candidate_id"],
            language="julia-ka",
            evaluator_version="test",
            correctness_cases=JULIA_CORRECTNESS_CASES,
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
        pass_path = tmp_path / "pass.json"
        pass_path.write_text(pass_result.model_dump_json(), encoding="utf-8")
        finished = graph.resume("trial-correctness-metrics", pass_path)
        assert finished["status"] == "finished"
        assert finished["first_correct_iteration"] == 2
        assert finished["tokens_to_correctness"] == 10
    finally:
        graph.close()

    (trial,) = read_jsonl([project / "results" / "trials.jsonl"])
    assert trial["first_correct_iteration"] == 2
    assert trial["tokens_to_correctness"] == 10
    evaluations = read_jsonl([project / "results" / "evaluation_events.jsonl"])
    assert evaluations[0]["correct"] is False
    assert "first_correct_iteration" not in evaluations[0]
    assert evaluations[1]["first_correct_iteration"] == 2
    assert evaluations[1]["tokens_to_correctness"] == 10


def test_cpp_opencl_packages_candidate(tmp_path: Path) -> None:
    bench_root = Path(__file__).resolve().parents[1]
    repository = tmp_path / "repo"
    project = repository / "KernelAgentBench"
    cpp_tree = bench_root / "cpp"
    for relative in (
        "gray_scott/baseline_kernel.cl",
        "gray_scott/candidate_kernel.cl",
        "gray_scott/baseline.hpp",
        "gray_scott/candidate.hpp",
        "gray_scott/opencl_common.hpp",
        "gray_scott/opencl_runner.hpp",
        "evaluate_candidate.cpp",
    ):
        src = cpp_tree / relative
        dest = project / "cpp" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    prompt = project / "prompts" / "optimize_gray_scott_cpp.md"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("optimize cpp", encoding="utf-8")
    (project / ".gitignore").write_text(
        ".kernel-agent-bench/\ncandidates/\nresults/\n", encoding="utf-8"
    )
    _git(repository, "init")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "baseline")

    study = _study("cpp")
    manager = CandidateManager(project)
    worktree = manager.create_worktree("trial-cpp-pack")
    kernel = worktree.project / "cpp/gray_scott/candidate_kernel.cl"
    kernel.write_text(
        kernel.read_text(encoding="utf-8") + "\n// agent tweak\n",
        encoding="utf-8",
    )
    manifest, bundle = manager.package(
        worktree,
        "trial-cpp-pack",
        1,
        study.languages[0].editable_paths,
        None,
        "cpp",
        study.task_version,
        canonical_hash(study),
        "optimize cpp",
        study.evaluation,
    )
    assert manifest.language == "cpp"
    assert "candidate_kernel.cl" in (bundle / "candidate.patch").read_text(encoding="utf-8")
    manager.cleanup_worktree(worktree.root)


def test_study_example_is_julia_backends() -> None:
    from kernel_agent_bench.config import load_study, project_root

    study = load_study(project_root() / "config" / "study.example.yaml")
    assert [target.id for target in study.languages] == [
        "julia-ka",
        "julia-cuda",
        "julia-cpu",
    ]
    assert [target.backend for target in study.languages] == ["ka", "cuda", "cpu"]
    assert study.languages[0].requires_gpu is True
    assert study.languages[2].evaluate_locally is True


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
