from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .schemas import CandidateManifest, OrcdConfig

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


def _require_safe_name(value: str, *, label: str) -> str:
    if not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def _require_contained(root: Path, child: Path, *, label: str) -> Path:
    resolved_root = root.resolve()
    resolved_child = child if child.is_absolute() else resolved_root / child
    escaped = ".." in Path(child).parts
    if escaped or not resolved_child.resolve().is_relative_to(resolved_root):
        raise ValueError(f"{label} escapes {resolved_root}: {child}")
    return resolved_child


@dataclass(frozen=True)
class RemoteTransfer:
    candidate_id: str
    remote_project: Path
    remote_bundle: Path
    remote_result: Path
    remote_script: Path
    local_result: Path


def remote_transfer(config: OrcdConfig, bundle: Path, local_project: Path) -> RemoteTransfer:
    manifest = CandidateManifest.model_validate_json(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )
    candidate_id = _require_safe_name(manifest.candidate_id, label="candidate_id")
    _require_safe_name(Path(manifest.patch_file).name, label="patch_file")
    if Path(manifest.patch_file).name != manifest.patch_file:
        raise ValueError(f"patch_file must be a basename: {manifest.patch_file!r}")
    remote_project = Path(config.remote_project)
    candidates_root = remote_project / "candidates"
    results_root = remote_project / "results"
    slurm_root = remote_project / ".kernel-agent-bench" / "slurm"
    local_results = local_project.resolve() / "results"
    return RemoteTransfer(
        candidate_id=candidate_id,
        remote_project=remote_project,
        remote_bundle=_require_contained(
            candidates_root, candidates_root / candidate_id, label="remote bundle"
        ),
        remote_result=_require_contained(
            results_root,
            results_root / f"{candidate_id}.json",
            label="remote result",
        ),
        remote_script=_require_contained(
            slurm_root,
            slurm_root / f"{candidate_id}.sbatch",
            label="remote script",
        ),
        local_result=_require_contained(
            local_results, local_results / f"{candidate_id}.json", label="local result"
        ),
    )


def rsync_upload_command(config: OrcdConfig, bundle: Path, transfer: RemoteTransfer) -> list[str]:
    return [
        "rsync",
        "-az",
        "-e",
        rsync_ssh(config),
        "--",
        str(bundle.resolve()) + "/",
        f"{config.ssh_user}@{config.ssh_host}:{transfer.remote_bundle}/",
    ]


def rsync_download_command(config: OrcdConfig, transfer: RemoteTransfer) -> list[str]:
    return [
        "rsync",
        "-az",
        "-e",
        rsync_ssh(config),
        "--",
        f"{config.ssh_user}@{config.ssh_host}:{transfer.remote_result}",
        str(transfer.local_result),
    ]


def _directive(name: str, value: str) -> str:
    return f"#SBATCH --{name}={value}" if value else ""


def salloc_command(config: OrcdConfig) -> list[str]:
    command = [
        "salloc",
        "-N",
        str(config.nodes),
        "-n",
        str(config.ntasks),
        "--cpus-per-task",
        str(config.cpus_per_task),
        "--mem",
        config.memory,
        "--time",
        config.wall_time,
        "-G",
        config.gpu_request,
    ]
    if config.partition:
        command.extend(["-p", config.partition])
    if config.account and config.account != "REQUIRED":
        command.extend(["-A", config.account])
    if config.qos:
        command.extend(["--qos", config.qos])
    if config.reservation:
        command.extend(["--reservation", config.reservation])
    return command


def srun_command(config: OrcdConfig, script: str) -> list[str]:
    command = ["srun", "--ntasks=1", f"--gres=gpu:{config.gpu_request}", "bash", script]
    if config.salloc_job_id:
        command[1:1] = [f"--jobid={config.salloc_job_id}"]
    return command


def ssh_base(config: OrcdConfig) -> list[str]:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-p",
        str(config.ssh_port),
    ]
    if config.identity_file:
        command.extend(["-i", config.identity_file])
    command.append(f"{config.ssh_user}@{config.ssh_host}")
    return command


def rsync_ssh(config: OrcdConfig) -> str:
    transport = ["ssh", "-o", "BatchMode=yes", "-p", str(config.ssh_port)]
    if config.identity_file:
        transport.extend(["-i", config.identity_file])
    return " ".join(shlex.quote(value) for value in transport)


def render_slurm(
    config: OrcdConfig,
    *,
    project: Path,
    bundle: Path,
    result: Path,
    repository: Path | None = None,
    patch_path: Path | None = None,
) -> str:
    manifest = CandidateManifest.model_validate_json(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )
    local_patch = bundle / manifest.patch_file
    if not local_patch.is_file():
        raise FileNotFoundError(local_patch)
    patch = patch_path or local_patch
    repository = (repository or project.parent).resolve()
    relative_project = project.resolve().relative_to(repository)
    scratch = Path(config.scratch_root) / "kernel-agent-bench"
    setup = "\n".join(config.setup_commands)
    evaluator_args = [
        "--candidate-id",
        manifest.candidate_id,
        "--output",
        str(result.resolve()),
        "--dtype",
        manifest.evaluation.dtype,
        "--shapes",
        ",".join(f"{height}x{width}" for height, width in manifest.evaluation.performance_shapes),
        "--correctness-shapes",
        ",".join(f"{height}x{width}" for height, width in manifest.evaluation.correctness_shapes),
        "--steps",
        str(manifest.evaluation.steps),
        "--warmups",
        str(manifest.evaluation.warmups),
        "--repetitions",
        str(manifest.evaluation.repetitions),
        "--atol",
        str(manifest.evaluation.atol),
        "--rtol",
        str(manifest.evaluation.rtol),
    ]
    if manifest.language == "julia":
        executable = [
            "julia",
            f"--project={relative_project}/julia",
            f"{relative_project}/julia/evaluate_candidate.jl",
        ]
        command = " ".join(shlex.quote(value) for value in executable + evaluator_args)
    elif manifest.language == "python":
        executable = ["python3", f"{relative_project}/python/evaluate_candidate.py"]
        command = " ".join(shlex.quote(value) for value in executable + evaluator_args)
    elif manifest.language == "cpp":
        if manifest.evaluation.dtype != "Float32":
            raise ValueError("the CUDA C++ evaluator currently supports Float32 only")
        binary = f"{relative_project}/.build/evaluate_candidate_cpp"
        source = f"{relative_project}/cpp/evaluate_candidate.cu"
        run_args = evaluator_args.copy()
        dtype_index = run_args.index("--dtype")
        del run_args[dtype_index : dtype_index + 2]
        command = (
            f'mkdir -p "$(dirname {binary})"\n'
            "COMPILE_STARTED=$(date +%s%N)\n"
            f"nvcc -O3 -std=c++17 -lineinfo {source} -o {binary}\n"
            "COMPILE_ENDED=$(date +%s%N)\n"
            "COMPILE_SECONDS=$(python3 -c "
            "'import sys; print((int(sys.argv[2])-int(sys.argv[1]))/1e9)' "
            '"$COMPILE_STARTED" "$COMPILE_ENDED")\n'
            + binary
            + " "
            + " ".join(shlex.quote(value) for value in run_args)
            + ' --compile-seconds "$COMPILE_SECONDS"'
        )
    else:
        raise ValueError(f"unsupported candidate language: {manifest.language}")
    if config.container_image:
        command = (
            "apptainer exec --nv "
            + shlex.quote(config.container_image)
            + " bash -lc "
            + shlex.quote(command)
        )

    directives = "\n".join(
        line
        for line in (
            _directive("job-name", f"kab-{manifest.candidate_id[:32]}"),
            _directive("partition", config.partition),
            _directive("nodes", str(config.nodes)),
            _directive("ntasks", str(config.ntasks)),
            f"#SBATCH -G {config.gpu_request}",
            _directive("cpus-per-task", str(config.cpus_per_task)),
            _directive("mem", config.memory),
            _directive("time", config.wall_time),
            _directive("account", config.account),
            _directive("qos", config.qos),
            _directive("reservation", config.reservation),
            _directive("mail-user", config.email),
            _directive("mail-type", "END,FAIL" if config.email else ""),
            _directive("output", str(project / "results" / "slurm-%j.out")),
        )
        if line
    )
    metadata_path = result.with_suffix(".environment.json")
    return f"""#!/usr/bin/env bash
{directives}

set -euo pipefail

REPOSITORY={shlex.quote(str(repository))}
PATCH={shlex.quote(str(patch.resolve()))}
WORKTREE={shlex.quote(str(scratch))}/${{SLURM_JOB_ID}}-{shlex.quote(manifest.candidate_id)}
RESULT={shlex.quote(str(result.resolve()))}
METADATA={shlex.quote(str(metadata_path.resolve()))}

mkdir -p "$(dirname "$RESULT")" "$(dirname "$WORKTREE")"
cleanup() {{
  git -C "$REPOSITORY" worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
}}
trap cleanup EXIT

git -C "$REPOSITORY" worktree add --detach "$WORKTREE" {shlex.quote(manifest.baseline_commit)}
git -C "$WORKTREE" apply --check "$PATCH"
git -C "$WORKTREE" apply "$PATCH"
cd "$WORKTREE"

{setup}

python3 - "$METADATA" <<'PY'
import json
import os
import platform
import subprocess
import sys

def capture(command):
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {{
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }}

metadata = {{
    "slurm": {{key: value for key, value in os.environ.items() if key.startswith("SLURM_")}},
    "platform": platform.platform(),
    "nvidia_smi": capture(["nvidia-smi", "-q"]),
    "git": capture(["git", "rev-parse", "HEAD"]),
}}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2, sort_keys=True)
    handle.write("\\n")
PY

{command}
"""


def candidate_id(bundle: Path) -> str:
    data = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    return str(data["candidate_id"])


def remote_eval_command(config: OrcdConfig, script: str) -> str:
    srun = " ".join(shlex.quote(value) for value in srun_command(config, script))
    if config.salloc_job_id:
        return srun
    salloc = " ".join(shlex.quote(value) for value in salloc_command(config))
    return f"{salloc} -- {srun}"


def ssh_handoff_commands(
    config: OrcdConfig,
    *,
    bundle: Path,
    local_project: Path,
) -> list[str]:
    manifest = CandidateManifest.model_validate_json(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )
    transfer = remote_transfer(config, bundle, local_project)
    return [
        "# Upload only this candidate bundle; the ORCD clone is not synced:",
        " ".join(shlex.quote(value) for value in rsync_upload_command(config, bundle, transfer)),
        "# Allocate the H200 and evaluate (or srun if salloc_job_id is set):",
        " ".join(shlex.quote(value) for value in ssh_base(config))
        + " "
        + shlex.quote(
            f"cd {shlex.quote(str(transfer.remote_project))} && "
            f"{remote_eval_command(config, str(transfer.remote_script))}"
        ),
        "# Download only this candidate's result JSON:",
        " ".join(shlex.quote(value) for value in rsync_download_command(config, transfer)),
        "kernel-agent-bench resume --trial "
        + shlex.quote(manifest.trial_id)
        + " --result "
        + shlex.quote(str(transfer.local_result)),
    ]


def evaluate_candidate_remote(
    config: OrcdConfig,
    bundle: Path,
    local_project: Path,
) -> Path:
    manifest = CandidateManifest.model_validate_json(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )
    transfer = remote_transfer(config, bundle, local_project)
    transfer.local_result.parent.mkdir(parents=True, exist_ok=True)

    script = render_slurm(
        config,
        project=transfer.remote_project,
        bundle=bundle,
        result=transfer.remote_result,
        repository=transfer.remote_project.parent,
        patch_path=transfer.remote_bundle / manifest.patch_file,
    )
    subprocess.run(rsync_upload_command(config, bundle, transfer), check=True)
    mkdir = (
        "mkdir -p "
        + shlex.quote(str(transfer.remote_script.parent))
        + " "
        + shlex.quote(str(transfer.remote_result.parent))
    )
    subprocess.run(ssh_base(config) + [mkdir], check=True)
    subprocess.run(
        ssh_base(config)
        + [
            "cat > "
            + shlex.quote(str(transfer.remote_script))
            + " && chmod 750 "
            + shlex.quote(str(transfer.remote_script))
        ],
        input=script,
        text=True,
        check=True,
    )
    evaluate = (
        f"cd {shlex.quote(str(transfer.remote_project))} && "
        f"{remote_eval_command(config, str(transfer.remote_script))}"
    )
    subprocess.run(ssh_base(config) + [evaluate], check=True)
    subprocess.run(rsync_download_command(config, transfer), check=True)
    if not transfer.local_result.is_file():
        raise FileNotFoundError(f"H200 result was not downloaded: {transfer.local_result}")
    return transfer.local_result
