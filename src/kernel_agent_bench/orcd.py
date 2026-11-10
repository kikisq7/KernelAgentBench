from __future__ import annotations

import fcntl
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .schemas import (
    CandidateManifest,
    OrcdConfig,
    is_julia_language,
    julia_backend_for_language,
    python_backend_for_language,
)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


def _require_safe_name(value: str, *, label: str) -> str:
    if not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def _normalize_abs(path: Path) -> Path:
    """Collapse . and .. without following local symlinks.

    Remote ORCD paths must not be Path.resolve()'d on macOS: /home becomes
    /System/Volumes/Data/home and the compute node then mkdir's /System.
    """
    if not path.is_absolute():
        raise ValueError(f"expected an absolute path: {path}")
    parts: list[str] = []
    for part in path.parts[1:]:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise ValueError(f"path escapes the filesystem root: {path}")
            parts.pop()
            continue
        parts.append(part)
    return Path("/") / Path(*parts) if parts else Path("/")


def _require_contained(root: Path, child: Path, *, label: str) -> Path:
    root_n = _normalize_abs(root if root.is_absolute() else root.resolve())
    child_path = child if child.is_absolute() else root_n / child
    child_n = _normalize_abs(child_path)
    if not child_n.is_relative_to(root_n):
        raise ValueError(f"{label} escapes {root_n}: {child}")
    return child_n


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


HOLD_WALL_TIME = "04:00:00"
HOLD_JOB_NAME = "kab-h200-hold"
_HOLD_STATES = {"PENDING", "RUNNING", "CONFIGURING", "REQUEUED"}


def salloc_command(config: OrcdConfig, *, wall_time: str | None = None) -> list[str]:
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
        wall_time or config.wall_time,
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


def sbatch_command(config: OrcdConfig, script: str) -> list[str]:
    del config
    return ["sbatch", "--parsable", script]


_TERMINAL_SLURM = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "SPECIAL_EXIT",
}


def ssh_control_path(config: OrcdConfig) -> str:
    # A fixed path (user@host:port). Do not use %C: ssh -O check without -p
    # hashes a different socket than ssh -p 22, so the mux looks dead.
    token = f"{config.ssh_user}@{config.ssh_host}:{config.ssh_port}"
    safe = re.sub(r"[^A-Za-z0-9._@:-]", "_", token)
    return str(Path.home() / ".ssh" / f"kab-{safe}")


def resolve_identity(config: OrcdConfig) -> str:
    if config.identity_file:
        return config.identity_file
    for name in ("id_ed25519", "id_rsa", "id_ecdsa"):
        path = Path.home() / ".ssh" / name
        if path.is_file():
            return str(path)
    return ""


def ssh_target(config: OrcdConfig) -> str:
    return f"{config.ssh_user}@{config.ssh_host}"


def ssh_master_running(config: OrcdConfig) -> bool:
    completed = subprocess.run(
        [
            "ssh",
            "-p",
            str(config.ssh_port),
            "-O",
            "check",
            "-o",
            f"ControlPath={ssh_control_path(config)}",
            ssh_target(config),
        ],
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def ssh_options(config: OrcdConfig, *, batch: bool | None = None) -> list[str]:
    if batch is None:
        batch = ssh_master_running(config) or not sys.stdin.isatty()
    (Path.home() / ".ssh").mkdir(parents=True, exist_ok=True)
    command = [
        "ssh",
        "-p",
        str(config.ssh_port),
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPath={ssh_control_path(config)}",
        "-o",
        "ControlPersist=8h",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=12",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    # After the key is accepted, ORCD still requires Duo keyboard-interactive
    # unless this process is a ControlMaster slave.
    if ssh_master_running(config):
        if batch:
            command.extend(["-o", "BatchMode=yes"])
        return command
    identity = resolve_identity(config)
    if identity:
        command.extend(["-i", identity, "-o", "IdentitiesOnly=yes"])
    command.extend(
        [
            "-o",
            "PreferredAuthentications=publickey,keyboard-interactive",
            "-o",
            "GSSAPIAuthentication=yes",
            "-o",
            "GSSAPIDelegateCredentials=yes",
        ]
    )
    if batch:
        command.extend(["-o", "BatchMode=yes"])
    return command


def ssh_base(config: OrcdConfig, *, batch: bool | None = None) -> list[str]:
    return ssh_options(config, batch=batch) + [ssh_target(config)]


def rsync_ssh(config: OrcdConfig, *, batch: bool | None = None) -> str:
    return " ".join(shlex.quote(value) for value in ssh_options(config, batch=batch))


def probe_ssh(config: OrcdConfig) -> None:
    """Reuse a ControlMaster if one exists; otherwise open one (Duo may prompt)."""
    if ssh_master_running(config):
        completed = subprocess.run(ssh_base(config, batch=True) + ["true"], check=False)
        if completed.returncode == 0:
            return
    completed = subprocess.run(ssh_base(config, batch=False) + ["true"], check=False)
    if completed.returncode == 0:
        return
    raise RuntimeError(
        f"SSH to {ssh_target(config)} needs a ControlMaster because ORCD asks "
        "for Duo after the public key. In a terminal run:\n"
        f"  kernel-agent-bench ssh-check --orcd config/orcd.yaml\n"
        "Approve Duo once; later rsync/sbatch reuse the ControlPersist socket."
    )


def remote_mkdir_command(transfer: RemoteTransfer) -> str:
    return (
        "mkdir -p "
        + shlex.quote(str(transfer.remote_bundle))
        + " "
        + shlex.quote(str(transfer.remote_script.parent))
        + " "
        + shlex.quote(str(transfer.remote_result.parent))
    )


def remote_commit_command(transfer: RemoteTransfer, commit: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{7,64}", commit):
        raise ValueError(f"baseline_commit must be a hex SHA: {commit!r}")
    repo = shlex.quote(str(transfer.remote_project))
    has = f"git -C {repo} cat-file -e {commit}^{{commit}}"
    return f"{has} 2>/dev/null || {{ git -C {repo} fetch --quiet origin && {has}; }}"


def _run(command: list[str], *, input_text: str | None = None) -> None:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        detail = "\n".join(
            part.strip() for part in (completed.stderr, completed.stdout) if part
        ).strip()
        raise RuntimeError(
            f"command failed ({completed.returncode}): {shlex.join(command)}\n{detail}"
        )


def manifest_evaluator_args(manifest: CandidateManifest, output: Path) -> list[str]:
    """One command line for every evaluator; each ignores options it does not use."""
    evaluation = manifest.evaluation
    args = [
        "--candidate-id",
        manifest.candidate_id,
        "--output",
        output.as_posix() if output.is_absolute() else str(output),
        "--language",
        manifest.language,
        "--dtype",
        evaluation.dtype,
        "--shapes",
        ",".join(f"{nx}x{ny}" for nx, ny in evaluation.performance_shapes),
        "--correctness-shapes",
        ",".join(f"{nx}x{ny}" for nx, ny in evaluation.correctness_shapes),
        "--steps",
        str(evaluation.steps),
        "--warmups",
        str(evaluation.warmups),
        "--repetitions",
        str(evaluation.repetitions),
        "--atol",
        str(evaluation.atol),
        "--rtol",
        str(evaluation.rtol),
    ]
    if evaluation.cpu_threads:
        args.extend(["--threads", str(evaluation.cpu_threads)])
    backend = julia_backend_for_language(manifest.language) or python_backend_for_language(
        manifest.language
    )
    if backend is not None:
        args.extend(["--backend", backend])
    return args


# -ffp-contract=off keeps the notebook's arithmetic: Julia never fuses a*b+c on its own.
def cpp_cpu_compile_command(workspace: Path, binary: Path) -> list[str]:
    """Host C++ CPU evaluator. No OpenCL: this track is the CPU comparison."""
    return [
        "g++",
        "-O3",
        "-std=c++17",
        f"-I{workspace / 'cpp' / 'gray_scott' / 'cpu'}",
        "-ffp-contract=off",
        str(workspace / "cpp" / "evaluate_cpu.cpp"),
        "-o",
        str(binary),
    ]


def julia_command(workspace: Path, threads: int) -> list[str]:
    command = ["julia", f"--project={workspace / 'julia'}"]
    if threads:
        command.insert(1, f"--threads={threads}")
    return command + [str(workspace / "julia" / "evaluate_candidate.jl")]


def _run_local_evaluator(command: list[str], workspace: Path, output: Path) -> Path:
    completed = subprocess.run(
        command,
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )
    if not output.is_file():
        detail = "\n".join(
            part.strip() for part in (completed.stderr, completed.stdout) if part
        ).strip()
        raise RuntimeError(
            f"local evaluator produced no result ({completed.returncode}): "
            f"{shlex.join(command)}\n{detail}"
        )
    return output


def evaluate_candidate_local(project: Path, workspace: Path, bundle: Path) -> Path:
    manifest = CandidateManifest.model_validate_json(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )
    output = project / "results" / f"{manifest.candidate_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if manifest.language == "cpp-cpu":
        binary_dir = project / ".kernel-agent-bench" / "build"
        binary_dir.mkdir(parents=True, exist_ok=True)
        binary = binary_dir / f"evaluate_cpu_{manifest.candidate_id}"
        compile_command = cpp_cpu_compile_command(workspace, binary)
        compile_started = time.perf_counter()
        compiled = subprocess.run(
            compile_command,
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
        )
        compile_seconds = time.perf_counter() - compile_started
        if compiled.returncode != 0 or not binary.is_file():
            detail = "\n".join(
                part.strip() for part in (compiled.stderr, compiled.stdout) if part
            ).strip()
            raise RuntimeError(
                f"C++ CPU evaluator failed to compile ({compiled.returncode}): "
                f"{shlex.join(compile_command)}\n{detail}"
            )
        command = [
            str(binary),
            *manifest_evaluator_args(manifest, output),
            "--compile-seconds",
            f"{compile_seconds:.6f}",
        ]
        return _run_local_evaluator(command, workspace, output)
    if manifest.language == "python-cpu":
        command = [
            sys.executable,
            str(workspace / "python" / "evaluate_candidate.py"),
            *manifest_evaluator_args(manifest, output),
        ]
        return _run_local_evaluator(command, workspace, output)
    if not is_julia_language(manifest.language):
        raise ValueError(
            f"local evaluation supports Julia, cpp-cpu, and python-cpu; got {manifest.language}"
        )
    command = [
        *julia_command(workspace, manifest.evaluation.cpu_threads),
        *manifest_evaluator_args(manifest, output),
    ]
    return _run_local_evaluator(command, workspace, output)


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
    repository = _normalize_abs(repository or project)
    relative_project = _normalize_abs(project).relative_to(repository)
    scratch = Path(config.scratch_root) / "kernel-agent-bench"
    setup = "\n".join(config.setup_commands)
    if is_julia_language(manifest.language):
        julia_project = f"{relative_project}/julia"
        setup = (
            setup
            + ("\n" if setup else "")
            + f"julia --project={shlex.quote(julia_project)} "
            + "-e 'using Pkg; Pkg.instantiate()'"
        )
    elif manifest.language == "cpp":
        setup = setup + ("\n" if setup else "") + "module load gcc"
    elif manifest.language == "python-triton":
        venv = config.triton_venv
        setup = "\n".join(
            [
                *([setup] if setup else []),
                *config.python_setup_commands,
                f'KAB_TRITON_VENV={venv}',
                'if ! "$KAB_TRITON_VENV/bin/python" -c "import numpy, torch, triton" '
                "2>/dev/null; then",
                '  rm -rf "$KAB_TRITON_VENV"',
                '  python3 -m venv "$KAB_TRITON_VENV"',
                '  "$KAB_TRITON_VENV/bin/pip" install --quiet --upgrade pip',
                '  "$KAB_TRITON_VENV/bin/pip" install --quiet -r '
                + shlex.quote(f"{relative_project}/python/requirements-gpu.txt"),
                "fi",
            ]
        )
    evaluator_args = manifest_evaluator_args(manifest, _normalize_abs(result))
    if is_julia_language(manifest.language):
        executable = [
            "julia",
            f"--project={relative_project}/julia",
            f"{relative_project}/julia/evaluate_candidate.jl",
        ]
        command = " ".join(shlex.quote(value) for value in executable + evaluator_args)
    elif manifest.language == "python-triton":
        command = (
            '"$KAB_TRITON_VENV/bin/python" '
            + shlex.quote(f"{relative_project}/python/evaluate_candidate.py")
            + " "
            + " ".join(shlex.quote(value) for value in evaluator_args)
        )
    elif manifest.language == "cpp":
        binary = f"{relative_project}/.build/evaluate_candidate_cpp"
        source = f"{relative_project}/cpp/evaluate_candidate.cpp"
        run_args = evaluator_args
        command = (
            f'mkdir -p "$(dirname {binary})"\n'
            "COMPILE_STARTED=$(date +%s%N)\n"
            f'g++ -O3 -std=c++17 -I{relative_project}/cpp '
            f'-I"${{CUDA_HOME:-/usr/local/cuda}}/include" '
            f"{source} -o {binary} "
            f'-L"${{CUDA_HOME:-/usr/local/cuda}}/lib64" -lOpenCL\n'
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

REPOSITORY={shlex.quote(repository.as_posix())}
PATCH={shlex.quote(_normalize_abs(Path(patch)).as_posix())}
WORKTREE={shlex.quote(_normalize_abs(scratch).as_posix())}/${{SLURM_JOB_ID}}-{shlex.quote(manifest.candidate_id)}
RESULT={shlex.quote(_normalize_abs(result).as_posix())}
METADATA={shlex.quote(_normalize_abs(metadata_path).as_posix())}

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
    if config.salloc_job_id:
        return " ".join(shlex.quote(value) for value in srun_command(config, script))
    return " ".join(shlex.quote(value) for value in sbatch_command(config, script))


def _ssh_stdout(config: OrcdConfig, remote: str) -> str:
    completed = subprocess.run(
        ssh_base(config, batch=True) + [remote],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        detail = "\n".join(
            part.strip() for part in (completed.stderr, completed.stdout) if part
        ).strip()
        raise RuntimeError(
            f"command failed ({completed.returncode}): "
            f"{shlex.join(ssh_base(config, batch=True) + [remote])}\n{detail}"
        )
    return completed.stdout.strip()


def _parse_sbatch_job_id(output: str) -> str:
    token = output.splitlines()[-1].strip() if output.strip() else ""
    job_id = token.split(";", 1)[0].strip()
    if not job_id.isdigit():
        raise RuntimeError(f"sbatch did not return a job id: {output!r}")
    return job_id


def _parse_salloc_job_id(output: str) -> str:
    for line in output.splitlines():
        match = re.search(r"(?:Pending job allocation|Granted job allocation)\s+(\d+)", line)
        if match:
            return match.group(1)
    raise RuntimeError(f"salloc did not print a job id: {output!r}")


def persist_salloc_job_id(local_project: Path, job_id: str) -> None:
    path = local_project / "config" / "orcd.yaml"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    replacement = f'salloc_job_id: "{job_id}"'
    updated, count = re.subn(r"^salloc_job_id:\s*.*$", replacement, text, count=1, flags=re.M)
    if count == 0:
        updated = text.rstrip() + "\n" + replacement + "\n"
    path.write_text(updated, encoding="utf-8")


def slurm_job_state(config: OrcdConfig, job_id: str) -> str:
    if not job_id:
        return ""
    try:
        output = _ssh_stdout(config, f"squeue -j {shlex.quote(job_id)} -h -o %T")
    except RuntimeError as exc:
        if "Invalid job id" in str(exc) or "slurm_load_jobs error" in str(exc):
            return ""
        raise
    return output.split()[0] if output else ""


def _submit_held_salloc(config: OrcdConfig) -> str:
    alloc = salloc_command(config, wall_time=HOLD_WALL_TIME)
    alloc[1:1] = ["--no-shell", "--job-name", HOLD_JOB_NAME]
    remote = (
        "set -euo pipefail; "
        'LOG="${HOME}/kab-salloc-h200.log"; '
        ': > "$LOG"; '
        "nohup "
        + " ".join(shlex.quote(part) for part in alloc)
        + ' >"$LOG" 2>&1 & '
        "for _ in $(seq 1 40); do "
        '  if grep -E "job allocation [0-9]+" "$LOG" >/dev/null 2>&1; then break; fi; '
        "  sleep 0.25; "
        "done; "
        'cat "$LOG"'
    )
    return _parse_salloc_job_id(_ssh_stdout(config, remote))


def ensure_held_salloc(config: OrcdConfig, local_project: Path) -> OrcdConfig:
    """Reuse one held H200 allocation (srun). Requires salloc_job_id or creates a new hold."""
    config = _orcd_with_disk_salloc(config, local_project)
    if not config.salloc_job_id.strip():
        raise RuntimeError("ensure_held_salloc requires salloc_job_id; leave it empty for sbatch")
    if slurm_job_state(config, config.salloc_job_id) in _HOLD_STATES:
        return config
    job_id = _submit_held_salloc(config)
    persist_salloc_job_id(local_project, job_id)
    return config.model_copy(update={"salloc_job_id": job_id})


def uses_held_salloc(config: OrcdConfig) -> bool:
    return bool(config.salloc_job_id.strip())


def wait_until_salloc_running(config: OrcdConfig, *, poll_seconds: int = 30) -> None:
    job_id = config.salloc_job_id
    if not job_id:
        raise RuntimeError("no salloc job id to wait for")
    while True:
        state = slurm_job_state(config, job_id)
        if state == "RUNNING":
            return
        if state in {"PENDING", "CONFIGURING", "REQUEUED"}:
            time.sleep(poll_seconds)
            continue
        raise RuntimeError(f"held salloc {job_id} ended in {state or 'missing'}")


def wait_for_slurm_job(config: OrcdConfig, job_id: str, *, poll_seconds: int = 30) -> str:
    """Poll with short SSH commands so the mux is not held open like salloc."""
    last_state = ""
    while True:
        try:
            last_state = _ssh_stdout(config, f"squeue -j {shlex.quote(job_id)} -h -o %T")
        except RuntimeError as exc:
            if "Invalid job id" in str(exc) or "slurm_load_jobs error" in str(exc):
                return "COMPLETED"
            probe_ssh(config)
            last_state = _ssh_stdout(config, f"squeue -j {shlex.quote(job_id)} -h -o %T")
        state = last_state.split()[0] if last_state else "COMPLETED"
        if state in _TERMINAL_SLURM or not last_state:
            return state
        time.sleep(poll_seconds)


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
        "# Submit the H200 eval (sbatch, or srun if salloc_job_id is set):",
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


def _orcd_with_disk_salloc(config: OrcdConfig, local_project: Path) -> OrcdConfig:
    """Pick up salloc_job_id written after a trial started."""
    path = local_project / "config" / "orcd.yaml"
    if not path.is_file():
        return config
    try:
        from .config import load_orcd

        job_id = load_orcd(path).salloc_job_id.strip()
    except Exception:
        return config
    if job_id and job_id != config.salloc_job_id:
        return config.model_copy(update={"salloc_job_id": job_id})
    return config


def evaluate_candidate_remote(
    config: OrcdConfig,
    bundle: Path,
    local_project: Path,
) -> Path:
    lock_path = local_project / ".kernel-agent-bench" / "h200.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _evaluate_candidate_remote_impl(config, bundle, local_project)


def _evaluate_candidate_remote_impl(
    config: OrcdConfig,
    bundle: Path,
    local_project: Path,
) -> Path:
    probe_ssh(config)
    config = _orcd_with_disk_salloc(config, local_project)
    held = uses_held_salloc(config)
    if held:
        config = ensure_held_salloc(config, local_project)
        wait_until_salloc_running(config)
    else:
        config = config.model_copy(update={"salloc_job_id": ""})
    manifest = CandidateManifest.model_validate_json(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )
    transfer = remote_transfer(config, bundle, local_project)
    transfer.local_result.parent.mkdir(parents=True, exist_ok=True)
    _run(ssh_base(config, batch=True) + [remote_mkdir_command(transfer)])
    try:
        _run(
            ssh_base(config, batch=True)
            + [remote_commit_command(transfer, manifest.baseline_commit)]
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"ORCD clone {transfer.remote_project} lacks baseline commit "
            f"{manifest.baseline_commit}; push it to origin first"
        ) from exc

    script = render_slurm(
        config,
        project=transfer.remote_project,
        bundle=bundle,
        result=transfer.remote_result,
        repository=transfer.remote_project,
        patch_path=transfer.remote_bundle / manifest.patch_file,
    )
    _run(rsync_upload_command(config, bundle, transfer))
    _run(
        ssh_base(config, batch=True)
        + [
            "cat > "
            + shlex.quote(str(transfer.remote_script))
            + " && chmod 750 "
            + shlex.quote(str(transfer.remote_script))
        ],
        input_text=script,
    )
    evaluate = (
        f"cd {shlex.quote(str(transfer.remote_project))} && "
        f"{remote_eval_command(config, str(transfer.remote_script))}"
    )
    eval_error: Exception | None = None
    if held:
        try:
            _run(ssh_base(config, batch=True) + [evaluate])
        except RuntimeError as exc:
            # Evaluators exit 2 when the candidate is incorrect but still write JSON.
            eval_error = exc
    else:
        try:
            job_id = _parse_sbatch_job_id(_ssh_stdout(config, evaluate))
            terminal = wait_for_slurm_job(config, job_id)
            if terminal != "COMPLETED":
                eval_error = RuntimeError(f"sbatch {job_id} ended in {terminal}")
        except RuntimeError as exc:
            eval_error = exc
    try:
        _run(rsync_download_command(config, transfer))
    except RuntimeError:
        if eval_error is not None:
            raise eval_error from None
        raise
    if transfer.local_result.is_file():
        return transfer.local_result
    if eval_error is not None:
        raise eval_error
    raise FileNotFoundError(f"H200 result was not downloaded: {transfer.local_result}")
