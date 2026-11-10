from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "3.0"

JULIA_BACKENDS = {
    "julia": "cuda",
    "julia-cpu": "cpu",
    "julia-cuda": "cuda",
    "julia-ka": "ka",
}

PYTHON_BACKENDS = {
    "python-cpu": "cpu",
    "python-triton": "triton",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSelector(StrictModel):
    alias: str
    match: str
    newest: bool = True
    # cursor: resolve `match` against Cursor.models.list() and run the Cursor agent.
    # anthropic: call `model_id` with ANTHROPIC_API_KEY. Never send that trial to Cursor.
    # openai: call `model_id` through the Codex CLI with OPENAI_API_KEY. Never Cursor.
    provider: Literal["cursor", "anthropic", "openai"] = "cursor"
    model_id: str = ""
    # Effort is the only reasoning-depth control on both direct APIs; set it
    # explicitly so a trial does not follow a later API default.
    effort: Literal["", "minimal", "low", "medium", "high", "xhigh", "max"] = ""

    @model_validator(mode="after")
    def direct_api_models_are_literal(self) -> ModelSelector:
        if self.provider in {"anthropic", "openai"}:
            api = "Anthropic" if self.provider == "anthropic" else "OpenAI"
            if not self.model_id or any(char in self.model_id for char in " ^$[]()*+?\\"):
                raise ValueError(f"{self.alias} uses the {api} API and needs a literal model_id")
            if not self.effort:
                raise ValueError(f"{self.alias} needs an explicit {api} effort level")
            if self.provider == "anthropic" and self.effort == "minimal":
                raise ValueError(f"{self.alias}: Anthropic has no 'minimal' effort")
            if self.provider == "openai" and self.effort == "max":
                raise ValueError(f"{self.alias}: OpenAI has no 'max' effort")
            return self
        if self.model_id or self.effort:
            raise ValueError("model_id and effort are only valid for direct API providers")
        return self


class Budgets(StrictModel):
    max_agent_iterations: int = Field(ge=1, le=5)
    max_h200_evaluations: int = Field(ge=1, le=5)
    max_total_tokens: int = Field(ge=1)
    max_wall_seconds: int = Field(ge=1)
    no_improvement_streak_to_stop: int = Field(default=2, ge=1, le=5)


class EvaluationConfig(StrictModel):
    dtype: str = "Float64"
    correctness_shapes: list[tuple[int, int]]
    performance_shapes: list[tuple[int, int]]
    steps: int = Field(ge=1)
    warmups: int = Field(ge=0)
    repetitions: int = Field(ge=1)
    atol: float = Field(ge=0)
    rtol: float = Field(ge=0)
    max_regression_fraction: float = Field(ge=0, lt=1)
    min_relative_improvement: float = Field(ge=0, lt=1)
    # The notebook baseline is Float64 only and doubles as the correctness oracle,
    # so `atol`/`rtol` bound candidate-vs-baseline differences. The Float32 and
    # float64_* fields remain so pre-v3 manifests still validate.
    correctness_dtypes: list[Literal["Float32", "Float64"]] = Field(
        default=["Float64"], min_length=1
    )
    float64_atol: float = Field(default=1e-10, ge=0)
    float64_rtol: float = Field(default=1e-8, ge=0)
    # Worker threads for CPU tracks (Julia --threads, C++ pool, Numba); 0 = all cores.
    cpu_threads: int = Field(default=0, ge=0)


def _path_covered(needed: str, allowed: str) -> bool:
    needed_n = needed.strip("/")
    allowed_n = allowed.strip("/")
    return needed_n == allowed_n or needed_n.startswith(allowed_n + "/")


class LanguageTarget(StrictModel):
    id: str
    editable_paths: list[str]
    public_test_command: str
    prompt_path: str
    backend: Literal["cpu", "cuda", "ka", "opencl", "triton"] = "ka"
    requires_gpu: bool = True
    evaluate_locally: bool = False
    workspace_paths: list[str] = Field(default_factory=list)
    evaluation: EvaluationConfig | None = None

    @model_validator(mode="after")
    def workspace_covers_editable_and_prompt(self) -> LanguageTarget:
        if self.evaluate_locally and self.requires_gpu:
            raise ValueError(f"{self.id} cannot evaluate locally while requiring a GPU")
        if not self.workspace_paths:
            return self
        required = [*self.editable_paths, self.prompt_path]
        for needed in required:
            if not any(_path_covered(needed, allowed) for allowed in self.workspace_paths):
                raise ValueError(
                    f"{self.id} workspace_paths must include {needed}"
                )
        return self


def julia_backend_for_language(language: str) -> str | None:
    return JULIA_BACKENDS.get(language)


def is_julia_language(language: str) -> bool:
    return language in JULIA_BACKENDS


def python_backend_for_language(language: str) -> str | None:
    return PYTHON_BACKENDS.get(language)


class StudyConfig(StrictModel):
    study_id: str
    task_version: str
    seed: int
    replicates: int = Field(ge=1)
    trace_content: bool = True
    models: list[ModelSelector]
    languages: list[LanguageTarget]
    budgets: Budgets
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def matrix_entries_are_unique(self) -> StudyConfig:
        model_aliases = [model.alias for model in self.models]
        language_ids = [language.id for language in self.languages]
        if len(model_aliases) != len(set(model_aliases)):
            raise ValueError("model aliases must be unique")
        if len(language_ids) != len(set(language_ids)):
            raise ValueError("language IDs must be unique")
        return self


class OrcdConfig(StrictModel):
    partition: str = "mit_normal_gpu"
    gpu_request: str = "h200:1"
    nodes: int = Field(default=1, ge=1)
    ntasks: int = Field(default=1, ge=1)
    cpus_per_task: int = Field(default=1, ge=1)
    memory: str = "8GB"
    wall_time: str = "00:20:00"
    account: str = ""
    qos: str = ""
    reservation: str = ""
    email: str = ""
    scratch_root: str
    ssh_host: str
    ssh_user: str
    remote_project: str
    ssh_port: int = Field(default=22, ge=1, le=65535)
    identity_file: str = ""
    auto_evaluate: bool = True
    salloc_job_id: str = ""
    setup_commands: list[str] = Field(default_factory=list)
    # python-triton jobs: commands that put python3 >= 3.10 on PATH, and the venv
    # (created on first use from python/requirements-gpu.txt) that holds torch/triton.
    python_setup_commands: list[str] = Field(default_factory=lambda: ["module load miniforge"])
    triton_venv: str = "$HOME/.kab-triton-venv"
    container_image: str = ""

    @model_validator(mode="after")
    def required_values_are_filled(self) -> OrcdConfig:
        missing = [
            key
            for key in ("scratch_root", "ssh_host", "ssh_user", "remote_project")
            if getattr(self, key).strip() in {"", "REQUIRED"}
        ]
        if missing:
            raise ValueError(f"fill required ORCD fields: {', '.join(missing)}")
        for label, value in (
            ("scratch_root", self.scratch_root),
            ("remote_project", self.remote_project),
        ):
            path = Path(value)
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{label} must be an absolute path without '..'")
            if path == Path("/"):
                raise ValueError(f"{label} cannot be the filesystem root")
        remote_project = Path(self.remote_project)
        if remote_project.name != "KernelAgentBench":
            raise ValueError("remote_project must be the KernelAgentBench directory on ORCD")
        if len(remote_project.parts) < 4:
            raise ValueError("remote_project is too close to the filesystem root")
        return self


class CandidateManifest(StrictModel):
    schema_version: str = SCHEMA_VERSION
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
    trial_id: str
    parent_candidate_id: str | None = None
    baseline_commit: str
    patch_sha256: str
    patch_file: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    iteration: int
    language: str
    task_version: str
    study_hash: str
    prompt_sha256: str
    evaluation: EvaluationConfig


class TokenRecord(StrictModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = None
    available: bool = True


class WorkloadResult(StrictModel):
    shape: tuple[int, int]
    baseline_samples_ms: list[float]
    candidate_samples_ms: list[float]
    baseline_median_ms: float
    candidate_median_ms: float
    baseline_std_ms: float = 0
    candidate_std_ms: float = 0
    speedup: float
    correct: bool
    max_abs_error: float
    relative_l2_error: float
    timer: str | None = None


class EvaluationResult(StrictModel):
    schema_version: str = SCHEMA_VERSION
    candidate_id: str
    language: str
    evaluator_version: str
    status: str
    correct: bool
    backend: str | None = None
    correctness_cases: list[dict[str, Any]] = Field(default_factory=list)
    workloads: list[WorkloadResult] = Field(default_factory=list)
    geometric_mean_speedup: float | None = None
    compile_seconds: float | None = None
    end_to_end_seconds: float | None = None
    environment: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    result_sha256: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TrialRecord(StrictModel):
    schema_version: str = SCHEMA_VERSION
    study_id: str
    trial_id: str
    model_alias: str
    resolved_model_id: str
    language: str
    replicate: int
    status: str
    iteration: int
    h200_evaluations: int
    token_usage: TokenRecord
    incumbent_candidate_id: str | None = None
    incumbent_speedup: float | None = None
    first_correct_iteration: int | None = Field(
        default=None,
        description="First agent iteration whose H200 evaluation passed the full correctness gate.",
    )
    tokens_to_correctness: int | None = Field(
        default=None,
        description="Cumulative agent tokens through the first fully correct H200 evaluation.",
    )
    termination_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def json_schema_bundle() -> dict[str, Any]:
    return {
        "study": StudyConfig.model_json_schema(),
        "candidate": CandidateManifest.model_json_schema(),
        "evaluation": EvaluationResult.model_json_schema(),
        "trial": TrialRecord.model_json_schema(),
    }


def ensure_relative_paths(paths: list[str], root: Path) -> list[Path]:
    resolved: list[Path] = []
    for value in paths:
        candidate = (root / value).resolve()
        if not candidate.is_relative_to(root.resolve()):
            raise ValueError(f"path escapes project root: {value}")
        resolved.append(candidate)
    return resolved
