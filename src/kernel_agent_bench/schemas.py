from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "2.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSelector(StrictModel):
    alias: str
    match: str
    newest: bool = True


class Budgets(StrictModel):
    max_agent_iterations: int = Field(ge=1, le=5)
    max_h200_evaluations: int = Field(ge=1, le=5)
    max_total_tokens: int = Field(ge=1)
    max_wall_seconds: int = Field(ge=1)


class EvaluationConfig(StrictModel):
    dtype: str = "Float32"
    correctness_shapes: list[tuple[int, int]]
    performance_shapes: list[tuple[int, int]]
    steps: int = Field(ge=1)
    warmups: int = Field(ge=0)
    repetitions: int = Field(ge=1)
    atol: float = Field(ge=0)
    rtol: float = Field(ge=0)
    max_regression_fraction: float = Field(ge=0, lt=1)
    min_relative_improvement: float = Field(ge=0, lt=1)


class LanguageTarget(StrictModel):
    id: str
    editable_paths: list[str]
    public_test_command: str
    prompt_path: str


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
    ntasks: int = Field(default=32, ge=1)
    cpus_per_task: int = Field(default=1, ge=1)
    memory: str = "128GB"
    wall_time: str = "06:00:00"
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


class EvaluationResult(StrictModel):
    schema_version: str = SCHEMA_VERSION
    candidate_id: str
    language: str
    evaluator_version: str
    status: str
    correct: bool
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
