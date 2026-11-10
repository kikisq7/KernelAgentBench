from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from .schemas import OrcdConfig, StudyConfig

T = TypeVar("T", bound=BaseModel)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_project_env(*, override: bool = False) -> None:
    """Load repo-root `.env` if present. Does not override existing env vars by default."""
    path = project_root() / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value


def load_model(path: Path, model: type[T]) -> T:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return model.model_validate(data)


def load_study(path: Path) -> StudyConfig:
    return load_model(path, StudyConfig)


def load_orcd(path: Path) -> OrcdConfig:
    return load_model(path, OrcdConfig)


def canonical_hash(value: BaseModel | dict[str, object]) -> str:
    raw = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
