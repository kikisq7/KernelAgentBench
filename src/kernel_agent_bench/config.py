from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from .schemas import OrcdConfig, StudyConfig

T = TypeVar("T", bound=BaseModel)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
