from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .schemas import CandidateManifest, EvaluationConfig, ensure_relative_paths

TRIAL_INPUT_PATHS = ("kernels", "julia", "prompts")


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        text=True,
        capture_output=True,
    )


def repository_root(project: Path) -> Path:
    result = _git(project, "rev-parse", "--show-toplevel")
    return Path(result.stdout.strip()).resolve()


@dataclass(frozen=True)
class Worktree:
    repository: Path
    root: Path
    project: Path
    baseline_commit: str


class CandidateManager:
    def __init__(self, project: Path) -> None:
        self.project = project.resolve()
        self.repository = repository_root(self.project)
        self.relative_project = self.project.relative_to(self.repository)
        self.state_root = self.project / ".kernel-agent-bench"
        self.worktrees = self.state_root / "worktrees"
        self.bundles = self.project / "candidates"

    def require_committed(self, paths: tuple[str, ...] = TRIAL_INPUT_PATHS) -> None:
        """Trials run the committed tree, so local edits here would be silently ignored."""
        existing = [path for path in paths if (self.project / path).exists()]
        if not existing:
            return
        output = _git(
            self.project, "status", "--porcelain", "--untracked-files=all", "--", *existing
        ).stdout
        dirty = sorted(line[3:] for line in output.splitlines() if len(line) >= 4)
        if dirty:
            raise RuntimeError(
                "commit (and push, for ORCD) these trial inputs first; trials run the "
                "committed baseline and would ignore them: " + ", ".join(dirty)
            )

    def create_worktree(self, trial_id: str, baseline_commit: str = "HEAD") -> Worktree:
        if baseline_commit == "HEAD":
            self.require_committed()
        resolved_commit = _git(self.repository, "rev-parse", baseline_commit).stdout.strip()
        destination = self.worktrees / trial_id
        if destination.exists():
            self.cleanup_worktree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _git(
            self.repository,
            "worktree",
            "add",
            "--detach",
            str(destination),
            resolved_commit,
        )
        project = destination / self.relative_project
        if not project.exists():
            self.cleanup_worktree(destination)
            raise RuntimeError(
                "KernelAgentBench is absent from the baseline commit; commit the scaffold "
                "before starting real agent trials"
            )
        return Worktree(self.repository, destination, project, resolved_commit)

    def cleanup_worktree(self, worktree_root: Path) -> None:
        resolved = worktree_root.resolve()
        if not resolved.is_relative_to(self.worktrees.resolve()):
            raise ValueError(f"refusing to remove unmanaged worktree: {resolved}")
        _git(self.repository, "worktree", "remove", "--force", str(resolved), check=False)
        if resolved.exists():
            shutil.rmtree(resolved)

    def materialize_agent_workspace(
        self,
        worktree: Worktree,
        workspace_paths: list[str],
        trial_id: str,
    ) -> Path:
        if not workspace_paths:
            return worktree.project
        ensure_relative_paths(workspace_paths, worktree.project)
        dest = Path(tempfile.mkdtemp(prefix=f"kab-{trial_id}-"))
        try:
            for relative in workspace_paths:
                src = worktree.project / relative
                if not src.exists():
                    raise FileNotFoundError(f"workspace path missing from baseline: {relative}")
                target = dest / relative
                if src.is_dir():
                    shutil.copytree(src, target, symlinks=False)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, target)
        except Exception:
            shutil.rmtree(dest, ignore_errors=True)
            raise
        return dest

    def sync_editable_from_agent(
        self,
        worktree: Worktree,
        agent_workspace: Path,
        editable_paths: list[str],
    ) -> None:
        if agent_workspace.resolve() == worktree.project.resolve():
            return
        for relative in editable_paths:
            src = agent_workspace / relative
            dest = worktree.project / relative
            if not src.is_file():
                raise ValueError(f"agent did not produce {relative}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    def cleanup_agent_workspace(self, agent_workspace: Path, project: Path) -> None:
        resolved = agent_workspace.resolve()
        if resolved == project.resolve() or not resolved.exists():
            return
        tmp = Path(tempfile.gettempdir()).resolve()
        if not resolved.is_relative_to(tmp) or not resolved.name.startswith("kab-"):
            raise ValueError(f"refusing to remove unmanaged agent workspace: {resolved}")
        shutil.rmtree(resolved)

    def changed_paths(self, worktree: Worktree) -> list[str]:
        output = _git(worktree.root, "status", "--porcelain", "--untracked-files=all").stdout
        paths: list[str] = []
        for line in output.splitlines():
            if len(line) >= 4:
                value = line[3:]
                if " -> " in value:
                    value = value.split(" -> ", 1)[1]
                paths.append(value)
        return paths

    def validate_changes(self, worktree: Worktree, editable_paths: list[str]) -> None:
        allowed = {str((self.relative_project / path).as_posix()) for path in editable_paths}
        changed = set(self.changed_paths(worktree))
        disallowed = sorted(changed - allowed)
        if disallowed:
            raise ValueError(f"candidate changed disallowed paths: {', '.join(disallowed)}")
        if not changed:
            raise ValueError("candidate did not change any editable file")

    def package(
        self,
        worktree: Worktree,
        trial_id: str,
        iteration: int,
        editable_paths: list[str],
        parent_candidate_id: str | None,
        language: str,
        task_version: str,
        study_hash: str,
        prompt: str,
        evaluation: EvaluationConfig,
    ) -> tuple[CandidateManifest, Path]:
        self.validate_changes(worktree, editable_paths)
        patch = _git(worktree.root, "diff", "--binary", "HEAD", "--").stdout.encode()
        patch_sha = hashlib.sha256(patch).hexdigest()
        candidate_id = f"{trial_id}-i{iteration}-{patch_sha[:12]}"
        bundle = self.bundles / candidate_id
        bundle.mkdir(parents=True, exist_ok=False)
        patch_path = bundle / "candidate.patch"
        patch_path.write_bytes(patch)
        manifest = CandidateManifest(
            candidate_id=candidate_id,
            trial_id=trial_id,
            parent_candidate_id=parent_candidate_id,
            baseline_commit=worktree.baseline_commit,
            patch_sha256=patch_sha,
            patch_file=patch_path.name,
            iteration=iteration,
            language=language,
            task_version=task_version,
            study_hash=study_hash,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            evaluation=evaluation,
        )
        (bundle / "manifest.json").write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest, bundle

    def restore_candidate(
        self,
        worktree: Worktree,
        editable_paths: list[str],
        candidate_id: str | None,
    ) -> None:
        for relative in editable_paths:
            repository_path = (self.relative_project / relative).as_posix()
            content = _git(worktree.root, "show", f"HEAD:{repository_path}").stdout
            destination = worktree.project / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        if candidate_id:
            patch = self.bundles / candidate_id / "candidate.patch"
            if not patch.is_file():
                raise FileNotFoundError(f"incumbent patch is missing: {patch}")
            _git(worktree.root, "apply", "--check", str(patch))
            _git(worktree.root, "apply", str(patch))
