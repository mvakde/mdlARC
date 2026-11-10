"""Create and independently validate one isolated per-puzzle Git workspace."""

from __future__ import annotations

import hashlib
import json
import secrets
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from task_variations.generate.artifacts import atomic_write_text
from task_variations.generate.generator_validation import (
    GeneratorValidation,
    validate_generator_code,
)


EDITABLE_FILE = "generator.py"
SELF_TEST_TIMEOUT_SECONDS = 300
MAX_CAPTURE_CHARACTERS = 20_000


@dataclass(frozen=True)
class WorkspaceValidation:
    valid: bool
    errors: list[str]
    static_validation: Optional[GeneratorValidation]
    execution_performed: bool
    self_test_returncode: Optional[int]
    self_test_stdout: str
    self_test_stderr: str
    changed_paths: list[str]
    baseline_commit: str
    current_commit: Optional[str]
    host_verification: Optional[Dict[str, Any]] = None

    @property
    def code_sha256(self) -> Optional[str]:
        if self.static_validation is None:
            return None
        return self.static_validation.code_sha256

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "code_sha256": self.code_sha256,
            "workspace_boundary": {
                "baseline_commit": self.baseline_commit,
                "current_commit": self.current_commit,
                "changed_paths": self.changed_paths,
                "allowed_changed_paths": [EDITABLE_FILE],
                "scratch_directory": "scratch/",
            },
            "static_validation": (
                self.static_validation.to_dict()
                if self.static_validation is not None
                else None
            ),
            "execution_performed": self.execution_performed,
            "host_verification": self.host_verification,
            "self_test": {
                "returncode": self.self_test_returncode,
                "stdout": self.self_test_stdout,
                "stderr": self.self_test_stderr,
                "timeout_seconds": SELF_TEST_TIMEOUT_SECONDS,
            },
            "validation_mode": (
                "workspace boundary, static checks, independent output/count/diversity checks, optional reference solver, and self_test()"
            ),
        }


def _run_git(workspace_dir: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(workspace_dir), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "git %s failed in %s: %s"
            % (" ".join(arguments), workspace_dir, completed.stderr.strip())
        )
    return completed.stdout


def _validate_workspace_file_name(name: str) -> None:
    path = Path(name)
    if path.is_absolute() or len(path.parts) != 1 or name in ("", ".", "..", ".git"):
        raise ValueError("Workspace file names must be simple relative names: %r" % name)


def stage_git_workspace(
    workspace_dir: Path,
    files: Mapping[str, str],
    *,
    allow_matching_existing: bool = False,
) -> str:
    """Write an exact file set and commit it as the immutable workspace baseline."""
    for name in files:
        _validate_workspace_file_name(name)

    git_dir = workspace_dir / ".git"
    if git_dir.exists():
        if not allow_matching_existing:
            raise FileExistsError("Workspace is already a Git repository: %s" % workspace_dir)
        for name, content in files.items():
            path = workspace_dir / name
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise ValueError("Existing dry-run workspace does not match %s" % name)
        if _run_git(workspace_dir, "status", "--porcelain=v1", "--untracked-files=all").strip():
            raise ValueError("Existing dry-run workspace has local changes: %s" % workspace_dir)
        scratch = workspace_dir / "scratch"
        if scratch.is_symlink() or (scratch.exists() and not scratch.is_dir()):
            raise ValueError("Existing scratch path must be a regular directory")
        scratch.mkdir(exist_ok=True)
        return _run_git(workspace_dir, "rev-parse", "HEAD").strip()

    if any(workspace_dir.iterdir()):
        raise ValueError("Workspace must be empty before staging: %s" % workspace_dir)
    for name, content in files.items():
        atomic_write_text(workspace_dir / name, content)
    _run_git(workspace_dir, "init", "--quiet")
    _run_git(workspace_dir, "add", "--all")
    _run_git(
        workspace_dir,
        "-c",
        "user.name=Codex Workspace",
        "-c",
        "user.email=codex-workspace@local",
        "commit",
        "--quiet",
        "--no-gpg-sign",
        "-m",
        "Stage reviewed generator inputs",
    )
    (workspace_dir / "scratch").mkdir(exist_ok=True)
    return _run_git(workspace_dir, "rev-parse", "HEAD").strip()


def _allowed_change(path: str) -> bool:
    return path == EDITABLE_FILE or path.startswith("scratch/")


def _check_files(workspace_dir: Path, baseline_commit: str) -> list[str]:
    """Check protected bytes as well as Git status, including ignored symlinks."""
    errors = []
    names = _run_git(workspace_dir, "ls-tree", "-r", "--name-only", baseline_commit).splitlines()
    for name in names:
        path = workspace_dir / name
        if path.is_symlink() or not path.is_file():
            errors.append(f"{name} must remain a regular file.")
        elif name != EDITABLE_FILE:
            expected = _run_git(workspace_dir, "show", f"{baseline_commit}:{name}")
            if path.read_text(encoding="utf-8") != expected:
                errors.append(f"Protected file changed: {name}")
    scratch = workspace_dir / "scratch"
    if scratch.exists() and not scratch.is_dir():
        errors.append("scratch must be a directory.")
    for path in workspace_dir.rglob("*"):
        relative = path.relative_to(workspace_dir)
        if relative.parts[0] == ".git":
            continue
        if path.is_symlink():
            errors.append(f"Workspace symlinks are not allowed: {relative}")
    return errors


def workspace_diff(workspace_dir: Path, base_revision: str = "HEAD") -> str:
    return _run_git(
        workspace_dir,
        "diff",
        base_revision,
        "--no-ext-diff",
        "--binary",
        "--",
        EDITABLE_FILE,
    )


def _bounded(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    elif not isinstance(value, str):
        value = str(value)
    if len(value) <= MAX_CAPTURE_CHARACTERS:
        return value
    return value[:MAX_CAPTURE_CHARACTERS] + "\n...[output truncated by host]...\n"


def _changed_paths(workspace_dir: Path) -> list[str]:
    output = _run_git(
        workspace_dir,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    paths: list[str] = []
    for line in output.splitlines():
        value = line[3:] if len(line) >= 4 else line
        if " -> " in value:
            paths.extend(value.split(" -> ", 1))
        elif value:
            paths.append(value)
    return sorted(set(paths))


def validate_workspace(
    workspace_dir: Path,
    baseline_commit: str,
    source_generator_sha256: str,
) -> WorkspaceValidation:
    errors: list[str] = []
    try:
        current_commit = _run_git(workspace_dir, "rev-parse", "HEAD").strip()
        changed_paths = _changed_paths(workspace_dir)
    except Exception as exc:
        return WorkspaceValidation(
            False,
            ["Could not inspect workspace Git state: %s" % exc],
            None,
            False,
            None,
            "",
            "",
            [],
            baseline_commit,
            None,
        )

    if current_commit != baseline_commit:
        errors.append("Do not create commits or rewrite the workspace baseline.")
    errors.extend(_check_files(workspace_dir, baseline_commit))
    unexpected = [path for path in changed_paths if not _allowed_change(path)]
    if unexpected:
        errors.append("Only generator.py and scratch/ may change; unexpected paths: %s" % ", ".join(unexpected))
    if EDITABLE_FILE not in changed_paths:
        errors.append("generator.py was not changed from the reviewed baseline.")

    generator_path = workspace_dir / EDITABLE_FILE
    static_validation: Optional[GeneratorValidation] = None
    if generator_path.is_symlink():
        errors.append("generator.py must remain a regular file, not a symbolic link.")
    elif not generator_path.is_file():
        errors.append("generator.py is missing.")
    else:
        try:
            source = generator_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append("Could not read generator.py: %s" % exc)
        else:
            source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
            if source_sha256 == source_generator_sha256:
                errors.append("generator.py still has the exact reviewed source hash.")
            static_validation = validate_generator_code(source)
            errors.extend(static_validation.errors)

    execution_performed = False
    returncode: Optional[int] = None
    stdout = ""
    stderr = ""
    host_verification = None
    if not errors and static_validation is not None and static_validation.valid:
        report_path = workspace_dir / "scratch" / "verification.json"
        report_path.parent.mkdir(exist_ok=True)
        report_path.unlink(missing_ok=True)
        try:
            completed = subprocess.run(
                [sys.executable, "verify.py", "--seed-base", str(secrets.randbits(63))],
                cwd=str(workspace_dir),
                check=False,
                capture_output=True,
                text=True,
                timeout=SELF_TEST_TIMEOUT_SECONDS,
            )
            execution_performed = True
            returncode = completed.returncode
            stdout = _bounded(completed.stdout)
            stderr = _bounded(completed.stderr)
            if report_path.is_file() and not report_path.is_symlink():
                host_verification = json.loads(report_path.read_text(encoding="utf-8"))
            if completed.returncode != 0:
                detail = stderr.strip() or stdout.strip()
                errors.append("verify.py failed%s" % ((": " + detail) if detail else "."))
            elif not isinstance(host_verification, dict) or host_verification.get("valid") is not True:
                errors.append("verify.py did not produce a passing independent verification report.")
        except subprocess.TimeoutExpired as exc:
            execution_performed = True
            stdout = _bounded(exc.stdout or "")
            stderr = _bounded(exc.stderr or "")
            errors.append(
                "verify.py exceeded the %d second timeout." % SELF_TEST_TIMEOUT_SECONDS
            )

        try:
            post_commit = _run_git(workspace_dir, "rev-parse", "HEAD").strip()
            post_changed_paths = _changed_paths(workspace_dir)
            errors.extend(_check_files(workspace_dir, baseline_commit))
            changed_paths = sorted(set(changed_paths + post_changed_paths))
            if post_commit != baseline_commit:
                errors.append("self_test changed the workspace Git baseline.")
            post_unexpected = [path for path in post_changed_paths if not _allowed_change(path)]
            if post_unexpected:
                errors.append(
                    "self_test changed unexpected paths: %s" % ", ".join(post_unexpected)
                )
            if generator_path.is_symlink() or not generator_path.is_file():
                errors.append("self_test removed or replaced generator.py.")
            else:
                post_source = generator_path.read_text(encoding="utf-8")
                post_sha256 = hashlib.sha256(post_source.encode("utf-8")).hexdigest()
                if post_sha256 != static_validation.code_sha256:
                    errors.append("self_test modified generator.py after static validation.")
        except Exception as exc:
            errors.append("Could not re-check workspace after self_test: %s" % exc)

    return WorkspaceValidation(
        valid=not errors,
        errors=list(dict.fromkeys(errors)),
        static_validation=static_validation,
        execution_performed=execution_performed,
        self_test_returncode=returncode,
        self_test_stdout=stdout,
        self_test_stderr=stderr,
        changed_paths=changed_paths,
        baseline_commit=baseline_commit,
        current_commit=current_commit,
        host_verification=host_verification,
    )
