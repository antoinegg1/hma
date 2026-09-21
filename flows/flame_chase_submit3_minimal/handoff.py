"""Deterministic allowlist export. Never execute or semantically classify files."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any

CODE = frozenset({".py", ".sh"})
CONFIG = frozenset({".json", ".yaml", ".yml", ".toml"})
POLICY = {
    "solution": CODE,
    "src": CODE,
    "scripts": CODE,
    "configs": CONFIG,
    "artifacts/checkpoints": frozenset(
        {
            ".pt",
            ".pth",
            ".safetensors",
            ".ckpt",
            ".bin",
            ".pkl",
            ".cbm",
            ".ubj",
            ".json",
        }
    ),
    "artifacts/predictions": frozenset({".csv", ".npy", ".npz", ".parquet"}),
    "artifacts/features": frozenset({".npy", ".npz", ".parquet"}),
}
EXACT = frozenset({"requirements.txt", "pyproject.toml", "uv.lock", "poetry.lock"})
DENIED_NAMES = frozenset(
    {
        "logs",
        "log",
        "analysis",
        "reports",
        "report",
        "notes",
        "tensorboard",
        "wandb",
        "__pycache__",
    }
)
DENIED_EXTENSIONS = frozenset(
    {".md", ".log", ".ipynb", ".html", ".png", ".svg", ".pdf", ".jsonl"}
)


def rule_for(relative: Path) -> str | None:
    """Deny rules win; only explicit paths and extensions are exported."""
    if relative.is_absolute() or ".." in relative.parts:
        return None
    if any(
        part.startswith(".") or part.lower() in DENIED_NAMES for part in relative.parts
    ):
        return None
    if relative.suffix.lower() in DENIED_EXTENSIONS:
        return None
    if relative.as_posix() in EXACT:
        return "exact"
    for prefix, extensions in POLICY.items():
        if relative.is_relative_to(prefix) and relative.suffix.lower() in extensions:
            return prefix
    return None


def atomic_json(path: Path, value: Any) -> None:
    """Persist protected controller metadata with an atomic, durable rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(name, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def copy_verified(
    source: Path, target: Path, *, deadline: float | None = None
) -> dict[str, Any]:
    """Copy regular non-symlink files, break hard links, verify the destination."""
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as incoming:
        metadata = os.fstat(incoming.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"not a regular file: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        hashed = hashlib.sha256()
        with target.open("xb") as outgoing:
            while chunk := incoming.read(1024 * 1024):
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError("global deadline during handoff")
                outgoing.write(chunk)
                hashed.update(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        after = os.fstat(incoming.fileno())
        if (metadata.st_size, metadata.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"file changed during handoff: {source}")
    with target.open("rb") as copied:
        target_hash = hashlib.sha256()
        while chunk := copied.read(1024 * 1024):
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError("global deadline during verification")
            target_hash.update(chunk)
        verified = target_hash.hexdigest()
    if verified != hashed.hexdigest():
        raise RuntimeError(f"handoff hash mismatch: {target}")
    target.chmod(0o755 if metadata.st_mode & 0o111 else 0o644)
    return {"sha256": verified, "size": metadata.st_size}


def validate_config(path: Path) -> None:
    """Only syntax validation, not an LLM or a claim of semantic sanitization."""
    if path.suffix.lower() == ".json":
        json.loads(path.read_text())
    elif path.suffix.lower() == ".toml":
        tomllib.loads(path.read_text())
    else:
        import yaml

        yaml.safe_load(path.read_text())


def export_workspace(
    source: Path,
    destination: Path,
    *,
    latest_candidate: Path | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Publish a complete new workspace only after every admitted file verifies.

    Caller MUST stop the source container first. The original workspace is never
    deleted, nor is its HOME mounted in the successor. Failed staging remains for
    audit and is never published. The manifest is returned to the controller.
    """
    if source.is_symlink() or not source.is_dir() or destination.exists():
        raise ValueError("source must be a stopped workspace; destination must be new")
    source = source.resolve()
    if destination.resolve().is_relative_to(source):
        raise ValueError("handoff destination must be outside source workspace")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".handoff-", dir=destination.parent))
    files: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for parent, directories, names in os.walk(source, followlinks=False):
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("global deadline during export")
        directory = Path(parent)
        for name in list(directories):
            path = directory / name
            relative = path.relative_to(source)
            if path.is_symlink() or any(
                p.startswith(".") or p.lower() in DENIED_NAMES for p in relative.parts
            ):
                directories.remove(name)
                rejected.append(
                    {"path": str(relative), "reason": "denied_directory_or_link"}
                )
        for name in sorted(names):
            path = directory / name
            relative = path.relative_to(source)
            matched = rule_for(relative)
            if (
                path.is_symlink()
                or not stat.S_ISREG(path.lstat().st_mode)
                or matched is None
            ):
                rejected.append(
                    {"path": str(relative), "reason": "not_allowlisted_regular_file"}
                )
                continue
            if matched == "configs":
                validate_config(path)
            metadata = copy_verified(path, staging / relative, deadline=deadline)
            files.append({"path": str(relative), "rule": matched, **metadata})
    if latest_candidate is not None:
        metadata = copy_verified(
            latest_candidate, staging / "solution/submission.csv", deadline=deadline
        )
        files.append(
            {
                "path": "solution/submission.csv",
                "rule": "evaluator_latest_accepted",
                **metadata,
            }
        )
    os.replace(staging, destination)
    return {
        "policy": "artifacts_only_v1",
        "files": sorted(files, key=lambda r: r["path"]),
        "excluded": rejected,
    }
