"""Content manifests for prepared MLE task data and controls."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Returns the SHA-256 of one regular file without following a symlink."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"manifest input is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(root: Path) -> dict[str, Any]:
    """Builds a deterministic file inventory beneath one task directory."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"manifest root is not a directory: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"manifest tree contains a symbolic link: {path}")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "file_count": len(files),
        "size_bytes": sum(int(item["size_bytes"]) for item in files),
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def atomic_json(path: Path, payload: object) -> None:
    """Atomically writes a stable JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def verify_tree(root: Path, expected: dict[str, Any]) -> bool:
    """Checks a tree against a previously frozen full-content manifest."""
    actual = tree_manifest(root)
    return all(
        actual.get(field) == expected.get(field)
        for field in ("file_count", "size_bytes", "tree_sha256", "files")
    )
