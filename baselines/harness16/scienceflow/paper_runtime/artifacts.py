"""Public-only validation and atomic final artifact publication."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    """Hash regular bytes without accepting a symlink at the leaf."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"not a regular file: {path}")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Publish JSON using a same-directory atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_csv(path: Path, public: Path) -> tuple[bool, str]:
    """Check CSV structure against public sample only, never target values."""
    samples = sorted(public.glob("*sample*submission*.csv"))
    try:

        def shape(candidate: Path) -> tuple[list[str], int]:
            with candidate.open(newline="", encoding="utf-8-sig") as stream:
                rows = csv.reader(stream)
                header = next(rows)
                if (
                    not header
                    or len(set(header)) != len(header)
                    or any(not v.strip() for v in header)
                ):
                    raise ValueError("invalid header")
                count = 0
                for row in rows:
                    if len(row) != len(header):
                        raise ValueError("inconsistent row width")
                    count += 1
                if not count:
                    raise ValueError("no data rows")
                return header, count

        actual = shape(path)
        if samples and actual != shape(samples[0]):
            return False, "public sample header or row count mismatch"
        return (
            True,
            "public CSV structure valid; metric and task-specific validity not checked",
        )
    except (OSError, UnicodeError, csv.Error, StopIteration, ValueError) as error:
        return False, f"invalid CSV: {error}"


def publish(
    candidate: Path, run_root: Path, workspace: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Freeze a candidate confined to this run, before requesting hidden grading."""
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(
        run_root.resolve()
    ):
        raise ValueError("candidate escapes the current run")
    valid, message = validate_csv(candidate, workspace / "input")
    if not valid:
        raise ValueError(message)
    target = workspace / "solution" / "submission.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink():
        raise ValueError("solution directory must not be a symlink")
    before = digest(candidate)
    fd, temporary = tempfile.mkstemp(dir=target.parent)
    os.close(fd)
    try:
        shutil.copyfile(candidate, temporary)
        if digest(Path(temporary)) != before or digest(candidate) != before:
            raise ValueError("candidate changed while being frozen")
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    record = metadata | {
        "artifact_sha256": before,
        "candidate": str(candidate.relative_to(run_root)),
        "selection_policy": metadata.get("selection_policy", "upstream_single_final"),
        "schema_version": 1,
    }
    atomic_json(workspace / ".flowbench" / "paper-final.json", record)
    return record
