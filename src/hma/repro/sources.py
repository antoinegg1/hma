"""Fetch exact public upstream revisions and verify baseline source inventories."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from hma.benchmark.integrity import sha256_file
from hma.benchmark.prepare_data import _clone_pinned
from hma.repro.config import ASSETS


def checkout(root: Path, name: str) -> Path:
    sources = json.loads((ASSETS / "sources.lock.json").read_text())["sources"]
    source = sources[name]
    target = root / name
    _clone_pinned(target, source["url"], source["commit"])
    if subprocess.check_output(
        ["git", "-C", str(target), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError(f"source checkout is modified: {target}")
    return target


def verify_baseline(source: Path, lock: Path, runtime: Path) -> None:
    spec = json.loads(lock.read_text())
    for directory, files in [(source, spec["files"]), (runtime, spec["runtime_files"])]:
        for name, expected in files.items():
            if sha256_file(directory / name) != expected:
                raise ValueError(f"source hash mismatch: {directory / name}")
