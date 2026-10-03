"""Runs one pinned upstream preparation function in a disposable environment."""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from collections.abc import Sequence
from pathlib import Path


def _mlebench(upstream: Path, dataset: Path, slug: str) -> None:
    sys.path.insert(0, str(upstream))
    try:
        import py7zr  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("py7zr")
        module.SevenZipFile = object  # type: ignore[attr-defined]
        sys.modules["py7zr"] = module
    prepare_module = importlib.import_module(f"mlebench.competitions.{slug}.prepare")
    prepare_module.prepare(
        raw=dataset / "raw",
        public=dataset / "prepared" / "public",
        private=dataset / "prepared" / "private",
    )


def _mledojo(upstream: Path, dataset: Path, slug: str) -> None:
    sys.path.insert(0, str(upstream))
    from mledojo.competitions.registry import get_prepare

    prepare = get_prepare(slug)
    if prepare is None:
        raise ValueError(f"MLE-Dojo has no preparation function for {slug}")
    prepare(
        dataset / "raw",
        dataset / "data" / "public",
        dataset / "data" / "private",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one upstream data preparer")
    parser.add_argument(
        "--suite",
        choices=(
            "mlebench_lite",
            "mlebench_medium",
            "mlebench_high",
            "mledojo_unique",
        ),
    )
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--slug", required=True)
    args = parser.parse_args(argv)
    if args.suite.startswith("mlebench_"):
        _mlebench(args.upstream, args.dataset, args.slug)
    else:
        _mledojo(args.upstream, args.dataset, args.slug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
