"""Stage generic evaluator code with a user-supplied task configuration."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

FIELDS = {
    "schema_version",
    "experiment_id",
    "suite",
    "slug",
    "higher_is_better",
    "feedback_mode",
    "submission_limit",
    "max_artifact_bytes",
    "score_timeout_seconds",
    "upstream_path",
    "dataset_path",
    "control_path",
}


def stage(config: Path, home: Path | None = None) -> Path:
    launch = json.loads(config.read_text())
    payload = launch.get("evaluator") if isinstance(launch, dict) else None
    if not isinstance(payload, dict) or set(payload) != FIELDS:
        raise ValueError("evaluator configuration fields do not match the schema")
    if payload["schema_version"] != 1 or payload["feedback_mode"] != "blind":
        raise ValueError("HMA requires schema version 1 and blind feedback")
    if payload["submission_limit"] is not None:
        raise ValueError("HMA enforces the option cap, not a cell-wide quota")
    if not str(payload["suite"]).startswith("mlebench_"):
        raise ValueError("select an MLE-bench suite, such as mlebench_full")
    if home is None:
        home = Path(launch["evaluator_seed_home"])
    if not home.is_absolute() or home.exists():
        raise ValueError("evaluator home must be a new absolute directory")
    target = home / "workspace/.flowbench"
    target.mkdir(parents=True)
    here = Path(__file__).parent
    shutil.copyfile(here / "evaluator_server.py", target / "mle-evaluator-server.py")
    shutil.copyfile(here / "score_worker.py", target / "mle-score-worker.py")
    (target / "mle-config.json").write_text(json.dumps(payload, indent=2) + "\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--home", type=Path, help="Override evaluator_seed_home from the launch configuration"
    )
    args = parser.parse_args()
    stage(args.config, args.home)
    print("Evaluator code staged; data and grader files remain externally supplied.")


if __name__ == "__main__":
    main()
