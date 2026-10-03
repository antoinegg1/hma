"""Execute the frozen external adapter with a score-independent no-final rule."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from hma.benchmark.integrity import atomic_json, sha256_file


def latest_candidate(root: Path, deadline: float) -> Path | None:
    eligible = []
    for path in (root / "runs").rglob("submission.csv"):
        resolved = path.resolve()
        if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
            continue
        stat = resolved.stat()
        if stat.st_size and stat.st_mtime <= deadline:
            eligible.append((stat.st_mtime_ns, str(path.relative_to(root)), resolved))
    return max(eligible)[2] if eligible else None


def main() -> int:
    workspace = Path.cwd()
    contract = json.loads((workspace / ".flowbench/paper.json").read_text())
    route = json.loads(Path("/run/hma/route.json").read_text())
    process = subprocess.run(
        [sys.executable, "-m", "paper_runtime.runner", "--workspace", str(workspace)], check=False
    )
    if process.returncode == 0:
        return 0
    receipt = workspace / ".flowbench/paper-receipt.json"
    if receipt.exists() and json.loads(receipt.read_text()).get("status") == "accepted":
        return 0
    root = workspace / "paper-run"
    state_path = root / "status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if "research_started_at_utc" not in state:
        return process.returncode or 1  # Preflight/validator failure is infrastructure failure.
    if contract["missing_final_policy"] != "latest_submission_before_deadline":
        return 1
    selected = latest_candidate(root, route["deadline_epoch"])
    if selected is None or time.time() >= route["deadline_epoch"]:
        atomic_json(
            workspace / ".flowbench/no-submission.json",
            {"upstream_exit": process.returncode, "reason": "no_eligible_candidate"},
        )
        return 0  # A counted miss, not an infrastructure retry.
    output = workspace / "solution/submission.csv"
    output.parent.mkdir(exist_ok=True)
    if output.is_symlink():
        output.unlink()
    shutil.copyfile(selected, output)
    atomic_json(
        workspace / ".flowbench/fallback-selection.json",
        {
            "policy": contract["missing_final_policy"],
            "selected": str(selected.relative_to(root)),
            "artifact_sha256": sha256_file(output),
            "upstream_exit": process.returncode,
        },
    )
    result = subprocess.run(
        [sys.executable, str(workspace / ".flowbench/paper-submit.py"), "submit", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    # Invalid selected candidates remain misses; never search backwards for a higher score.
    atomic_json(
        workspace / ".flowbench/fallback-receipt.json",
        {"returncode": result.returncode, "response": result.stdout},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
