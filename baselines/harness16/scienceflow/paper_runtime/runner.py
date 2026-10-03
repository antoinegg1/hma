"""Deadline supervisor: stop research, freeze one artifact, then request grading."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from .artifacts import atomic_json, digest, publish


def adopt_orphans() -> None:
    """Make this dedicated Linux supervisor own double-forked training children."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot enable child subreaper")


def stop_adopted(excluded_pid: int | None) -> None:
    """Drain all remaining descendants except the public validation service."""
    for _ in range(5):
        children = [
            child
            for child in psutil.Process().children(recursive=True)
            if child.pid != excluded_pid
        ]
        if not children:
            return
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(children, timeout=1)
    if any(p.pid != excluded_pid for p in psutil.Process().children(recursive=True)):
        raise RuntimeError("research descendants remain alive")


def redact_credentials(root: Path) -> None:
    """Remove provider keys from upstream config copies and textual run logs."""
    secrets = [
        value
        for key, value in os.environ.items()
        if key.startswith("PAPER_") and key.endswith("_API_KEY") and value
    ]
    if not secrets:
        return
    for path in root.rglob("*"):
        if (
            path.is_symlink()
            or not path.is_file()
            or path.suffix.lower()
            not in {
                ".yaml",
                ".yml",
                ".json",
                ".jsonl",
                ".log",
                ".txt",
                ".md",
            }
        ):
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            continue
        try:
            content = path.read_text()
        except (UnicodeError, OSError):
            continue
        redacted = content
        for secret in secrets:
            redacted = redacted.replace(secret, "[REDACTED]")
        if redacted != content:
            path.write_text(redacted)


def stop_tree(process: subprocess.Popen[Any]) -> None:
    """Stop descendants, including processes that opened their own sessions."""
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for child in descendants:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(descendants, timeout=3)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("research process did not stop") from error


def select_candidate(
    root: Path,
    source: str,
    candidate_pattern: str | None = None,
    candidate_policy: str = "unique",
) -> Path:
    """Inventory upstream finals; never infer an ordering from hidden scores."""
    pattern = {
        "scienceflow": "**/merge/finals/final_*/submission.csv",
        "mlevolve": "**/ensembles_csv/*.csv",
    }.get(source, "**/best_submission/submission.csv")
    candidates = sorted((root / "runs").glob(pattern))
    inventory = []
    for path in candidates:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root.resolve()):
            raise ValueError("upstream final points outside this run")
        inventory.append(
            {
                "path": str(path.relative_to(root / "runs")),
                "artifact_sha256": digest(resolved),
            }
        )
    atomic_json(root / "candidates.json", inventory)
    if candidate_policy == "last_final":
        if candidate_pattern is not None:
            raise ValueError("last_final cannot be combined with a candidate pattern")
        selected, evidence = last_final(root, source)
        if selected not in candidates:
            raise ValueError(
                "last recorded final is not an available upstream candidate"
            )
        # ScienceFlow publishes merge/finals/final_NN/submission.csv as a
        # symlink and digest() refuses a symlink at the leaf, so hashing the raw
        # manifest path here destroyed two complete 24h campaigns before line
        # 172 ever got to resolve it. Resolve first, with the same containment
        # check the later resolve applies.
        resolved_final = selected.resolve(strict=True)
        if not resolved_final.is_relative_to(root.resolve()):
            raise ValueError("upstream final points outside this run")
        atomic_json(
            root / "selection.json",
            evidence
            | {
                "path": str(selected.relative_to(root / "runs")),
                "resolved_path": str(resolved_final.relative_to(root)),
                "was_symlink": selected.is_symlink(),
                "artifact_sha256": digest(resolved_final),
            },
        )
        candidates = [selected]
    elif candidate_policy != "unique":
        raise ValueError("unknown final candidate policy")
    elif candidate_pattern is not None:
        permitted = set((root / "runs").glob(candidate_pattern))
        candidates = [path for path in candidates if path in permitted]
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one upstream final; found {len(candidates)}; "
            "freeze an explicit candidate_pattern before the run; see candidates.json"
        )
    selected = candidates[0].resolve(strict=True)
    if not selected.is_relative_to(root.resolve()):
        raise ValueError("upstream final points outside this run")
    return selected


def last_final(root: Path, source: str) -> tuple[Path, dict[str, Any]]:
    """Use the upstream final output sequence, independent of scores and mtimes."""
    if source == "scienceflow":
        manifests = list((root / "runs").glob("**/merge/global_merge_manifest.json"))
        if len(manifests) != 1:
            raise ValueError("last_final requires exactly one global merge manifest")
        manifest = json.loads(manifests[0].read_text())
        finals = manifest.get("finals", [])
        if not finals:
            raise ValueError("no final in the upstream merge manifest")
        record = finals[-1]
        if (
            record.get("artifact_exists") is not True
            or record.get("validation_ok") is False
        ):
            raise ValueError("last recorded final failed upstream validation")
        path = Path(record["final_dir"]) / record["artifact_path"]
        return path, {
            "order_source": str(manifests[0].relative_to(root)),
            "index": len(finals) - 1,
        }
    if source == "mlevolve":
        entries = []
        for line_number, line in enumerate(
            (root / "fusion.log").read_text().splitlines(), 1
        ):
            for prefix in ("Saved: ", "Copied top1 to: "):
                if line.startswith(prefix):
                    entries.append((line_number, Path(line[len(prefix) :])))
        if not entries:
            raise ValueError("no completed final recorded by upstream fusion")
        line_number, path = entries[-1]
        return path, {"order_source": "fusion.log", "line": line_number}
    raise ValueError("last_final is supported only for MLEvolve and ScienceFlow")


def fusion_command(root: Path, slug: str, executable: str) -> list[str]:
    """Invoke upstream fusion on this run, without its cross-run timestamp search."""
    runs = [p for p in (root / "runs").iterdir() if (p / "workspace").is_dir()]
    if len(runs) != 1:
        raise ValueError("fusion requires exactly one task run")
    return [
        executable,
        str(root / "upstream/utils/submission_fusion_utils.py"),
        "--task_id",
        slug,
        "--exp_name",
        runs[0].name,
        "--runs_root",
        str(root / "runs"),
        "--tag_path",
        str(root / "upstream/engine/coldstart/competition_tag_classified.json"),
    ]


def verify_source(root: Path, lock: dict[str, Any], contract: dict[str, Any]) -> None:
    """Require exactly the sanitized files from the locked upstream revision."""
    if (
        lock["commit"] != contract["source_commit"]
        or lock["source"] != contract["source"]
    ):
        raise ValueError("agent image contains a different upstream revision")
    for relative, expected in lock["files"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"upstream hash mismatch: {relative}")


def run(workspace: Path, image_root: Path = Path("/opt/paper")) -> int:
    """Execute once; interrupted runs require a new explicit experimental attempt."""
    workspace = workspace.resolve()
    adopt_orphans()
    contract = json.loads((workspace / ".flowbench/paper.json").read_text())
    root = workspace / "paper-run"
    if root.exists():
        raise ValueError("paper run already exists; implicit restart is prohibited")
    started = time.monotonic()
    budget = contract["budget_sec"]
    execution_seconds = budget - contract["finalize_reserve_sec"]
    postprocess_seconds = contract.get("postprocess_sec", 0)
    research_seconds = contract.get(
        "research_budget_sec", execution_seconds - postprocess_seconds
    )
    source_lock = json.loads((image_root / "source.lock.json").read_text())
    verify_source(image_root / "upstream", source_lock, contract)
    if source_lock["runtime_files"] != contract["runtime_files"]:
        raise ValueError("agent image adapter differs from the experiment lock")
    for name, expected in contract["runtime_files"].items():
        if digest(image_root / "paper_runtime" / name) != expected:
            raise ValueError(f"adapter hash mismatch: {name}")
    root.mkdir(mode=0o700)
    shutil.copytree(
        image_root / "upstream",
        root / "upstream",
        ignore=shutil.ignore_patterns(".venv", "__pycache__", "*.egg-info"),
    )
    env = os.environ.copy()
    env.update(
        PYTHONHASHSEED=str(contract["agent_seed"]),
        SEED=str(contract["agent_seed"]),
        TIME_LIMIT=f"{research_seconds} seconds",
        STEP_LIMIT="500",
        MEMORY_INDEX="0",
    )
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        port = bound.getsockname()[1]
    env["GRADING_SERVER_PORT"] = str(port)
    executable = (
        str(image_root / "upstream/.venv/bin/python")
        if contract["source"] == "scienceflow"
        else sys.executable
    )
    status: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "budget_sec": budget,
        "research_budget_sec": research_seconds,
        "postprocess_budget_sec": postprocess_seconds,
    }
    atomic_json(root / "status.json", status)
    child: subprocess.Popen[Any] | None = None
    validator: subprocess.Popen[Any] | None = None
    try:
        with (root / "validator.log").open("w") as log:
            validator = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "paper_runtime.validator",
                    "--public",
                    str(workspace / "input"),
                    "--slug",
                    contract["slug"],
                    "--port",
                    str(port),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        for _ in range(100):
            if validator.poll() is not None:
                raise RuntimeError("public validator failed to start")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=0.2
                ):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("public validator health timeout")
        startup_elapsed = time.monotonic() - started
        if "research_budget_sec" in contract and startup_elapsed > contract.get(
            "startup_reserve_sec", 60
        ):
            raise RuntimeError("startup exceeded its separate reserve")
        remaining = (
            research_seconds
            if "research_budget_sec" in contract
            else int(research_seconds - startup_elapsed)
        )
        if remaining < 1:
            raise RuntimeError("budget exhausted during startup")
        with (root / "upstream.log").open("w") as log:
            research_started = time.monotonic()
            status["research_started_at_utc"] = datetime.now(UTC).isoformat()
            atomic_json(root / "status.json", status)
            child = subprocess.Popen(
                [
                    executable,
                    "-m",
                    "paper_runtime.upstream",
                    "--workspace",
                    str(workspace),
                    "--root",
                    str(root / "upstream"),
                    "--seconds",
                    str(remaining),
                    "--port",
                    str(port),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            try:
                # ML-Master 2's own watchdog starts task-level wisdom promotion.
                # Allow it the postprocessing window before enforcing the hard cap.
                wait_seconds = remaining + (
                    postprocess_seconds if contract["source"] == "ml_master_v2" else 0
                )
                returncode = child.wait(timeout=wait_seconds)
                status["upstream_exit_code"] = returncode
                status["termination"] = (
                    "natural" if returncode == 0 else "upstream_error"
                )
            except subprocess.TimeoutExpired:
                status["termination"] = "research_deadline"
            finally:
                stop_tree(child)
                stop_adopted(validator.pid)
                status["research_elapsed_sec"] = time.monotonic() - research_started
        if (
            contract["source"] in {"ml_master_v2", "scienceflow"}
            and status["termination"] != "natural"
        ):
            raise RuntimeError(
                "upstream did not complete its research/finalization lifecycle"
            )
        if contract["source"] == "mlevolve":
            status["postprocess"] = "running_upstream_submission_fusion"
            atomic_json(root / "status.json", status)
            with (root / "fusion.log").open("w") as log:
                child = subprocess.Popen(
                    fusion_command(root, contract["slug"], executable),
                    cwd=root / "upstream",
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                try:
                    available = min(
                        postprocess_seconds,
                        execution_seconds - (time.monotonic() - started),
                    )
                    if available <= 0 or child.wait(timeout=available) != 0:
                        raise RuntimeError(
                            "upstream submission fusion failed or missed its budget"
                        )
                finally:
                    stop_tree(child)
                    stop_adopted(validator.pid)
            status["postprocess"] = "complete"
        candidate = select_candidate(
            root,
            contract["source"],
            contract.get("candidate_pattern"),
            contract.get("candidate_policy", "unique"),
        )
        elapsed = time.monotonic() - started
        if elapsed >= budget:
            raise RuntimeError("final artifact missed deadline")
        final = publish(
            candidate,
            root,
            workspace,
            {
                "profile": contract["profile"],
                "seed": contract["seed"],
                "source_commit": contract["source_commit"],
                "selected_at_utc": datetime.now(UTC).isoformat(),
                "selected_elapsed_sec": elapsed,
                "budget_sec": budget,
                "termination": status["termination"],
                "selection_policy": "last_upstream_final"
                if contract.get("candidate_policy") == "last_final"
                else "frozen_explicit_candidate"
                if contract.get("candidate_pattern")
                else "upstream_single_final",
                "candidate_pattern": contract.get("candidate_pattern"),
            },
        )
        # No research process is alive when the first hidden-grader request is made.
        submission = subprocess.run(
            [
                sys.executable,
                str(workspace / ".flowbench/paper-submit.py"),
                "submit",
                str(workspace / "solution/submission.csv"),
            ],
            capture_output=True,
            text=True,
            timeout=max(1, budget - (time.monotonic() - started)),
            check=False,
        )
        if submission.returncode:
            atomic_json(
                workspace / ".flowbench/paper-receipt.json",
                {
                    "status": "failed",
                    "returncode": submission.returncode,
                    "stdout": submission.stdout,
                    "stderr": submission.stderr,
                },
            )
            raise RuntimeError("final grading failed; see paper-receipt.json")
        receipt = json.loads(submission.stdout)
        if receipt.get("artifact_sha256") != final["artifact_sha256"]:
            raise ValueError(
                "final grading receipt does not match the selected artifact"
            )
        atomic_json(workspace / ".flowbench/paper-receipt.json", receipt)
        status["status"] = "complete"
        return 0
    except BaseException as error:
        status.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        if child is not None:
            stop_tree(child)
        if validator is not None:
            stop_tree(validator)
        stop_adopted(None)
        redact_credentials(root)
        status["elapsed_sec"] = time.monotonic() - started
        atomic_json(root / "status.json", status)


def main() -> None:
    """Entrypoint inside an agent image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()

    def interrupted(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(run(args.workspace))


if __name__ == "__main__":
    main()
