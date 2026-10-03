"""Normalize new controller, evaluator and native CLI records."""

from __future__ import annotations

import hashlib
import importlib
import json
from datetime import datetime
from pathlib import Path

from hma.benchmark.integrity import sha256_file
from hma.repro.config import TASKS


def read_json(path: Path, default: object = None) -> object:
    return json.loads(path.read_text()) if path.exists() else default


def ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    previous = "0" * 64
    for index, row in enumerate(rows, 1):
        unsigned = {k: v for k, v in row.items() if k != "record_hash"}
        digest = hashlib.sha256(
            json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            row["sequence"] != index
            or row["previous_record_hash"] != previous
            or row["record_hash"] != digest
        ):
            raise ValueError(f"invalid evaluator ledger chain: {path}:{index}")
        previous = digest
    return rows


def native_responses(
    home: Path, cli: str, model: str, start: float, end: float
) -> tuple[list[dict], list[dict]]:
    reader = importlib.import_module(f"hmz.tracing.readers.{cli}")
    locations = {
        "claude": [".claude"],
        "codex": [".codex"],
        "kimi": [".kimi", ".kimi-code", ".config/kimi"],
        "dsh": [".dsh"],
    }
    from hma.analysis.usage import response_usage

    events, sessions = [], []
    seen = set()
    for relative in locations[cli]:
        for session in reader.collect(home / relative, None, None, (start, end)):
            if session.parent or session.label not in ("main", "", "agent", "/root"):
                continue
            path = Path(session.args["log"])
            decoded, issues = response_usage(path, cli, start, end)
            bounds = {"session": session.key, "issues": issues}
            if session.actions:
                bounds.update(
                    start=min(a.start for a in session.actions),
                    end=max(a.end for a in session.actions),
                )
            sessions.append(bounds)
            for event in decoded:
                identity = (session.key, event["response_id"])
                if identity in seen:
                    continue
                seen.add(identity)
                events.append(
                    event
                    | {
                        "response_id": session.key + ":" + event["response_id"],
                        "session": session.key,
                        "model": model,
                    }
                )
    return sorted(events, key=lambda r: r["epoch"]), sessions


def normalize(root: Path) -> dict:
    plan = read_json(root / "plan.json")
    if not isinstance(plan, dict):
        raise ValueError("run-root has no plan.json")
    experiments = {e["id"]: e for e in plan["suite"]["experiments"]}
    models = plan["suite"]["models"]
    exclusions = read_json(root / "exclusions.json", {})
    if not isinstance(exclusions, dict) or set(exclusions) - {c["id"] for c in plan["cells"]}:
        raise ValueError("exclusions must map planned cell IDs to reason and artifact_sha256")
    runs, responses, options, submissions, reviews = [], [], [], [], []
    issues = []
    for cell in plan["cells"]:
        directory = root / "cells" / cell["id"] / "execution"
        experiment = experiments[cell["experiment"]]
        result = read_json(directory / "result.json", {})
        deadline = read_json(directory / "deadline.json", {})
        started = deadline.get("started_epoch", 0)
        ended = deadline.get("deadline_epoch", started + cell["seconds"])
        turns = read_json(directory / "turn-history.json", [])
        review = read_json(directory / "review.json", {})
        log = directory / "evaluator/workspace/.flowbench/mle-submissions.jsonl"
        records = ledger(log)
        regraded = read_json(directory.parent / "regraded.json", {})
        context = {
            "run": cell["id"],
            "experiment": cell["experiment"],
            "task": cell["task"],
            "repeat": cell["repeat"],
        }
        baseline = 0
        run_options = []
        for turn in turns:
            alias = experiment["actors"][turn["actor"]]
            option = context | {
                "option": turn["turn"],
                "model": alias,
                "reason": turn["reason"],
                "accepted": turn["accepted"],
                "start": max(0, turn.get("started_epoch", started) - started),
                "end": min(cell["seconds"], turn.get("ended_epoch", ended) - started),
                "first_sequence": baseline + 1,
                "last_sequence": baseline + turn["accepted"],
            }
            run_options.append(option)
            baseline += turn["accepted"]
        options.extend(run_options)
        exploration = []
        for record in records:
            digest = record["artifact_sha256"]
            artifact = log.parent / "mle-candidates" / f"{digest}.csv"
            if not artifact.exists() or sha256_file(artifact) != digest:
                raise ValueError(f"missing or changed accepted artifact in {cell['id']}: {digest}")
            score = regraded.get(digest, record["result"])
            board = score.get("leaderboard", {})
            option = next(
                (
                    o
                    for o in run_options
                    if o["first_sequence"] <= record["sequence"] <= o["last_sequence"]
                ),
                None,
            )
            row = context | {
                "submission_id": record["submission_id"],
                "artifact_sha256": digest,
                "sequence": record["sequence"],
                "seconds": datetime.fromisoformat(record["accepted_at_utc"]).timestamp() - started,
                "score": score.get("raw_score"),
                "medal": bool(board.get("any_medal")),
                "gold": bool(board.get("gold_medal")),
                "med_plus": bool(board.get("above_median")),
                "quality": 1 - (board["position"] - 1) / board["total"]
                if board.get("total")
                else 0,
                "option": option["option"] if option else None,
                "review": option is None,
                "gold_threshold": board.get("gold_threshold"),
                "bronze_threshold": board.get("bronze_threshold"),
                "model": option["model"] if option else None,
            }
            submissions.append(row)
            if option is not None:
                exploration.append(row)
        final = next((s for s in reversed(submissions) if s["run"] == cell["id"]), None)
        excluded = exclusions.get(cell["id"])
        if excluded:
            if (
                not excluded.get("reason")
                or not final
                or excluded.get("artifact_sha256") != final["artifact_sha256"]
            ):
                raise ValueError("exclusion needs a reason and the final candidate digest")
        state = result.get("status", "interrupted" if directory.parent.exists() else "pending")
        row = context | {
            "status": state,
            "workflow": experiment["workflow"],
            "seconds": cell["seconds"],
            "split": TASKS[cell["task"]].suite.removeprefix("mlebench_"),
            "excluded": excluded or None,
            "final": final,
            "before": exploration[-1] if exploration else None,
            "historical_medal": any(s["medal"] for s in exploration),
        }
        runs.append(row)
        reviews.append(context | review)
        if state != "complete":
            issues.append(f"{cell['id']}: {state}")
        if experiment["workflow"] == "harness" or not started:
            continue
        run_response_count = 0
        for option in run_options + (
            [{"option": "review", "model": experiment["actors"][review["reviewer_actor"]]}]
            if "reviewer_actor" in review
            else []
        ):
            native = models[option["model"]]
            turn_name = "review" if option["option"] == "review" else f"{option['option']:05d}"
            home = directory / "turns" / turn_name / "agent"
            events, sessions = native_responses(
                home, native["cli"], option["model"], started, ended
            )
            run_response_count += len(events)
            for session in sessions:
                issues.extend(f"{cell['id']}/{turn_name}: {issue}" for issue in session["issues"])
            if not events and (
                turn_name != "review"
                or review.get("reason") not in {"no_candidates", "review_disabled"}
            ):
                issues.append(f"{cell['id']}/{turn_name}: no main-agent response usage decoded")
            sessions = [s for s in sessions if "start" in s]
            for event in events:
                responses.append(
                    context
                    | event
                    | {
                        "option": option["option"],
                        "seconds": event["epoch"] - started,
                        "review": turn_name == "review",
                    }
                )
            if isinstance(option["option"], int) and sessions:
                # Runtime attribution uses native starts. Unfinished sessions extend to
                # the recorded controller boundary; cap/deadline intervals include tools.
                option["native_start"] = max(
                    option["start"], min(s["start"] for s in sessions) - started
                )
                option["native_end"] = (
                    option["end"]
                    if option["reason"] != "natural_exit"
                    else min(option["end"], max(s["end"] for s in sessions) - started)
                )
        if state == "complete" and not run_response_count:
            issues.append(f"{cell['id']}: no main-agent response usage decoded")
    return {
        "schema_version": 1,
        "plan_sha256": plan["plan_sha256"],
        "experiments": experiments,
        "runs": runs,
        "responses": responses,
        "options": options,
        "submissions": submissions,
        "reviews": reviews,
        "issues": issues,
    }
