"Trusted subprocess that calls the graders pinned in an upstream checkout."

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import statistics
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("score config must be an object")
    return payload


def _board_report(score: float, path: Path, *, higher_is_better: bool) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows or "score" not in rows[0]:
        raise ValueError("official leaderboard has no scores")
    scores = [float(row["score"]) for row in rows]
    expected = sorted(scores, reverse=higher_is_better)
    if scores != expected:
        raise ValueError("official leaderboard conflicts with metric direction")
    position = next(
        (
            index
            for index, value in enumerate(scores, start=1)
            if (score >= value if higher_is_better else score <= value)
        ),
        len(scores) + 1,
    )
    total = len(scores)
    if total < 100:
        medal_positions = (
            max(1, int(total * 0.1)),
            max(1, int(total * 0.2)),
            max(1, int(total * 0.4)),
        )
    elif total < 250:
        medal_positions = (10, max(1, int(total * 0.2)), max(1, int(total * 0.4)))
    elif total < 1_000:
        medal_positions = (10 + int(total * 0.002), 50, 100)
    else:
        medal_positions = (
            10 + int(total * 0.002),
            max(1, int(total * 0.05)),
            max(1, int(total * 0.1)),
        )
    gold_threshold, silver_threshold, bronze_threshold = (
        scores[index - 1] for index in medal_positions
    )
    beats = (lambda value: score >= value) if higher_is_better else (lambda value: score <= value)
    gold = beats(gold_threshold)
    silver = not gold and beats(silver_threshold)
    bronze = not gold and not silver and beats(bronze_threshold)
    median_threshold = float(statistics.median(scores))
    return {
        "position": position,
        "total": total,
        "is_lower_better": not higher_is_better,
        "gold_threshold": gold_threshold,
        "silver_threshold": silver_threshold,
        "bronze_threshold": bronze_threshold,
        "median_threshold": median_threshold,
        "gold_medal": gold,
        "silver_medal": silver,
        "bronze_medal": bronze,
        "any_medal": gold or silver or bronze,
        "above_median": beats(median_threshold) and score != median_threshold,
    }


def _configure_mlebench_pandas(pd: Any) -> None:
    "Restores the frame behaviour used by the pinned MLE-bench environment."
    try:
        pd.set_option("future.infer_string", False)
    except (AttributeError, KeyError):
        pass
    if not hasattr(pd.DataFrame, "applymap"):
        pd.DataFrame.applymap = pd.DataFrame.map


def _mlebench_score(config: dict[str, Any], submission_path: Path) -> dict[str, Any]:
    import pandas as pd
    import yaml

    _configure_mlebench_pandas(pd)
    upstream = Path(str(config["upstream_path"]))
    dataset = Path(str(config["dataset_path"]))
    slug = str(config["slug"])
    sys.path.insert(0, str(upstream))
    py7zr = types.ModuleType("py7zr")
    py7zr.SevenZipFile = object
    sys.modules.setdefault("py7zr", py7zr)
    competition_dir = upstream / "mlebench" / "competitions" / slug
    task_config = yaml.safe_load((competition_dir / "config.yaml").read_text(encoding="utf-8"))
    module_name, function_name = task_config["grader"]["grade_fn"].split(":", 1)
    grade = getattr(importlib.import_module(module_name), function_name)
    answer_parts = Path(task_config["dataset"]["answers"]).parts[1:]
    submission = pd.read_csv(submission_path)
    answers_path = dataset.joinpath(*answer_parts)
    if answers_path.suffix == ".jsonl":
        from mlebench.utils import read_jsonl

        answers: Any = read_jsonl(str(answers_path))
    else:
        answers = pd.read_csv(answers_path)
    score = round(float(grade(submission, answers)), 5)
    if not math.isfinite(score):
        raise ValueError("official grader returned a non-finite score")
    report = _board_report(
        score,
        Path(str(config["control_path"])) / "leaderboard.csv",
        higher_is_better=bool(config["higher_is_better"]),
    )
    return {"valid": True, "raw_score": score, "leaderboard": report}


def _dojo_position(score: float, dataset: Path, *, higher_is_better: bool) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    position_scores: list[float] = []
    for name in ("private", "public"):
        path = dataset / "data" / "private" / f"{name}_leaderboard.csv"
        with path.open(newline="", encoding="utf-8-sig") as file:
            rows = list(csv.reader(file))
        if len(rows) < 2 or len(rows[0]) < 2:
            raise ValueError("MLE-Dojo leaderboard has no scores")
        scores = sorted((float(row[1]) for row in rows[1:]), reverse=higher_is_better)
        position = next(
            (
                index
                for index, value in enumerate(scores, start=1)
                if (score >= value if higher_is_better else score <= value)
            ),
            len(scores) + 1,
        )
        position_score = (len(scores) - position + 1) / len(scores)
        reports[name] = {
            "position": position,
            "total": len(scores),
            "position_score": position_score,
        }
        position_scores.append(position_score)
    reports["avg_score"] = sum(position_scores) / len(position_scores)
    return reports


def _mledojo_score(config: dict[str, Any], submission_path: Path) -> dict[str, Any]:
    import pandas as pd

    upstream = Path(str(config["upstream_path"]))
    dataset = Path(str(config["dataset_path"]))
    slug = str(config["slug"])
    sys.path.insert(0, str(upstream))
    from mledojo.competitions.registry import get_metric

    metric_class = get_metric(slug)
    if metric_class is None:
        raise ValueError("official metric is unavailable")
    metric = metric_class()
    submission = pd.read_csv(submission_path)
    answers = pd.read_csv(dataset / "data" / "private" / "test_answer.csv")
    metric.validate_submission(submission, answers)
    score = float(metric.evaluate(answers, submission))
    if not math.isfinite(score):
        raise ValueError("official metric returned a non-finite score")
    report = _dojo_position(
        score,
        dataset,
        higher_is_better=bool(metric.higher_is_better),
    )
    return {"valid": True, "raw_score": score, "position": report}


def _basic_csv_check(path: Path) -> None:
    with path.open(newline="", encoding="utf-8-sig") as file:
        header = next(csv.reader(file))
    if header and not header[0].strip():
        header = header[1:]
    if not header or any(not value.strip() for value in header):
        raise ValueError("CSV header is empty")
    if len(header) != len(set(header)):
        raise ValueError("CSV header contains duplicate columns")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score one protected MLE submission")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--submission", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        config = _load_config(args.config)
        _basic_csv_check(args.submission)
        if str(config["suite"]).startswith("mlebench_"):
            result = _mlebench_score(config, args.submission)
        else:
            result = _mledojo_score(config, args.submission)
        score = result["raw_score"]
        if isinstance(score, bool) or not math.isfinite(float(score)):
            raise ValueError("grader returned a non-finite score")
    except Exception as error:
        result = {
            "valid": False,
            "error": str(error)[:2_000],
            "error_type": type(error).__name__,
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    sys.exit(main())
