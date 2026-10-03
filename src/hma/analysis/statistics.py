"""Paper aggregation rules, kept separate from I/O and plotting."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, stdev


def mean_se(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return mean(values), stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None


def kl_lower(value: float, n: int, comparisons: int = 1100, delta: float = 0.05) -> float:
    if n <= 0 or value <= 0:
        return 0.0
    boundary = math.log(comparisons / delta) / n
    lo, hi = 0.0, value
    for _ in range(80):
        mid = (lo + hi) / 2
        divergence = value * math.log(value / mid)
        if value < 1:
            divergence += (1 - value) * math.log((1 - value) / (1 - mid))
        if divergence > boundary:
            lo = mid
        else:
            hi = mid
    return hi


def summaries(runs: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["experiment"]].append(run)
    rows = []
    for experiment, group in sorted(grouped.items()):
        repeats = sorted({r["repeat"] for r in group})
        row = {
            "experiment": experiment,
            "tasks_per_repeat": len(group) / len(repeats),
            "repeats": len(repeats),
            "complete": sum(r["status"] == "complete" for r in group),
            "planned": len(group),
        }
        for metric in ("medal", "gold", "med_plus", "quality", "lite", "medium", "high"):
            values = []
            for repeat in repeats:
                subset = [
                    r
                    for r in group
                    if r["repeat"] == repeat
                    and (metric not in {"lite", "medium", "high"} or r["split"] == metric)
                ]
                if not subset:
                    continue
                field = "medal" if metric in {"lite", "medium", "high"} else metric
                values.append(
                    100
                    * sum(
                        float((r["final"] or {}).get(field, 0)) if not r["excluded"] else 0
                        for r in subset
                    )
                    / len(subset)
                )
            row[metric], row[metric + "_se"] = mean_se(values)
        rows.append(row)
    return rows


def response_mean(
    events: list[dict], lo: float, hi: float, inclusive_end: bool = True
) -> float | None:
    chosen = (
        [e["tokens"] for e in events if lo < e["seconds"] <= hi]
        if inclusive_end
        else [e["tokens"] for e in events if lo <= e["seconds"] < hi]
    )
    return mean(chosen) if chosen else None


def prefix_retention(data: dict) -> list[dict]:
    """Average gains within task before averaging tasks and configurations."""
    selected = {
        e for e, config in data["experiments"].items() if config["workflow"] in {"goal", "nta"}
    }
    per_option = defaultdict(list)
    for sub in data["submissions"]:
        if sub["experiment"] in selected and not sub["review"] and sub["repeat"] == 0:
            per_option[sub["run"], sub["option"]].append(sub)
    options = defaultdict(list)
    for option in data["options"]:
        if option["experiment"] in selected and option["repeat"] == 0:
            options[option["run"]].append(option)
    raw = defaultdict(lambda: defaultdict(list))
    fractions = defaultdict(list)
    for run, opts in options.items():
        best = 0.0
        for option in sorted(opts, key=lambda o: o["option"]):
            scores = sorted(per_option[run, option["option"]], key=lambda s: s["sequence"])
            gains = [0.0]
            entry = best
            for sub in scores:
                best = max(best, sub["quality"])
                gains.append(best - entry)
            raw[option["experiment"]][option["task"]].append(gains)
            fractions[option["experiment"]].append(len(scores))
    result = []
    for experiment, tasks in raw.items():
        for k in range(1, 51):
            ratios = []
            for sequences in tasks.values():
                total = sum(s[-1] for s in sequences)
                if total > 0:
                    ratios.append(sum(s[min(k, len(s) - 1)] for s in sequences) / total)
            lengths = [n for n in fractions[experiment] if n > 0]
            retained = mean(ratios) if ratios else None
            ninety = mean(r >= 0.9 for r in ratios) if ratios else None
            result.append(
                {
                    "experiment": experiment,
                    "k": k,
                    "defined_tasks": len(ratios),
                    "gain_retention": retained,
                    "experiment_fraction": mean(min(k, n) / n for n in lengths)
                    if lengths
                    else None,
                    "fraction_above_90": ninety,
                    "gain_lower": kl_lower(retained or 0, len(ratios)),
                    "above90_lower": kl_lower(ninety or 0, len(ratios)),
                }
            )
    return result
