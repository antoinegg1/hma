"""Produce auditable tables and figures entirely from a new campaign."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from contextvars import ContextVar
from pathlib import Path
from statistics import mean, median

from hma.analysis.events import normalize
from hma.analysis.statistics import prefix_retention, response_mean, summaries
from hma.benchmark.integrity import atomic_json
from hma.repro.config import ASSETS

_REPORT_LABEL = ContextVar("report_label", default="New-run report")

MAIN = [
    "hma-opus-gpt",
    "hma-gpt-opus",
    "hma-gpt-ds41",
    "hma-gpt-kimi",
    "hma-gpt-glm",
    "hma-ds41-kimi",
]
GOAL_WEIGHTS = {
    "goal-gpt": 5 / 12,
    "goal-opus": 1 / 6,
    "goal-ds41": 1 / 6,
    "goal-kimi": 1 / 6,
    "goal-glm": 1 / 12,
}
CASES = [
    ("fig05", "hma-gpt-opus", "mbh_07"),
    ("fig07", "hma-gpt-ds41", "mbm_10"),
    ("fig08", "case-glm-kimi", "mbm_27"),
    ("fig09", "hma-ds41-kimi", "mbh_12"),
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                for k, v in row.items()
            }
            for row in rows
        )


def plot_lines(path: Path, series: dict[str, tuple[list, list]], xlabel: str, ylabel: str) -> None:
    series = {
        label: values for label, values in series.items() if any(v is not None for v in values[1])
    }
    if not series:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, (xs, ys) in series.items():
        ax.plot(xs, [float("nan") if y is None else y for y in ys], label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    fig.suptitle(_REPORT_LABEL.get(), fontsize=9)
    fig.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(path.with_suffix("." + ext), dpi=160)
    plt.close(fig)


def trajectory(data: dict) -> list[dict]:
    runs = defaultdict(list)
    responses = defaultdict(list)
    submissions = defaultdict(list)
    for run in data["runs"]:
        if run["workflow"] != "harness":
            runs[run["experiment"]].append(run)
    for event in data["responses"]:
        responses[event["experiment"]].append(event)
    for sub in data["submissions"]:
        if not sub["review"]:
            submissions[sub["run"]].append(sub)
    for rows in submissions.values():
        rows.sort(key=lambda s: s["seconds"])
    result = []
    for experiment, group in runs.items():
        for seconds in range(0, min(r["seconds"] for r in group) + 1, 60):
            medals = 0
            for run in group:
                eligible = [s for s in submissions[run["run"]] if s["seconds"] <= seconds]
                last = eligible[-1] if eligible else None
                if seconds == run["seconds"]:
                    last = run["final"]
                medals += bool(last and last["medal"] and not run["excluded"])
            result.append(
                {
                    "experiment": experiment,
                    "seconds": seconds,
                    "medal_rate": 100 * medals / len(group),
                    "tokens_per_response": response_mean(
                        responses[experiment], max(0, seconds - 1800), seconds
                    ),
                }
            )
    return result


def option_review_tables(data: dict) -> tuple[list, list, list]:
    closure, retention, participation = [], [], []
    for experiment in sorted({run["experiment"] for run in data["runs"]}):
        group = [r for r in data["runs"] if r["experiment"] == experiment and r["repeat"] == 0]
        opts = [o for o in data["options"] if o["experiment"] == experiment and o["repeat"] == 0]
        if experiment in MAIN:
            counts = Counter(
                ("initial" if o["option"] == 0 else "later", o["reason"]) for o in opts
            )
            natural = [o for o in opts if o["reason"] == "natural_exit"]
            row = {
                "experiment": experiment,
                "options": len(opts),
                "natural_zero_accepted": sum(o["accepted"] == 0 for o in natural),
            }
            for which in ("initial", "later"):
                for reason in ("submission_cap", "natural_exit", "deadline", "actor_error"):
                    row[which + "_" + reason] = counts[which, reason]
                observed = counts[which, "natural_exit"] + counts[which, "submission_cap"]
                row[which + "_natural_rate"] = (
                    counts[which, "natural_exit"] / observed if observed else None
                )
            closure.append(row)
            historical = sum(r["historical_medal"] for r in group)
            before = sum(bool(r["before"] and r["before"]["medal"]) for r in group)
            after = sum(
                bool(r["final"] and r["final"]["medal"] and not r["excluded"]) for r in group
            )
            retention.append(
                {
                    "experiment": experiment,
                    "tasks": len(group),
                    "historical": historical,
                    "changed": sum(
                        bool(
                            r["final"]
                            and r["before"]
                            and r["final"]["artifact_sha256"] != r["before"]["artifact_sha256"]
                        )
                        for r in group
                    ),
                    "before": before,
                    "after": after,
                    "missed_after": historical - after,
                    "retention_before": before / historical if historical else None,
                    "retention_after": after / historical if historical else None,
                    "gained": sum(
                        bool(
                            r["final"]
                            and r["final"]["medal"]
                            and not (r["before"] and r["before"]["medal"])
                        )
                        for r in group
                    ),
                    "lost": sum(
                        bool(
                            r["before"]
                            and r["before"]["medal"]
                            and not (r["final"] and r["final"]["medal"])
                        )
                        for r in group
                    ),
                }
            )
        if experiment in {"nta-gpt-kimi", "nta-kimi-gpt", "hma-gpt-kimi", "order-kimi-gpt"}:
            first, partner = data["experiments"][experiment]["actors"]
            events = [
                e
                for e in data["responses"]
                if e["experiment"] == experiment and e["repeat"] == 0 and not e["review"]
            ]
            arrival = []
            token_shares = []
            for run in group:
                partner_events = [
                    e["seconds"] for e in events if e["run"] == run["run"] and e["model"] == partner
                ]
                arrival.append(min(partner_events) if partner_events else run["seconds"])
                early = [e for e in events if e["run"] == run["run"] and e["seconds"] <= 20700]
                total = sum(e["tokens"] for e in early)
                if total:
                    token_shares.append(
                        sum(e["tokens"] for e in early if e["model"] == partner) / total
                    )
            row = {
                "experiment": experiment,
                "tasks": len(group),
                "final_medals": sum(
                    bool(r["final"] and r["final"]["medal"] and not r["excluded"]) for r in group
                ),
                "median_first_response_h_censored": median(arrival) / 3600 if arrival else None,
                "partner_output_share": mean(token_shares) if token_shares else None,
            }
            for cutoff in (10800, 18000, 20700):
                row[f"by_{cutoff}s"] = sum(t <= cutoff for t in arrival)
            for model in (first, partner):
                seconds = sum(
                    max(0, o.get("native_end", o["end"]) - o.get("native_start", o["start"]))
                    for o in opts
                    if o["model"] == model
                )
                for review in data["reviews"]:
                    if (
                        review["experiment"] == experiment
                        and review["repeat"] == 0
                        and "reviewer_actor" in review
                        and data["experiments"][experiment]["actors"][review["reviewer_actor"]]
                        == model
                    ):
                        seconds += review.get("lived_seconds", 0)
                row[f"{model}_time_share"] = (
                    seconds / sum(r["seconds"] for r in group) if group else None
                )
            participation.append(row)
    return closure, retention, participation


def case_plot(
    output: Path,
    name: str,
    subs: list[dict],
    events: list[dict],
    opts: list[dict],
    references: list[dict],
) -> None:
    if not subs:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (score_ax, token_ax) = plt.subplots(2, 1, figsize=(9, 6.5))
    seq = list(range(1, len(subs) + 1))
    score_ax.plot(seq, [s["score"] for s in subs], "o-", ms=3)
    if name in {"fig05", "fig08"} and all(s["score"] > 0 for s in subs):
        if name == "fig05":
            score_ax.set_yscale("log")
        inset = score_ax.inset_axes([0.53, 0.48, 0.43, 0.45])
        tail = max(0, len(subs) // 2)
        inset.plot(seq[tail:], [s["score"] for s in subs[tail:]], "o-", ms=2)
    if references:
        lower = name != "fig07"
        best = (min if lower else max)(r["final"]["score"] for r in references)
        score_ax.axhline(best, linestyle="--", label="Best constituent goal endpoint")
    threshold_kind = "bronze" if name == "fig09" else "gold"
    threshold = subs[0].get(threshold_kind + "_threshold")
    if threshold is not None:
        score_ax.axhline(threshold, linestyle=":", label=threshold_kind.capitalize() + " threshold")
    score_ax.set(ylabel="Test score (offline)", xlabel="Accepted experiment")
    palette = ["#d9e7ff", "#e8d9fa"]
    windows = []
    for index, option in enumerate(opts):
        owned = [i + 1 for i, s in enumerate(subs) if s["option"] == option["option"]]
        if owned:
            score_ax.axvspan(
                min(owned) - 0.5, max(owned) + 0.5, alpha=0.4, color=palette[index % 2]
            )
            score_ax.axvline(min(owned) - 0.5, linestyle=":", color="grey", linewidth=0.7)
        replies = sorted(
            [e for e in events if e["option"] == option["option"]], key=lambda e: e["seconds"]
        )
        windows.append(
            {
                "option": option["option"],
                "model": option["model"],
                "responses": len(replies),
                "first20": mean(e["tokens"] for e in replies[:20]) if len(replies) >= 40 else None,
                "last20": mean(e["tokens"] for e in replies[-20:]) if len(replies) >= 40 else None,
            }
        )
    if name == "fig05":
        means = []
        previous = {}
        for sub in subs:
            option = next(o for o in opts if o["option"] == sub["option"])
            lo = previous.get(sub["option"], option["start"])
            means.append(response_mean(events, lo, sub["seconds"]))
            previous[sub["option"]] = sub["seconds"]
        token_ax.plot(seq, [float("nan") if v is None else v for v in means], "o-")
        write_csv(
            output / (name + "-experiment-responses.csv"),
            [{"experiment": n, "mean_output_tokens": v} for n, v in zip(seq, means)],
        )
    else:
        for key in ("first20", "last20"):
            token_ax.plot(
                [w["option"] for w in windows],
                [w[key] if w[key] is not None else float("nan") for w in windows],
                "o-",
                label=key,
            )
        for w in windows:
            if w["responses"] < 40:
                token_ax.annotate(f"n={w['responses']}", (w["option"], 0))
        token_ax.legend()
    write_csv(output / (name + "-windows.csv"), windows)
    token_ax.set(
        ylabel="Output tokens / response", xlabel="Experiment" if name == "fig05" else "Option"
    )
    if references or threshold is not None:
        score_ax.legend(fontsize=8)
    fig.suptitle(_REPORT_LABEL.get(), fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(output / f"{name}.{ext}", dpi=160)
    plt.close(fig)


def report(root: Path, output: Path, partial: bool = False) -> dict:
    data = normalize(root)
    output.mkdir(parents=True, exist_ok=True)
    from hma.repro.config import load_suite, plan

    full_paper = data["plan_sha256"] == plan(load_suite(), [], [], [])["plan_sha256"]
    coverage = {
        "issues": data["issues"],
        "planned": len(data["runs"]),
        "partial": bool(data["issues"]),
        "full_paper_plan": full_paper,
        "by_experiment": {
            e: {
                "planned": sum(r["experiment"] == e for r in data["runs"]),
                "complete": sum(
                    r["experiment"] == e and r["status"] == "complete" for r in data["runs"]
                ),
            }
            for e in sorted({r["experiment"] for r in data["runs"]})
        },
    }
    atomic_json(output / "coverage.json", coverage)
    _REPORT_LABEL.set(
        ("PARTIAL — " if data["issues"] else "")
        + ("Full paper plan" if full_paper else "Selected/smoke plan")
        + f"; {len(data['runs'])} planned runs"
    )
    if data["issues"] and not partial:
        raise ValueError(
            "incomplete runs or missing response records; see coverage.json, or use --allow-partial"
        )
    atomic_json(output / "events.json", data)
    for name in ("runs", "submissions", "responses", "options", "reviews"):
        write_csv(output / f"{name}.csv", data[name])
    table2 = summaries(data["runs"])
    lookup = {r["experiment"]: r for r in table2}
    for row in table2:
        experiment = data["experiments"][row["experiment"]]
        if row["experiment"] in MAIN:
            a, b = [lookup.get("goal-" + m) for m in experiment["actors"]]
            if a and b:
                for metric in ("medal", "gold", "med_plus", "quality", "lite", "medium", "high"):
                    if a[metric] is not None and b[metric] is not None:
                        row[metric + "_single_mean"] = (a[metric] + b[metric]) / 2
                        row[metric + "_gain"] = (
                            row[metric] - row[metric + "_single_mean"]
                            if row[metric] is not None
                            else None
                        )
                        row[metric + "_single_se"] = (
                            ((a[metric + "_se"] ** 2 + b[metric + "_se"] ** 2) ** 0.5 / 2)
                            if a[metric + "_se"] is not None and b[metric + "_se"] is not None
                            else None
                        )
    write_csv(output / "table02-main.csv", table2)
    subset = {t["task"] for t in json.loads((ASSETS / "tasks-16.json").read_text())["tasks"]}
    harness = [
        r
        for r in data["runs"]
        if r["task"] in subset
        and (r["workflow"] == "harness" or r["experiment"] in {"goal-ds4", "goal-ds41"})
    ]
    table1 = summaries(harness)
    native = {r["experiment"]: r for r in table1 if r["experiment"].startswith("goal-")}
    for row in table1:
        alias = data["experiments"][row["experiment"]]["actors"][0]
        reference = native.get("goal-" + alias)
        row["gain_vs_native_pp"] = row["medal"] - reference["medal"] if reference else None
    write_csv(output / "table01-harness.csv", table1)
    task_rows = []
    for experiment in sorted({r["experiment"] for r in harness}):
        for task in sorted(subset):
            selected = [r for r in harness if r["experiment"] == experiment and r["task"] == task]
            if selected:
                task_rows.append(
                    {
                        "experiment": experiment,
                        "task": task,
                        "medals": sum(
                            bool(r["final"] and r["final"]["medal"] and not r["excluded"])
                            for r in selected
                        ),
                        "runs": len(selected),
                        "medal_rate": 100
                        * sum(
                            bool(r["final"] and r["final"]["medal"] and not r["excluded"])
                            for r in selected
                        )
                        / len(selected),
                        "gold": sum(
                            bool(r["final"] and r["final"]["gold"] and not r["excluded"])
                            for r in selected
                        ),
                    }
                )
    write_csv(output / "table07-harness-tasks.csv", task_rows)
    write_csv(
        output / "table03-configurations.csv",
        [e for e in data["experiments"].values() if e["workflow"] in {"goal", "nta"}],
    )
    plan_data = json.loads((root / "plan.json").read_text())
    write_csv(
        output / "table04-models.csv",
        [
            {"alias": alias, "effort": "max", **model}
            for alias, model in plan_data["suite"]["models"].items()
        ],
    )
    write_csv(
        output / "table05-published-configurations.csv",
        json.loads((ASSETS / "published-baselines.json").read_text()),
    )
    comparisons = []
    for experiment in MAIN:
        actors = data["experiments"].get(experiment, {}).get("actors", [])
        for task in sorted({r["task"] for r in data["runs"] if r["experiment"] == experiment}):
            ours = [r for r in data["runs"] if r["experiment"] == experiment and r["task"] == task]
            single = [
                [r for r in data["runs"] if r["experiment"] == "goal-" + a and r["task"] == task]
                for a in actors
            ]
            if len(single) != 2 or not all(single):
                continue
            for metric in ("medal", "quality"):

                def value(rows):
                    return 100 * mean(
                        float((r["final"] or {}).get(metric, 0)) if not r["excluded"] else 0
                        for r in rows
                    )

                reference = mean(value(rows) for rows in single)
                comparisons.append(
                    {
                        "experiment": experiment,
                        "task": task,
                        "metric": metric,
                        "hma": value(ours),
                        "constituent_mean": reference,
                        "gain_pp": value(ours) - reference,
                    }
                )
    write_csv(output / "task-comparisons.csv", comparisons)
    early_late = []
    for experiment in sorted({e["experiment"] for e in data["responses"]}):
        events = [e for e in data["responses"] if e["experiment"] == experiment]
        early, late = (
            response_mean(events, 0, 7200, False),
            response_mean(events, 10800, 18000, False),
        )
        early_late.append(
            {
                "experiment": experiment,
                "early": early,
                "late": late,
                "retention": late / early if early and late is not None else None,
                "change_pct": 100 * (late / early - 1) if early and late is not None else None,
            }
        )
    write_csv(output / "table06-early-late.csv", early_late)
    early_lookup = {r["experiment"]: r for r in early_late}
    weighted = []
    for label, weights in [
        ("HMA mean", dict.fromkeys(MAIN, 1 / 6)),
        ("Matched goal", GOAL_WEIGHTS),
    ]:
        if all(
            e in early_lookup
            and early_lookup[e]["early"] is not None
            and early_lookup[e]["late"] is not None
            for e in weights
        ):
            early = sum(early_lookup[e]["early"] * w for e, w in weights.items())
            late = sum(early_lookup[e]["late"] * w for e, w in weights.items())
            weighted.append(
                {
                    "group": label,
                    "early": early,
                    "late": late,
                    "retention": late / early if early else None,
                }
            )
    write_csv(output / "matched-early-late.csv", weighted)
    curves = trajectory(data)
    write_csv(output / "trajectories.csv", curves)
    groups = defaultdict(list)
    for row in curves:
        groups[row["experiment"]].append(row)
    plot_lines(
        output / "fig02",
        {
            e: ([r["seconds"] / 3600 for r in rows], [r["tokens_per_response"] for r in rows])
            for e, rows in groups.items()
            if e.startswith("goal-")
        },
        "Active runtime (h)",
        "Output tokens / response",
    )
    prefix = prefix_retention(data)
    write_csv(output / "prefix-retention.csv", prefix)
    prefix_series = {}
    for experiment in sorted({r["experiment"] for r in prefix}):
        rows = [
            r
            for r in prefix
            if r["experiment"] == experiment
            and r["gain_retention"] is not None
            and r["experiment_fraction"] is not None
        ]
        if rows:
            prefix_series[experiment] = (
                [0] + [100 * r["experiment_fraction"] for r in rows],
                [0] + [100 * r["gain_retention"] for r in rows],
            )
    if prefix_series:
        xs = [mean(values[0][k] for values in prefix_series.values()) for k in range(51)]
        ys = [mean(values[1][k] for values in prefix_series.values()) for k in range(51)]
        prefix_series["Configuration mean"] = (xs, ys)
    plot_lines(
        output / "fig03",
        prefix_series,
        "Within-option experiments retained (%)",
        "Observed gain retained (%)",
    )
    matched = []
    for panel, metric in [("fig04a", "tokens_per_response"), ("fig04b", "medal_rate")]:
        series = {
            e: ([r["seconds"] / 3600 for r in groups[e]], [r[metric] for r in groups[e]])
            for e in MAIN
            if e in groups
        }
        time_sets = [
            {r["seconds"] for r in groups[e]} for e in MAIN + list(GOAL_WEIGHTS) if e in groups
        ]
        common = sorted(set.intersection(*time_sets)) if time_sets else []
        indexes = {e: {r["seconds"]: r[metric] for r in rows} for e, rows in groups.items()}
        if all(e in indexes for e in MAIN + list(GOAL_WEIGHTS)):
            ys, refs = [], []
            for second in common:
                values = [indexes[e].get(second) for e in MAIN]
                goal = [indexes[e].get(second) for e in GOAL_WEIGHTS]
                ours = mean(values) if all(v is not None for v in values) else None
                ref = (
                    sum(indexes[e][second] * w for e, w in GOAL_WEIGHTS.items())
                    if all(v is not None for v in goal)
                    else None
                )
                ys.append(ours)
                refs.append(ref)
                matched.append(
                    {"metric": metric, "seconds": second, "hma": ours, "matched_goal": ref}
                )
            series["HMA mean"] = ([t / 3600 for t in common], ys)
            series["Matched goal"] = ([t / 3600 for t in common], refs)
        plot_lines(output / panel, series, "Active runtime (h)", metric)
    write_csv(output / "matched-curves.csv", matched)
    closure, retention, participation = option_review_tables(data)
    write_csv(output / "table10-option-closures.csv", closure)
    write_csv(output / "table11-review.csv", retention)
    write_csv(output / "table08-participation.csv", participation)
    ablations = summaries(
        [
            r
            for r in data["runs"]
            if r["repeat"] == 0
            and r["experiment"]
            in {"nta-gpt-kimi", "nta-kimi-gpt", "hma-gpt-kimi", "order-kimi-gpt", "cap-1", "cap-3"}
        ]
    )
    write_csv(output / "table09-scheduling.csv", ablations)
    abl = {r["experiment"]: r["medal"] for r in ablations}
    plot_lines(
        output / "fig06a",
        {
            "NTA": ([0, 1], [abl.get("nta-gpt-kimi"), abl.get("nta-kimi-gpt")]),
            "HMA": ([0, 1], [abl.get("hma-gpt-kimi"), abl.get("order-kimi-gpt")]),
        },
        "Starting order (0=GPT, 1=Kimi)",
        "Medal rate (%)",
    )
    plot_lines(
        output / "fig06b",
        {
            "GPT-first HMA": (
                [1, 3, 5],
                [abl.get("cap-1"), abl.get("cap-3"), abl.get("hma-gpt-kimi")],
            )
        },
        "Cap",
        "Medal rate (%)",
    )
    for name, experiment, task in CASES:
        subs = [
            s
            for s in data["submissions"]
            if s["experiment"] == experiment
            and s["task"] == task
            and s["repeat"] == 0
            and not s["review"]
        ]
        events = [
            e
            for e in data["responses"]
            if e["experiment"] == experiment
            and e["task"] == task
            and e["repeat"] == 0
            and not e["review"]
        ]
        opts = [
            o
            for o in data["options"]
            if o["experiment"] == experiment and o["task"] == task and o["repeat"] == 0
        ]
        actors = data["experiments"].get(experiment, {}).get("actors", [])
        refs = [
            r
            for r in data["runs"]
            if r["task"] == task
            and r["experiment"] in ["goal-" + a for a in actors]
            and r["final"]
            and not r["excluded"]
        ]
        write_csv(output / f"{name}-scores.csv", subs)
        case_plot(output, name, subs, events, opts, refs)
    (output / "README.txt").write_text(
        "New-run report. No historical paper scores are used.\nSee coverage.json for partial/missing runs.\nFailed/no-submission tasks remain in planned denominators. Single-repeat SE is undefined.\nExternal published baseline rows are not rerun here.\n"
    )
    return {"output": str(output), "runs": len(data["runs"]), "issues": data["issues"]}
