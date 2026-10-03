"""One documented entry point for data, experiments, grading and reports."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from hma.benchmark.integrity import atomic_json
from hma.repro.config import ASSETS, TASKS, Suite, load_local, load_suite, plan, resolve_task


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "smoke", "doctor", "build"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, default=Path("configs/local.json"))
        if name in {"plan", "run", "smoke"}:
            sub.add_argument(
                "--suite",
                choices=["paper", "goal", "hma", "nta", "ablation", "harness16", "cases"],
                default="paper",
            )
            sub.add_argument("--experiments", type=Path)
            sub.add_argument("--experiment", action="append", default=[])
            sub.add_argument("--task", action="append", default=[])
            sub.add_argument("--repeat", action="append", type=int, default=[])
            sub.add_argument("--run-root", type=Path, default=Path("runs/paper"))
            if name == "run":
                sub.add_argument("--resume", action="store_true")
            if name == "smoke":
                sub.add_argument("--seconds", type=int, default=300)
            if name == "plan":
                sub.add_argument("--output", type=Path)
        if name == "build":
            sub.add_argument("--image", action="append", default=[])
        if name == "doctor":
            sub.add_argument(
                "--offline",
                action="store_true",
                help="check configuration without requiring Docker, GPU or credentials",
            )
    data = commands.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    for name in ("prepare", "verify"):
        sub = data_sub.add_parser(name)
        sub.add_argument("--config", type=Path, default=Path("configs/local.json"))
        sub.add_argument("--suite", choices=["paper", "harness16"], default="paper")
        sub.add_argument("--task", action="append", default=[])
        if name == "prepare":
            sub.add_argument("--prune-working-data", action="store_true")
    for name in ("status", "export", "grade", "report"):
        sub = commands.add_parser(name)
        sub.add_argument("--run-root", type=Path, required=True)
        if name in {"export", "report"}:
            sub.add_argument("--output", type=Path, default=Path("outputs/rerun"))
        if name == "report":
            sub.add_argument("--allow-partial", action="store_true")
    return root


def selected_plan(args: argparse.Namespace) -> dict:
    suite = load_suite(args.experiments)
    groups = args.experiment
    if not groups and args.suite != "paper":
        groups = [
            e.id
            for e in suite.experiments
            if (args.suite == e.workflow and not e.id.startswith(("cap-", "order-", "case-")))
            or (args.suite == "harness16" and e.workflow == "harness")
            or (args.suite == "ablation" and e.id.startswith(("cap-", "order-")))
            or (args.suite == "cases" and e.id == "case-glm-kimi")
        ]
    tasks, repeats = args.task, args.repeat
    if args.command == "smoke":
        groups = groups or ["goal-gpt"]
        tasks = tasks or ["mbl_09"]
        repeats = [0]
        definition = suite.model_dump(mode="json")
        for e in definition["experiments"]:
            if e["id"] in groups:
                if e["workflow"] == "harness":
                    if args.seconds <= 180:
                        raise ValueError("harness smoke needs more than 180 seconds")
                    e["research_seconds"] = args.seconds - 180
                    e["postprocess_seconds"] = 0
                e["seconds"] = args.seconds
        suite = Suite.model_validate(definition)
        if args.run_root == Path("runs/paper"):
            args.run_root = Path("runs/smoke")
    return plan(suite, groups, tasks, repeats)


def doctor(local, offline: bool) -> dict:
    from hma.repro.providers import credentials

    result = {
        "configuration": "valid",
        "experiments": len(load_suite().experiments),
        "tasks": len(TASKS),
        "mode": "offline" if offline else "live",
    }
    if offline:
        return result
    result["docker_cli"] = bool(shutil.which("docker"))
    result["nvidia_smi"] = bool(shutil.which("nvidia-smi"))
    result["uv"] = bool(shutil.which("uv"))
    result["nonroot"] = os.getuid() != 0
    try:
        import docker

        docker.from_env().ping()
        result["docker_daemon"] = True
    except Exception:
        result["docker_daemon"] = False
    from hma.repro.hardware import inspect_hardware

    try:
        result["hardware"] = inspect_hardware(local)
        result["gpu_model_matches"] = True
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        result["gpu_model_matches"] = False
        result["hardware_error"] = str(error)
    result["providers"] = {}
    for name in local.providers:
        try:
            credentials(local, {name})
            result["providers"][name] = "configured (not API-tested)"
        except ValueError:
            result["providers"][name] = "missing"
    result["ready"] = all(
        result[k]
        for k in ("docker_cli", "docker_daemon", "nvidia_smi", "gpu_model_matches", "uv", "nonroot")
    ) and all(v != "missing" for v in result["providers"].values())
    return result


def execute(args: argparse.Namespace) -> object:
    if args.command in {"status", "export", "grade", "report"}:
        root = args.run_root.resolve()
        if args.command == "status":
            from hma.repro.campaign import inventory

            return inventory(root)
        if args.command == "grade":
            from hma.repro.grade import grade

            return grade(root)
        if args.command == "report":
            from hma.analysis.report import report

            return report(root, args.output, args.allow_partial)
        from hma.analysis.events import normalize

        data = normalize(root)
        atomic_json(args.output / "events.json", data)
        return {"output": str(args.output), "issues": data["issues"]}
    if args.command == "plan":
        selected = selected_plan(args)
        if args.output:
            atomic_json(args.output, selected)
        counts = {}
        for cell in selected["cells"]:
            counts[cell["experiment"]] = counts.get(cell["experiment"], 0) + 1
        return {
            "cells": len(selected["cells"]),
            "max_gpu_hours": selected["max_gpu_hours"],
            "plan_sha256": selected["plan_sha256"],
            "experiments": counts,
        }
    local = load_local(args.config)
    if args.command == "doctor":
        return doctor(local, args.offline)
    if args.command == "build":
        from hma.repro.build import build

        build(local, args.image)
        return {"build": "complete"}
    if args.command == "data":
        from hma.benchmark import prepare_data

        prepare_data.BASE_RUNTIME_IMAGE = prepare_data.FULL_EVALUATOR_IMAGE = local.evaluator_image
        subset = (
            [r["task"] for r in json.loads((ASSETS / "tasks-16.json").read_text())["tasks"]]
            if args.suite == "harness16"
            else list(TASKS)
        )
        names = [resolve_task(t) for t in args.task] if args.task else subset
        if set(names) - set(subset):
            raise ValueError("requested task is outside selected suite")
        specs = [TASKS[name] for name in names]
        if args.data_command == "prepare":
            prepare_data.prepare_sources(local.data_root, specs)
            prepare_data.prepare_tasks(
                local.data_root, specs, prune_working_data=args.prune_working_data
            )
        else:
            for spec in specs:
                manifest = json.loads(
                    (local.data_root / "manifests" / f"{spec.task_name}.json").read_text()
                )
                prepare_data._verify_frozen_task(spec, local.data_root, manifest)
        return {"tasks": len(specs), "data": args.data_command}
    from hma.repro.campaign import run_plan

    return run_plan(selected_plan(args), local, args.run_root, getattr(args, "resume", False))


def main() -> int:
    args = parser().parse_args()
    try:
        result = execute(args)
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command in {"run", "smoke"} and any(r["status"] != "complete" for r in result):
        return 1
    if isinstance(result, dict) and result.get("ready") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
