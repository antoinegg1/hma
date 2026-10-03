"""Run frozen experiment plans; resume schedules only never-started cells."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import docker
from hma.benchmark.integrity import atomic_json, sha256_file
from hma.benchmark.stage import stage
from hma.repro.config import ASSETS, TASKS, Experiment, Local, Suite, fingerprint
from hma.repro.hardware import inspect_hardware
from hma.repro.providers import credentials, names
from hma.supervisor import Config, Supervisor


def baseline_root() -> Path:
    installed = ASSETS / "harness16"
    return (
        installed
        if installed.is_dir()
        else Path(__file__).resolve().parents[3] / "baselines/harness16"
    )


def code_identity() -> str:
    source = Path(__file__).resolve().parents[2]
    files = {}
    for package in ("hma", "hmz"):
        for path in (source / package).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json"}:
                files[str(path.relative_to(source))] = sha256_file(path)
    for path in baseline_root().rglob("*"):
        if path.is_file() and path.suffix in {".py", ".json"}:
            files["baselines/" + str(path.relative_to(baseline_root()))] = sha256_file(path)
    return fingerprint(files)


def task_prompt(task: str, experiment: Experiment) -> str:
    spec = TASKS[task]
    mode = "single native goal" if experiment.workflow == "goal" else "current agent session"
    return f"""# {spec.experiment_id}: machine-learning optimization

Build the strongest legal, reproducible solution within the {experiment.seconds}-second
whole-run budget. Read the unmodified task description and public data under input/.
The metric is {spec.metric}; {"maximize" if spec.higher_is_better else "minimize"} it.
Start from the shared workspace. Continue this {mode} until naturally finished or stopped.
No task-specific prior solutions are supplied. Use your own leakage-safe local validation.

Write predictions to solution/submission.csv. Submit complete, meaningfully different
candidates as they become ready, including the final candidate before declaring completion:

    python mle_submit.py validate solution/submission.csv
    python mle_submit.py submit solution/submission.csv

Validation reports format acceptance only. Test scores, ranks and medal thresholds are
hidden. There is no medal-based stopping condition. The latest accepted candidate is
the standing result; do not assume an earlier best-scoring candidate is selected.

## Test-time information policy

Do not seek existing solutions, notebooks, discussions, write-ups, recovered labels, or answer files for this task.
Do not hand-label test data. Only public training labels and local validation may guide selection.
"""


def stage_cell(cell: dict, suite: Suite, local: Local, directory: Path, gpu: str) -> Path:
    experiment = next(e for e in suite.experiments if e.id == cell["experiment"])
    spec = TASKS[cell["task"]]
    data = local.data_root
    manifest = data / "manifests" / f"{spec.task_name}.json"
    if not manifest.is_file():
        raise ValueError(f"prepare and verify task first: {spec.task_name}")
    directory.mkdir(parents=True, exist_ok=False)
    task_file = directory / "task.md"
    task_file.write_text(task_prompt(cell["task"], experiment))
    actors = []
    for alias in experiment.actors:
        model = suite.models[alias]
        actors.append(
            {
                "spec": f"cli={model.cli},model={model.model},effort=max,permission=bypass,web_search=on",
                "provider": model.provider,
                "environment_names": names(model.provider),
            }
        )
    config = {
        "workflow": "goal" if experiment.workflow == "harness" else experiment.workflow,
        "root": str(directory / "execution"),
        "task_file": str(task_file),
        "agent_image": local.harness_images[experiment.harness]
        if experiment.harness
        else local.agent_image,
        "evaluator_image": local.evaluator_image,
        "evaluator_seed_home": str(directory / "evaluator-seed"),
        "actors": actors,
        "uid": os.getuid(),
        "gid": os.getgid(),
        "gpu": gpu,
        "cpus": local.cpus,
        "evaluator_cpus": local.evaluator_cpus,
        "evaluator_memory": local.evaluator_memory,
        "memory": local.memory,
        "shm_size": local.shm_size,
        "active_time_limit_seconds": experiment.seconds,
        "max_valid_submissions_per_session": experiment.cap,
        "review_reserve_seconds": experiment.review_reserve,
        "agent_data": [
            {
                "source": str(data / spec.public_subpath),
                "target": "/home/user/.flowbench-data/input",
            }
        ],
        "evaluator_data": [
            {
                "source": str(data / spec.data_subpath / "prepared"),
                "target": "/home/user/.flowbench-data/prepared",
            },
            {
                "source": str(data / "packages" / spec.task_name / "control"),
                "target": "/home/user/.flowbench-data/control",
            },
            {
                "source": str(data / "sources/mle-bench"),
                "target": "/home/user/.flowbench-data/upstream",
            },
        ],
        "evaluator": {
            "schema_version": 1,
            "experiment_id": cell["id"],
            "suite": spec.suite,
            "slug": spec.slug,
            "higher_is_better": spec.higher_is_better,
            "feedback_mode": "blind",
            "submission_limit": None,
            "max_artifact_bytes": 10737418240,
            "score_timeout_seconds": 3600,
            "upstream_path": "/home/user/.flowbench-data/upstream",
            "dataset_path": "/home/user/.flowbench-data/prepared",
            "control_path": "/home/user/.flowbench-data/control",
        },
    }
    if "kimi" in experiment.actors:
        config["kimi_proxy_module"] = str(Path(__file__).parents[1] / "adapters/provider_proxy.py")
    if experiment.workflow == "harness":
        seed = directory / "workspace-seed"
        flowbench = seed / ".flowbench"
        flowbench.mkdir(parents=True)
        lock = json.loads((baseline_root() / experiment.harness / "source.lock.json").read_text())
        model = suite.models[experiment.actors[0]].model
        contract = {
            "schema_version": 1,
            "require_mechanisms": True,
            "source": lock["source"],
            "source_commit": lock["commit"],
            "runtime_files": lock["runtime_files"],
            "slug": spec.slug,
            "task": spec.task_name,
            "profile": "ml_master_v2_no_prior"
            if experiment.harness == "ml_master_v2"
            else experiment.harness,
            "seed": cell["repeat"],
            "agent_seed": 42 + cell["repeat"],
            "budget_sec": experiment.seconds,
            "research_budget_sec": experiment.research_seconds
            or (43200 if experiment.harness == "mlevolve_no_prior" else 86400),
            "finalize_reserve_sec": 120,
            "startup_reserve_sec": 60,
            "postprocess_sec": experiment.postprocess_seconds
            if experiment.postprocess_seconds is not None
            else (900 if experiment.harness != "scienceflow" else 0),
            "code_model": model,
            "feedback_model": model,
            "temperature": 1.0,
            "cpu": int(local.cpus),
            "gpu": 1,
            "lower_is_better": not spec.higher_is_better,
            "candidate_policy": "unique" if experiment.harness == "ml_master_v2" else "last_final",
            "missing_final_policy": "latest_submission_before_deadline",
        }
        atomic_json(flowbench / "paper.json", contract)
        shutil.copyfile(task_file, flowbench / "task.md")
        shutil.copyfile(
            Path(__file__).parents[1] / "submit_client.py", flowbench / "paper-submit.py"
        )
        config.update(
            workspace_seed=str(seed), actor_command=["python3", "-m", "hma.repro.harness_entry"]
        )
    path = directory / "launch.json"
    atomic_json(path, config)
    stage(path)
    atomic_json(
        directory / "cell.json",
        cell
        | {
            "experiment_config": experiment.model_dump(mode="json"),
            "data_manifest_sha256": sha256_file(manifest),
            "task_prompt_sha256": sha256_file(task_file),
        },
    )
    return path


def inventory(root: Path) -> list[dict]:
    plan = json.loads((root / "plan.json").read_text())
    result = []
    for cell in plan["cells"]:
        directory = root / "cells" / cell["id"]
        status_path = directory / "execution/result.json"
        state = json.loads(status_path.read_text()) if status_path.exists() else {}
        result.append(
            cell
            | {"status": state.get("status", "interrupted" if directory.exists() else "pending")}
        )
    return result


def run_plan(plan: dict, local: Local, root: Path, resume: bool = False) -> list[dict]:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".campaign.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(plan, local, root, resume)


def _run_locked(plan: dict, local: Local, root: Path, resume: bool) -> list[dict]:
    suite = Suite.model_validate(plan["suite"])
    selected = {c["experiment"] for c in plan["cells"]}
    provider_ids = {
        suite.models[a].provider for e in suite.experiments if e.id in selected for a in e.actors
    }
    secret_env = credentials(local, provider_ids)
    if os.getuid() == 0:
        raise ValueError("run as a non-root Docker user")
    hardware = inspect_hardware(local)
    client = docker.from_env()
    client.ping()
    images = {local.evaluator_image}
    for e in suite.experiments:
        if e.id in selected:
            images.add(local.harness_images[e.harness] if e.harness else local.agent_image)
    image_ids = {name: client.images.get(name).id for name in sorted(images)}
    for task in {c["task"] for c in plan["cells"]}:
        from hma.benchmark.prepare_data import _verify_frozen_task

        manifest = json.loads((local.data_root / "manifests" / f"{task}.json").read_text())
        _verify_frozen_task(TASKS[task], local.data_root, manifest)
    settings = local.model_dump(mode="json", exclude={"providers"})
    code = code_identity()
    manifests = {
        task: sha256_file(local.data_root / "manifests" / f"{task}.json")
        for task in sorted({c["task"] for c in plan["cells"]})
    }
    identity = fingerprint(
        {
            "settings": settings,
            "images": image_ids,
            "code": code,
            "data": manifests,
            "hardware": hardware,
        }
    )
    if (root / "plan.json").exists():
        previous = json.loads((root / "plan.json").read_text())
        if not resume or previous["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError("existing campaign requires --resume and exactly the same plan")
        if json.loads((root / "environment.json").read_text())["identity"] != identity:
            raise ValueError("environment changed; use a new run-root")
    else:
        if resume:
            raise ValueError("cannot resume a campaign without plan.json")
        atomic_json(root / "plan.json", plan)
        atomic_json(
            root / "environment.json",
            {
                "identity": identity,
                "settings": settings,
                "hardware": hardware,
                "images": image_ids,
                "code": code,
                "data": manifests,
            },
        )
    old_env = {name: os.environ.get(name) for name in secret_env}
    os.environ.update(secret_env)
    devices: Queue[str] = Queue()
    for gpu in local.gpus:
        devices.put(gpu)

    def execute(cell: dict) -> None:
        directory = root / "cells" / cell["id"]
        if directory.exists():
            return  # Complete, failed and interrupted attempts are never silently retried.
        gpu = devices.get()
        try:
            config = stage_cell(cell, suite, local, directory, gpu)
            print(f"Starting {cell['id']} on GPU {gpu}", flush=True)
            Supervisor(Config.model_validate_json(config.read_text())).run()
        except Exception as error:
            directory.mkdir(parents=True, exist_ok=True)
            if not (directory / "execution/result.json").exists():
                atomic_json(
                    directory / "execution/result.json",
                    {"status": "failed", "error_type": type(error).__name__},
                )
            # Exception strings can contain provider responses. Keep credentials out of host logs.
            print(f"Failed {cell['id']}: {type(error).__name__}", flush=True)
        finally:
            devices.put(gpu)

    try:
        with ThreadPoolExecutor(max_workers=len(local.gpus)) as pool:
            list(pool.map(execute, plan["cells"]))
    finally:
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return inventory(root)
