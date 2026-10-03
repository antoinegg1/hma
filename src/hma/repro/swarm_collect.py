"""Collect finished partitions of one frozen cluster without copying their evidence."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from hma.benchmark.integrity import atomic_json
from hma.repro.config import Local, Suite, experiment_tasks, fingerprint
from hma.repro.hardware import check_gpus


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular collection input: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"collection input must be an object: {path}")
    return value


def _partition(plan: dict, task: str) -> dict:
    cells = [cell for cell in plan["cells"] if cell["task"] == task]
    return {
        "schema_version": 1,
        "suite": plan["suite"],
        "cells": cells,
        "plan_sha256": fingerprint({"suite": plan["suite"], "cells": cells}),
        "max_gpu_hours": sum(cell["seconds"] for cell in cells) / 3600,
    }


def _manifest(cluster: dict) -> tuple[dict, dict, Local, str]:
    if cluster.get("schema_version") != 1:
        raise ValueError("unsupported cluster schema")
    identity = fingerprint(
        {key: value for key, value in cluster.items() if key != "cluster_sha256"}
    )
    if cluster.get("cluster_sha256", identity) != identity:
        raise ValueError("cluster manifest hash is invalid")
    plan = cluster["plan"]
    if plan.get("schema_version") != 1 or not plan.get("cells"):
        raise ValueError("cluster requires a nonempty version 1 plan")
    if plan.get("plan_sha256") != fingerprint({"suite": plan["suite"], "cells": plan["cells"]}):
        raise ValueError("global plan hash is invalid")
    suite = Suite.model_validate(plan["suite"])
    experiments = {experiment.id: experiment for experiment in suite.experiments}
    seen = set()
    for cell in plan["cells"]:
        experiment = experiments.get(cell["experiment"])
        repeat = cell["repeat"]
        if (
            experiment is None
            or cell["task"] not in experiment_tasks(experiment)
            or type(repeat) is not int
            or not 0 <= repeat < experiment.repeats
            or cell["seconds"] != experiment.seconds
            or cell["id"] != f"{experiment.id}/{cell['task']}/r{repeat}"
            or cell["id"] in seen
        ):
            raise ValueError("global plan contains an invalid or duplicate cell")
        seen.add(cell["id"])
    if plan.get("max_gpu_hours") != sum(cell["seconds"] for cell in plan["cells"]) / 3600:
        raise ValueError("global plan GPU-hour total is invalid")
    assignments = cluster["assignments"]
    if not isinstance(assignments, list) or any(
        not isinstance(row, dict)
        or set(row) != {"node_id", "node_hostname", "task"}
        or any(not isinstance(value, str) or not value for value in row.values())
        for row in assignments
    ):
        raise ValueError("invalid cluster assignments")
    tasks = {cell["task"] for cell in plan["cells"]}
    if (
        len(assignments) != len(tasks)
        or {row["task"] for row in assignments} != tasks
        or len({row["node_id"] for row in assignments}) != len(assignments)
    ):
        raise ValueError("assignments must cover every task exactly once with unique node IDs")
    if "providers" in cluster["settings"]:
        raise ValueError("cluster settings must not contain provider credentials")
    local = Local.model_validate(cluster["settings"] | {"providers": {}})
    if local.model_dump(mode="json", exclude={"providers"}) != cluster["settings"]:
        raise ValueError("cluster settings must contain the complete frozen local configuration")
    if not isinstance(cluster["code"], str) or not cluster["code"]:
        raise ValueError("cluster requires a frozen code identity")
    return plan, {row["task"]: row for row in assignments}, local, identity


def _environment(environment: dict, cluster: dict, local: Local, shard_plan: dict) -> None:
    task = shard_plan["cells"][0]["task"]
    if environment.get("settings") != cluster["settings"]:
        raise ValueError(f"shard resources/settings differ from the cluster: {task}")
    if environment.get("code") != cluster["code"]:
        raise ValueError(f"shard code differs from the cluster: {task}")
    data = environment.get("data")
    if not isinstance(data, dict) or set(data) != {task} or not data[task]:
        raise ValueError(f"shard data manifests must cover only its assigned task: {task}")
    hardware = environment.get("hardware")
    if not isinstance(hardware, dict) or not hardware.get("gpus"):
        raise ValueError(f"shard must retain its actual hardware: {task}")
    check_gpus(hardware["gpus"], local)
    selected = {cell["experiment"] for cell in shard_plan["cells"]}
    expected_images = {local.evaluator_image}
    for experiment in shard_plan["suite"]["experiments"]:
        if experiment["id"] in selected:
            expected_images.add(
                local.harness_images[experiment["harness"]]
                if experiment["harness"]
                else local.agent_image
            )
    images = environment.get("images")
    if (
        not isinstance(images, dict)
        or set(images) != expected_images
        or any(not isinstance(value, str) or not value for value in images.values())
    ):
        raise ValueError(f"shard image IDs do not cover its execution environments: {task}")
    identity = fingerprint(
        {key: environment[key] for key in ("settings", "images", "code", "data", "hardware")}
    )
    if environment.get("identity") != identity:
        raise ValueError(f"shard environment hash is invalid: {task}")


def collect(cluster_root: Path, output: Path, allow_partial: bool = False) -> dict:
    """Validate all finished shards, then publish an atomic symlink-based report root.

    Every shard must be initialized and match the frozen cluster. By default all
    cells must have terminal outcomes; explicit partial mode preserves missing or
    interrupted cells for diagnosis. Failures remain failures for analysis. Source
    shards must remain in place: this is a view, not a portable evidence archive.
    """
    cluster_root = cluster_root.expanduser().resolve()
    raw_output = output.expanduser().absolute()
    if os.path.lexists(raw_output):
        raise ValueError(f"collection output already exists: {raw_output}")
    output = raw_output.resolve()
    cluster = _read(cluster_root / "cluster.json")
    plan, assignments, local, identity = _manifest(cluster)
    if "root" in cluster and Path(cluster["root"]) != cluster_root:
        raise ValueError("cluster must retain its frozen absolute source root")
    shard_roots = {task: cluster_root / "shards" / task for task in assignments}
    if any(output.is_relative_to(root.resolve()) for root in shard_roots.values()):
        raise ValueError("collection output must be outside source shards")
    sources, environments, images, data, exclusions = {}, {}, {}, {}, {}
    statuses: Counter = Counter()
    with ExitStack() as stack:
        # Acquire every lock before reading any result, and keep all locks through
        # publication. Opening existing locks read-only leaves source shards intact.
        for task, root in sorted(shard_roots.items()):
            lock_path = root / ".campaign.lock"
            if root.is_symlink() or not root.is_dir() or lock_path.is_symlink():
                raise ValueError(f"missing regular shard directory or campaign lock: {task}")
            try:
                lock = stack.enter_context(lock_path.open("r"))
                # NFS requires a writable descriptor for an exclusive lock.
                # A shared reader lock still excludes each campaign's writer lock.
                fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except FileNotFoundError as error:
                raise ValueError(f"missing shard campaign lock: {task}") from error
            except BlockingIOError as error:
                raise ValueError(f"shard is still running or locked: {task}") from error
        for task, root in sorted(shard_roots.items()):
            assignment = assignments[task] | {"cluster_sha256": identity}
            if _read(root / "cluster-assignment.json") != assignment:
                raise ValueError(f"shard belongs to a different frozen cluster assignment: {task}")
            expected = _partition(plan, task)
            if _read(root / "plan.json") != expected:
                raise ValueError(f"shard plan differs from its exact cluster partition: {task}")
            environment = _read(root / "environment.json")
            _environment(environment, cluster, local, expected)
            for image, image_id in environment["images"].items():
                if image in images and images[image] != image_id:
                    raise ValueError(f"image ID differs between shards: {image}")
                images[image] = image_id
            data.update(environment["data"])
            environments[task] = {
                **assignments[task],
                "root": str(root),
                "environment": environment,
            }
            for cell in expected["cells"]:
                source = root / "cells" / cell["id"]
                if source.is_symlink() or source.resolve() != source:
                    raise ValueError(f"missing regular shard cell directory: {cell['id']}")
                if not source.exists():
                    if not allow_partial:
                        raise ValueError(f"missing regular shard cell directory: {cell['id']}")
                    statuses["pending"] += 1
                    continue
                if not source.is_dir():
                    raise ValueError(f"missing regular shard cell directory: {cell['id']}")
                result_path = source / "execution/result.json"
                status = (
                    _read(result_path).get("status")
                    if os.path.lexists(result_path)
                    else "interrupted"
                )
                if status not in {"complete", "failed", "pending", "interrupted", "cleanup_failed"}:
                    raise ValueError(f"invalid shard cell outcome: {cell['id']}")
                if not allow_partial and status not in {"complete", "failed"}:
                    raise ValueError(f"shard cell has no terminal outcome: {cell['id']}")
                if os.path.lexists(source / "cell.json"):
                    record = _read(source / "cell.json")
                    if (
                        any(record.get(key) != value for key, value in cell.items())
                        or record.get("data_manifest_sha256") != data[task]
                    ):
                        raise ValueError(
                            f"shard cell evidence differs from its plan/data: {cell['id']}"
                        )
                sources[cell["id"]] = source
                statuses[status] += 1
            if (root / "exclusions.json").exists():
                shard_exclusions = _read(root / "exclusions.json")
                if set(shard_exclusions) - {cell["id"] for cell in expected["cells"]}:
                    raise ValueError(f"shard exclusions contain unassigned cells: {task}")
                # Artifact binding is checked by the normal report/export path.
                exclusions.update(shard_exclusions)
        aggregate = {
            "schema_version": 1,
            "kind": "swarm-collection",
            "partial": allow_partial,
            "statuses": dict(statuses),
            "cluster_sha256": identity,
            "source_root": str(cluster_root),
            "settings": cluster["settings"],
            "code": cluster["code"],
            "images": images,
            "data": data,
            "shards": environments,
        }
        aggregate["identity"] = fingerprint(aggregate)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
        try:
            atomic_json(temporary / "plan.json", plan)
            atomic_json(temporary / "cluster.json", cluster)
            atomic_json(temporary / "environment.json", aggregate)
            if exclusions:
                atomic_json(temporary / "exclusions.json", exclusions)
            for cell_id, source in sources.items():
                target = temporary / "cells" / cell_id
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source, target_is_directory=True)
            if os.path.lexists(output):
                raise ValueError(f"collection output already exists: {output}")
            temporary.rename(output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return {
        "output": str(output),
        "cluster_sha256": identity,
        "tasks": len(assignments),
        "cells": len(plan["cells"]),
        "linked_cells": len(sources),
        "partial": allow_partial,
        "statuses": dict(statuses),
    }
