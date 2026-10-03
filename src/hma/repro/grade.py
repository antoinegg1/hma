"""Regrade retained candidates offline; preserve immutable acceptance receipts."""

from __future__ import annotations

import json
from pathlib import Path

import docker
from hma.analysis.events import ledger
from hma.benchmark.integrity import atomic_json, sha256_file


def grade(root: Path) -> dict:
    client = docker.from_env()
    environment = json.loads((root / "environment.json").read_text())
    from hma.benchmark.prepare_data import _verify_frozen_task
    from hma.repro.config import TASKS

    data_root = Path(environment["settings"]["data_root"])
    for name, expected in environment["images"].items():
        if client.images.get(name).id != expected:
            raise ValueError(
                "image changed since execution; restore the recorded image before regrading"
            )
    for task, digest in environment["data"].items():
        manifest = data_root / "manifests" / f"{task}.json"
        if sha256_file(manifest) != digest:
            raise ValueError("data manifest changed since execution")
        _verify_frozen_task(TASKS[task], data_root, json.loads(manifest.read_text()))
    count = 0
    for launch in sorted((root / "cells").glob("*/*/*/launch.json")):
        config = json.loads(launch.read_text())
        flowbench = launch.parent / "execution/evaluator/workspace/.flowbench"
        records = ledger(flowbench / "mle-submissions.jsonl")
        results = {}
        volumes = {str(flowbench.resolve()): {"bind": "/evaluation", "mode": "ro"}}
        volumes.update(
            {m["source"]: {"bind": m["target"], "mode": "ro"} for m in config["evaluator_data"]}
        )
        for record in records:
            digest = record["artifact_sha256"]
            if digest in results:
                continue
            if sha256_file(flowbench / "mle-candidates" / f"{digest}.csv") != digest:
                raise ValueError("retained candidate hash mismatch")
            raw = client.containers.run(
                config["evaluator_image"],
                entrypoint=[],
                command=[
                    "python3",
                    "/evaluation/mle-score-worker.py",
                    "--config",
                    "/evaluation/mle-config.json",
                    "--submission",
                    f"/evaluation/mle-candidates/{digest}.csv",
                ],
                volumes=volumes,
                network_disabled=True,
                remove=True,
                user=f"{config['uid']}:{config['gid']}",
            )
            results[digest] = json.loads(raw)
            count += 1
        atomic_json(launch.parent / "regraded.json", results)
    return {"candidates_regraded": count}
