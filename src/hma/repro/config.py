"""Validated, credential-free experiment plans for fresh runs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hma.benchmark.catalog import MLEBENCH_FULL

ASSETS = Path(__file__).parent / "assets"
TASKS = {t.task_name: t for t in MLEBENCH_FULL}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Provider(Strict):
    api_key: str = ""
    base_url: str = ""


class Local(Strict):
    schema_version: Literal[1] = 1
    data_root: Path = Path("data")
    agent_image: str = "hma-agent:local"
    evaluator_image: str = "hma-evaluator:local"
    gpus: list[str] = Field(default_factory=lambda: ["0"])
    expected_gpu_model: str | None = Field(default="NVIDIA A10", min_length=1)
    cpus: float = Field(default=30, gt=0)
    memory: str = "220g"
    shm_size: str = "65536m"
    evaluator_cpus: float = Field(default=2, gt=0)
    evaluator_memory: str = "16384m"
    providers: dict[str, Provider]
    harness_images: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_devices(self) -> Local:
        if not self.gpus or len(self.gpus) != len(set(self.gpus)):
            raise ValueError("gpus must contain unique device IDs")
        self.data_root = self.data_root.expanduser().resolve()
        return self


class Model(Strict):
    cli: Literal["codex", "claude", "kimi", "dsh"]
    model: str
    provider: str


class Experiment(Strict):
    id: str
    workflow: Literal["goal", "hma", "nta", "harness"]
    actors: list[str]
    repeats: int = Field(ge=1)
    cap: int | None = Field(default=None, ge=1)
    tasks: str | list[str] = "all"
    seconds: int = Field(default=21600, gt=0)
    review_reserve: int = Field(default=0, ge=0)
    harness: str | None = None
    research_seconds: int | None = Field(default=None, gt=0)
    postprocess_seconds: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def protocol(self) -> Experiment:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.id):
            raise ValueError("unsafe experiment ID")
        expected = 2 if self.workflow in {"hma", "nta"} else 1
        if len(self.actors) != expected:
            raise ValueError("wrong number of actors")
        if self.workflow == "hma":
            if self.cap is None or not 695 < self.review_reserve < self.seconds:
                raise ValueError("HMA needs a cap and a review reserve exceeding 695s")
        elif self.cap is not None or self.review_reserve:
            raise ValueError("goal/NTA/harness must have null cap and no review")
        if (self.workflow == "harness") != (self.harness is not None):
            raise ValueError("harness workflow requires a harness name")
        if self.workflow == "harness":
            research = self.research_seconds or (
                43200 if self.harness == "mlevolve_no_prior" else 86400
            )
            postprocess = (
                self.postprocess_seconds
                if self.postprocess_seconds is not None
                else (0 if self.harness == "scienceflow" else 900)
            )
            if research + postprocess + 180 > self.seconds:
                raise ValueError(
                    "harness budget must fit research, postprocess and 180s startup/finalization"
                )
        elif self.research_seconds is not None or self.postprocess_seconds is not None:
            raise ValueError("research/postprocess budgets apply only to harnesses")
        return self


class Suite(Strict):
    schema_version: Literal[1] = 1
    models: dict[str, Model]
    experiments: list[Experiment]

    @model_validator(mode="after")
    def references(self) -> Suite:
        ids = [e.id for e in self.experiments]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate experiment IDs")
        for e in self.experiments:
            if set(e.actors) - self.models.keys():
                raise ValueError(f"unknown model in {e.id}")
            experiment_tasks(e)
        return self


def resolve_task(name: str) -> str:
    for task in MLEBENCH_FULL:
        if name in (task.task_name, task.slug, task.experiment_id):
            return task.task_name
    raise ValueError(f"unknown task: {name}")


def experiment_tasks(experiment: Experiment) -> list[str]:
    if experiment.tasks == "all":
        return list(TASKS)
    if experiment.tasks == "harness16":
        return [t["task"] for t in json.loads((ASSETS / "tasks-16.json").read_text())["tasks"]]
    if isinstance(experiment.tasks, list) and experiment.tasks:
        tasks = [resolve_task(t) for t in experiment.tasks]
        if len(tasks) != len(set(tasks)):
            raise ValueError("duplicate tasks")
        return tasks
    raise ValueError("tasks must be all, harness16, or a nonempty task list")


def load_suite(path: Path | None = None) -> Suite:
    return Suite.model_validate_json((path or ASSETS / "experiments.json").read_text())


def load_local(path: Path) -> Local:
    return Local.model_validate_json(path.read_text())


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def plan(suite: Suite, groups: list[str], tasks: list[str], repeats: list[int]) -> dict:
    known = {e.id for e in suite.experiments}
    if set(groups) - known:
        raise ValueError(f"unknown experiments: {set(groups) - known}")
    requested = {resolve_task(t) for t in tasks}
    cells = []
    covered = set()
    for e in suite.experiments:
        if groups and e.id not in groups:
            continue
        if repeats and any(r < 0 or r >= e.repeats for r in repeats):
            raise ValueError(f"repeat outside [0,{e.repeats}) for {e.id}")
        for task in experiment_tasks(e):
            if requested and task not in requested:
                continue
            covered.add(task)
            for repeat in range(e.repeats):
                if repeats and repeat not in repeats:
                    continue
                cells.append(
                    {
                        "id": f"{e.id}/{task}/r{repeat}",
                        "experiment": e.id,
                        "task": task,
                        "repeat": repeat,
                        "seconds": e.seconds,
                    }
                )
    if requested - covered or not cells:
        raise ValueError("selection has no matching cells for requested tasks")
    definition = suite.model_dump(mode="json")
    return {
        "schema_version": 1,
        "suite": definition,
        "cells": cells,
        "plan_sha256": fingerprint({"suite": definition, "cells": cells}),
        "max_gpu_hours": sum(c["seconds"] for c in cells) / 3600,
    }
