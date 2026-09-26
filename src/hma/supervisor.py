'HMA controller for capped alternation with a shared workspace and final Review.'

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

import docker
import requests
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .handoff import atomic_json, copy_verified

PACKAGE = Path(__file__).resolve().parent
NAME = "hma"
LABEL = "hma.owner"

TEARDOWN_MARGIN_SECONDS = 30.0

MIN_FINALIZE_SECONDS = 60.0

class ActorFailure(RuntimeError):
    'An actor turn exited non-zero: the cell is discarded, deliberately.'

class Mount(BaseModel):
    'A trusted, pre-staged read-only task/control mount.'

    model_config = ConfigDict(extra="forbid")
    source: Path
    target: str

class SeedFile(BaseModel):
    'Explicit authentication/configuration file, not a previous actor HOME.'

    model_config = ConfigDict(extra="forbid")
    source: Path
    target: str

class Actor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: str
    environment_names: list[str] = Field(default_factory=list)
    seed_files: list[SeedFile] = Field(default_factory=list)

class Config(BaseModel):
    'Host-only launch contract. No pool enrollment or remote-node selection.'

    model_config = ConfigDict(extra="forbid")
    root: Path
    task_file: Path
    agent_image: str
    evaluator_image: str
    evaluator_seed_home: Path
    evaluator: dict[str, Any] | None = None
    native_evaluator_relative: str = "workspace/.flowbench/mle-evaluator-server.py"
    evaluator_python: str = "python3"
    controller_container: str | None = None
    kimi_proxy_module: Path | None = None
    evaluator_cpus: float = 2
    evaluator_memory: str = "4096m"
    actors: tuple[Actor, Actor]
    agent_data: list[Mount]
    evaluator_data: list[Mount]
    active_time_limit_seconds: int = Field(default=21600, gt=0, strict=True)
    max_valid_submissions_per_session: int = Field(default=5, ge=1, strict=True)
    cpus: float = Field(default=26, gt=0)
    memory: str = "200000m"
    shm_size: str = "65536m"
    gpu: str | None = "0"
    uid: int = Field(default_factory=os.getuid, gt=0)
    gid: int = Field(default_factory=os.getgid)

    review_turn_seconds: int = Field(default=600, gt=0, strict=True)

    review_reserve_seconds: int = Field(default=900, gt=0, strict=True)
    poll_seconds: float = Field(default=0.25, gt=0, le=5)
    stop_seconds: int = Field(default=5, ge=0, le=30)

    @model_validator(mode="after")
    def validate_paths(self) -> Config:
        'Refuse broad paths, unsafe mount targets, and historical HOME copying.'
        if self.uid <= 0:
            raise ValueError("actor/evaluator UID must be non-root")
        if (
            not self.root.is_absolute()
            or len(self.root.parts) < 5
            or self.root.is_symlink()
        ):
            raise ValueError("root must be a new, explicit campaign directory")
        if not self.task_file.is_file() or not self.evaluator_seed_home.is_dir():
            raise ValueError("prepared task and evaluator seed are required")
        for mounts in [self.agent_data, self.evaluator_data]:
            targets: set[str] = set()
            for mount in mounts:
                target = Path(mount.target)
                if (
                    not mount.source.is_absolute()
                    or not mount.source.exists()
                    or not target.is_relative_to("/home/user/.flowbench-data")
                    or ".." in target.parts
                    or str(target) in targets
                ):
                    raise ValueError(
                        "data mounts must be unique, explicit, read-only .flowbench-data paths"
                    )
                targets.add(str(target))
        if not any(
            m.target == "/home/user/.flowbench-data/input" for m in self.agent_data
        ):
            raise ValueError("agent_data must include the public input directory")
        if any(
            not Path(m.target).is_relative_to("/home/user/.flowbench-data/input")
            for m in self.agent_data
        ):
            raise ValueError(
                "actors may mount public input only, never evaluator/control data"
            )
        native = Path(self.native_evaluator_relative)
        if native.is_absolute() or ".." in native.parts:
            raise ValueError("native evaluator path must be relative")
        allowed_seeds = {
            ".codex/auth.json",
            ".codex/config.toml",
            ".codex/cloud-config-bundle-cache.json",
            ".kimi/config.toml",
            ".kimi-code/config.toml",
            ".claude/settings.json",
            ".claude/.credentials.json",
            ".config/kimi/config.toml",
        }
        for actor in self.actors:
            for seed in actor.seed_files:
                if (
                    seed.target not in allowed_seeds
                    or seed.source.is_symlink()
                    or not seed.source.is_file()
                ):
                    raise ValueError(
                        "only explicit regular provider auth/config seeds are allowed"
                    )
            if len({s.target for s in actor.seed_files}) != len(actor.seed_files):
                raise ValueError("duplicate provider seed")
            for name in actor.environment_names:
                if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name) or name in {
                    "HOME",
                    "PYTHONPATH",
                    "BASH_ENV",
                    "DOCKER_HOST",
                    "HUMANIZE_HOME",
                }:
                    raise ValueError("unsafe provider environment name")
                if not os.environ.get(name):
                    raise ValueError(f"missing provider environment variable: {name}")
        return self

    @model_validator(mode="after")
    def validate_review_budget(self) -> Config:
        'The reserve must actually fit everything it is reserved for.'
        owed = (
            self.review_turn_seconds
            + self.stop_seconds
            + TEARDOWN_MARGIN_SECONDS
            + MIN_FINALIZE_SECONDS
        )
        if owed >= self.review_reserve_seconds:
            raise ValueError(
                "review_reserve_seconds must exceed review_turn_seconds plus teardown "
                "and the finalize budget"
            )
        if self.review_reserve_seconds >= self.active_time_limit_seconds:
            raise ValueError("review reserve leaves no exploration time")
        return self

def prompt_for(task: str, seconds: int) -> str:
    'Adapt the budget only. The prompt is otherwise the plain HMA one.'
    duration = f"{seconds // 3600}-hour" if seconds % 3600 == 0 else f"{seconds}-second"
    task = re.sub(r"\b\d+(?:\.\d+)?-hour\b", duration, task)
    task = task.replace(
        "There is no medal-based stopping condition and no submission quota.",
        "There is no medal-based stopping condition.",
    )
    if "no submission quota" in task.lower():
        raise ValueError("unrecognized prompt contradicts the session quota")
    policy = "Do not seek existing solutions, notebooks, discussions, write-ups, recovered labels, or answer files for this task."
    if "## Test-time information policy" not in task:
        task += "\n\n## Test-time information policy\n\n" + policy
    return task

def review_prompt_for(task: str, seconds: int) -> str:
    'The closing review turn: one ballot, no new work.'
    if "no submission quota" in task.lower():
        raise ValueError("review prompt built from the unrendered task statement")
    if not isinstance(seconds, int) or isinstance(seconds, bool) or seconds < 1:
        raise ValueError("review prompt needs this turn's wall in whole seconds")
    return (
        task
        + f"""

## Closing review -- one ballot, nothing else

Exploration on this cell is closed and no further submissions will be accepted for it.
Nothing you write to the working tree from here on counts, and this session cannot
submit: the attempt will be refused.

The other model wrote the submission that currently stands as this cell's result. Your
only job is to say whether it should stand.

`.flowbench/review/ballot.json` lists every submission this cell accepted, in order, with
which side wrote each one and which is standing. `.flowbench/review/candidates/N.csv`
holds the artifact for sequence N. The working tree is the one both of you worked in, so
whatever either of you measured is still here.

Write `.flowbench/review/nomination.json`:

    {{"nominate": "<submission_id>"}}

Name the standing submission if it should stand. Name a different accepted submission if
you are confident it is better -- it then becomes the result. Write the file early and
revise it if you change your mind; a turn that ends with no file leaves the standing
submission in place.

No scores or leaderboard positions are available to you, here or anywhere. Judge on the
evidence in the working tree.

## Your wall clock for this turn

You have {seconds} seconds. Any duration stated earlier in this document described the
exploration phase, which is over; it does not apply to you. At {seconds} seconds this
session is killed outright -- there is no grace period and no chance to finish a thought,
so a nomination you were still composing is simply lost. Budget accordingly: read the
ballot, write `nomination.json` early even if you expect to revise it, and treat any
further reading as optional.
"""
    )

class Supervisor:
    "Own only containers labelled with this new experiment's random identifier."

    def __init__(self, config: Config, client: Any = None) -> None:
        self.config = config
        self.client = client or docker.from_env(timeout=20)
        self.owner = secrets.token_hex(8)
        self.resources: list[Any] = []
        self.network: Any = None
        self.deadline = 0.0
        self.url = ""
        self.key = secrets.token_urlsafe(32)
        self.runtime = config.root / "runtime" / NAME

        self.shared_workspace = config.root / "workspace"
        self.evaluator_container: Any = None
        self.review_deadline = 0.0
        self.explore_deadline = 0.0

    def control(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        timeout: float = 5,
        retries: int = 0,
    ) -> dict[str, Any]:
        'One control call, with a budget the CALLER sizes.'
        attempt = 0
        while True:
            try:
                response = requests.post(
                    self.url + "/turn/" + endpoint,
                    json=payload,
                    headers={"X-Control-Key": self.key},
                    timeout=timeout,
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as error:

                if attempt >= retries or getattr(error, "response", None) is not None:
                    raise
                attempt += 1
                time.sleep(0.5)

    def close_exploration_turn(
        self, turn_id: str, opened: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        'Close a submitting turn, and degrade without inventing anything.'
        config = self.config
        budget = min(
            180.0,
            max(
                30.0,
                self.deadline
                - config.review_turn_seconds
                - config.stop_seconds
                - time.time(),
            ),
        )
        try:
            return self.control("close", {"id": turn_id}, timeout=budget)
        except Exception as error:  
            failure = type(error).__name__

            try:
                records = self.ledger_records()
            except (OSError, ValueError):
                records = []
            baseline = int((opened or {}).get("baseline") or 0)
            return {
                "id": turn_id,
                "closed": True,

                "close_failed": failure,
                "accepted": max(0, len(records) - baseline),
                "last_submission_id": (
                    records[-1]["submission_id"] if records else None
                ),
            }

    def stop(self, container: Any) -> None:
        'Kill the entire container cgroup, never a fuzzy host process pattern.'
        container.reload()
        if container.labels.get(LABEL) != self.owner:
            raise RuntimeError("refusing to stop an unowned container")
        if container.attrs["State"].get("Running"):
            container.stop(timeout=self.config.stop_seconds)
        container.reload()
        if container.attrs["State"].get("Running") or container.attrs["State"].get(
            "Restarting"
        ):
            raise RuntimeError("container survived stop; refusing handoff")
        if container.attrs["State"].get("Pid", 0) != 0:
            raise RuntimeError("container PID still present; refusing handoff")

    def inject(
        self,
        container: Any,
        files: dict[str, bytes],
        *,
        private: set[str] | None = None,
    ) -> None:
        'Copy immutable protocol files into a stopped container, with no host mounts.'
        archive = io.BytesIO()
        directories: set[str] = set()
        with tarfile.open(fileobj=archive, mode="w") as bundle:
            for path, payload in files.items():
                if (
                    not path.startswith(("opt/hma/", "run/hma/"))
                    or ".." in Path(path).parts
                ):
                    raise ValueError("Unsafe protocol injection path")
                for parent in reversed(Path(path).parents):
                    name = str(parent)
                    if name in {".", "opt", "run"} or name in directories:
                        continue
                    entry = tarfile.TarInfo(name)
                    entry.type = tarfile.DIRTYPE
                    entry.mode = 0o755
                    bundle.addfile(entry)
                    directories.add(name)
                entry = tarfile.TarInfo(path)
                entry.mode = 0o444
                if private and path in private:
                    entry.mode = 0o400
                    entry.uid, entry.gid = self.config.uid, self.config.gid
                entry.size = len(payload)
                bundle.addfile(entry, io.BytesIO(payload))
        if not container.put_archive("/", archive.getvalue()):
            raise RuntimeError("Protocol file injection failed")

    def latest_candidate(self) -> Path | None:
        ledger = (
            self.config.root / "evaluator/workspace/.flowbench/mle-submissions.jsonl"
        )
        if not ledger.exists():
            return None
        valid = [
            json.loads(line) for line in ledger.read_text().splitlines() if line.strip()
        ]
        if not valid:
            return None
        record = valid[-1]
        path = ledger.parent / "mle-candidates" / f"{record['artifact_sha256']}.csv"
        with path.open("rb") as artifact:
            if (
                hashlib.file_digest(artifact, "sha256").hexdigest()
                != record["artifact_sha256"]
            ):
                raise RuntimeError("latest accepted artifact hash mismatch")
        return path

    def ledger_records(self) -> list[dict[str, Any]]:
        'Every accepted submission, in order. Identity and time only are used.'
        ledger = (
            self.config.root / "evaluator/workspace/.flowbench/mle-submissions.jsonl"
        )
        if not ledger.exists():
            return []
        return [
            json.loads(line)
            for line in ledger.read_text().splitlines()
            if line.strip()
        ]

    @staticmethod
    def exploration_failure(error: Exception) -> str:
        'Name what ended exploration when a control call raised.'
        response = getattr(error, "response", None)
        if response is not None and getattr(response, "status_code", None) == 409:
            if "expired deadline" in str(getattr(response, "text", "")):
                return "deadline"
        return "control_failed"

    @staticmethod
    def attribution(
        turns: list[dict[str, Any]], records: list[dict[str, Any]]
    ) -> dict[str, int]:
        'Map every accepted submission id to the actor slot that produced it.'
        authors: dict[str, int] = {}
        index = 0
        for turn in turns:
            actor = int(turn.get("actor", int(turn.get("turn", 0)) % 2))
            end = turn.get("last_submission_id")

            if turn.get("accepted") is None:
                return {}
            accepted = int(turn["accepted"] or 0)
            if accepted <= 0:

                continue
            taken = 0
            while index < len(records) and taken < accepted:
                record = records[index]
                authors[str(record["submission_id"])] = actor
                index += 1
                taken += 1
                if end and record["submission_id"] == end:
                    break
        return authors

    @staticmethod
    def standing_author(
        turns: list[dict[str, Any]],
        records: list[dict[str, Any]],
        submission_id: str,
    ) -> int | None:
        'Which actor slot produced a given accepted submission, or None if unknown.'
        return Supervisor.attribution(turns, records).get(str(submission_id))

    def review_phase(self, task_text: str, turns: list[dict[str, Any]]) -> dict[str, Any]:
        'One ballot, cast by the model that did NOT write the standing submission.'
        config = self.config
        records = self.ledger_records()
        if len(records) < 2:
            return {"reason": "review_noop", "detail": "fewer than two candidates"}
        standing = records[-1]
        authors = self.attribution(turns, records)
        author = authors.get(str(standing["submission_id"]))
        if author is None:
            return {"reason": "review_skipped", "detail": "standing submission unattributed"}
        reviewer = (author + 1) % 2

        home = config.root / "turns" / "review" / "agent"
        workspace = self.shared_workspace
        review = workspace / ".flowbench/review"
        if review.exists():
            shutil.rmtree(review)
        review.mkdir(parents=True)

        atomic_json(
            review / "ballot.json",
            {
                "candidates": [
                    {
                        "submission_id": r["submission_id"],
                        "sequence": r["sequence"],
                        "accepted_at_utc": r["accepted_at_utc"],
                        "author": (
                            "you"
                            if authors.get(str(r["submission_id"])) == reviewer
                            else "peer"
                        ),
                        "standing": r["submission_id"] == standing["submission_id"],
                    }
                    for r in records
                ],
                "standing_submission_id": standing["submission_id"],
            },
        )
        candidates = review / "candidates"
        candidates.mkdir()
        source = config.root / "evaluator/workspace/.flowbench/mle-candidates"
        for record in records:
            artifact = source / f"{record['artifact_sha256']}.csv"
            if artifact.is_file() and not artifact.is_symlink():
                copy_verified(artifact, candidates / f"{record['sequence']}.csv")

        if turns and turns[-1].get("close_failed") and turns[-1].get("id"):
            try:
                self.control(
                    "close", {"id": str(turns[-1]["id"])}, timeout=30.0
                )
            except Exception:  
                pass

        turn_id = f"{self.owner}:review"

        opened = self.control(
            "open",
            {"id": turn_id, "limit": 0, "deadline_epoch": self.review_deadline},
        )
        route = home.parent / "route.json"
        home.mkdir(parents=True, exist_ok=True)
        evaluator = self.evaluator_container
        evaluator.reload()
        atomic_json(
            route,
            {
                "host": evaluator.attrs["NetworkSettings"]["Networks"][
                    self.network.name
                ]["IPAddress"],
                "port": 80,
                "token": opened["token"],
                "deadline_epoch": self.review_deadline,
            },
        )
        route.chmod(0o444)
        container = self.actor(
            reviewer, home, route,
            review_prompt_for(task_text, self.config.review_turn_seconds),
            container_name=f"hma-{self.owner}-review",
        )
        turn_started = time.time()
        container.start()

        until = min(
            self.review_deadline
            - config.stop_seconds
            - TEARDOWN_MARGIN_SECONDS
            - MIN_FINALIZE_SECONDS,
            turn_started + config.review_turn_seconds,
        )
        while time.time() < until:
            container.reload()
            if not container.attrs["State"].get("Running"):
                break
            time.sleep(config.poll_seconds)
        container.reload()

        timed_out = bool(container.attrs["State"].get("Running"))
        self.stop(container)
        container.reload()

        exit_code = container.attrs["State"].get("ExitCode")
        (home.parent / "container.log").write_bytes(container.logs())
        lived_seconds = round(time.time() - turn_started, 3)
        sessions = self.session_evidence(home)
        outcome: dict[str, Any] = {
            "reviewer_actor": reviewer,
            "standing_author_actor": author,
            "standing_submission_id": standing["submission_id"],
            "nominate": None,
            "exit_code": exit_code,
            "lived_seconds": lived_seconds,
            "timed_out": timed_out,
            "session_files": sessions,
        }

        try:
            self.control(
                "close",
                {"id": turn_id},
                timeout=min(
                    30.0,
                    max(5.0, self.review_deadline - MIN_FINALIZE_SECONDS - time.time()),
                ),
            )
        except Exception as error:  
            outcome["close_failed"] = type(error).__name__

        nominated = self.read_nomination(review, records)
        outcome["nominate"] = nominated
        if nominated is not None and nominated != standing["submission_id"]:
            return self.finalize_nomination(outcome, nominated)

        wall_seconds = max(0.0, until - turn_started)
        if nominated == standing["submission_id"]:
            outcome["reason"] = "review_noop"
        elif not sessions:

            outcome["reason"] = "review_session_died"
        elif timed_out:

            outcome["reason"] = "review_silent"
        elif exit_code not in (0, None) or lived_seconds < min(60.0, wall_seconds):

            outcome["reason"] = "review_session_died"
        else:

            outcome["reason"] = "review_silent"
        return outcome

    @staticmethod
    def session_evidence(home: Path) -> int:
        'Count provider session transcripts under one turn HOME, BOTH backends.'
        total = 0
        for relative in (".claude/projects", ".codex/sessions"):
            root = home / relative
            if not root.is_dir() or root.is_symlink():
                continue
            total += sum(
                1 for p in root.rglob("*.jsonl") if p.is_file() and not p.is_symlink()
            )
        return total

    def finalize_nomination(
        self, outcome: dict[str, Any], nominated: str
    ) -> dict[str, Any]:
        'Bind the nomination, then label the result from the LEDGER, not from hope.'
        try:
            outcome["finalized"] = self.control(
                "finalize",
                {"submission_id": nominated},

                timeout=max(MIN_FINALIZE_SECONDS, self.deadline - time.time()),
            )
            outcome["reason"] = "review_finalized"
            return outcome
        except Exception as error:  
            outcome["finalize_error"] = type(error).__name__
            response = getattr(error, "response", None)
            refused = response is not None
            if refused:

                outcome["finalize_refused"] = str(getattr(response, "text", ""))[:400]
        try:
            records = self.ledger_records()
        except (OSError, ValueError):
            records = []

        by_id = {str(r.get("submission_id")): r for r in records}
        nominee_artifact = (by_id.get(str(nominated)) or {}).get("artifact_sha256")
        tail_record = records[-1] if records else {}
        tail = (
            str(tail_record.get("submission_id"))
            if tail_record.get("submission_id") is not None
            else None
        )
        tail_artifact = tail_record.get("artifact_sha256")

        bound = (
            tail is not None
            and isinstance(nominee_artifact, str)
            and bool(nominee_artifact)
            and tail_artifact == nominee_artifact
            and tail != str(nominated)
        )
        if bound:
            outcome["reason"] = "review_finalized"
            outcome.setdefault(
                "finalized",
                {"nominated_from": nominated, "reconciled_from_ledger": True},
            )
        elif tail is not None and tail == str(outcome["standing_submission_id"]):
            outcome["reason"] = (
                "review_finalize_refused" if refused else "review_finalize_failed"
            )
        else:
            outcome["reason"] = "review_finalize_unconfirmed"
        return outcome

    @staticmethod
    def read_nomination(review: Path, records: list[dict[str, Any]]) -> str | None:
        'Parse the single ballot, or return None. Never raises.'
        path = review / "nomination.json"
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 8192:
                return None
            nominated = json.loads(path.read_text())["nominate"]
            if not isinstance(nominated, str):
                return None
            if not any(r["submission_id"] == nominated for r in records):
                return None
            return nominated
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def evaluator(self) -> Any:
        config = self.config
        target = config.root / "evaluator"
        shutil.copytree(config.evaluator_seed_home, target, symlinks=True)
        ledger = target / "workspace/.flowbench/mle-submissions.jsonl"
        if ledger.exists() and ledger.stat().st_size:
            raise ValueError("evaluator seed must be fresh, not a historical attempt")
        secret = config.root / "control/admin-key"
        secret.parent.mkdir(parents=True)
        secret.write_text(self.key)
        secret.chmod(0o600)
        (config.root / "control/turns").mkdir()
        (target / "workspace/.flowbench/hma-state").mkdir()
        for data in config.evaluator_data:
            name = Path(data.target).relative_to("/home/user/.flowbench-data").parts[0]
            link = target / "workspace" / name
            if not link.exists() and not link.is_symlink():
                link.symlink_to("/home/user/.flowbench-data/" + name)
        for tree in (target, config.root / "control"):
            for base, dirs, files in os.walk(tree):
                os.chown(base, config.uid, config.gid)
                for name in dirs + files:
                    os.lchown(Path(base) / name, config.uid, config.gid)
        volumes = {
            str(target): {"bind": "/home/user", "mode": "rw"},
        }
        volumes.update(
            {
                str(m.source.resolve()): {"bind": m.target, "mode": "ro"}
                for m in config.evaluator_data
            }
        )
        container = self.client.containers.create(
            config.evaluator_image,
            entrypoint=[],
            command=[
                config.evaluator_python,
                "-m",
                f"{NAME}.evaluator",
                "--base",
                f"/home/user/{config.native_evaluator_relative}",
                "--control",
                "/home/user/workspace/.flowbench/hma-state/turns.json",
                "--key-file",
                "/run/hma/admin-key",
            ],
            name=f"hma-{self.owner}-evaluator",
            labels={
                LABEL: self.owner,
                "io.flowbench.execution": self.owner,
                "io.flowbench.role": "evaluator",
            },
            user=f"{config.uid}:{config.gid}",
            environment={"PYTHONPATH": "/opt/hma", "HOME": "/home/user"},
            working_dir="/home/user/workspace",
            volumes=volumes,
            network=self.network.name,
            ports={"80/tcp": ("127.0.0.1", None)},
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            nano_cpus=int(config.evaluator_cpus * 1_000_000_000),
            mem_limit=config.evaluator_memory,
        )
        self.resources.append(container)
        files = {
            f"opt/hma/{NAME}/{p.name}": p.read_bytes()
            for p in self.runtime.glob("*.py")
        }
        files["run/hma/admin-key"] = self.key.encode()
        self.inject(container, files, private={"run/hma/admin-key"})
        container.start()
        container.reload()
        port = container.attrs["NetworkSettings"]["Ports"]["80/tcp"][0]["HostPort"]
        self.url = f"http://127.0.0.1:{port}"
        if config.controller_container:
            address = container.attrs["NetworkSettings"]["Networks"][self.network.name][
                "IPAddress"
            ]
            self.url = f"http://{address}:80"
        for _ in range(60):
            try:
                if requests.get(self.url + "/healthz", timeout=1).ok:
                    return container
            except requests.RequestException:
                pass
            container.reload()
            if not container.attrs["State"].get("Running"):
                raise RuntimeError("guarded evaluator failed to start")
            time.sleep(0.5)
        raise RuntimeError("guarded evaluator health timeout")

    def actor(
        self,
        index: int,
        home: Path,
        route: Path,
        prompt: str,
        *,
        container_name: str | None = None,
    ) -> Any:
        config = self.config
        actor = config.actors[index % 2]
        for seed in actor.seed_files:
            copy_verified(seed.source, home / seed.target)
            (home / seed.target).chmod(0o600)

        workspace = self.shared_workspace
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "solution").mkdir(exist_ok=True)
        target_input = workspace / "input"
        if not target_input.exists() and not target_input.is_symlink():
            target_input.symlink_to("/home/user/.flowbench-data/input")
        (workspace / ".flowbench").mkdir(exist_ok=True)
        (workspace / ".flowbench/task.md").write_text(prompt)
        submit_link = workspace / "mle_submit.py"
        if not submit_link.exists() and not submit_link.is_symlink():
            submit_link.symlink_to("/opt/hma/submit_client.py")
        for tree in (home, workspace):
            for root, directories, files in os.walk(tree):
                os.chown(root, config.uid, config.gid)
                for name in directories + files:
                    os.lchown(Path(root) / name, config.uid, config.gid)
        environment = {name: os.environ[name] for name in actor.environment_names}
        environment.update(HOME="/home/user", HUMANIZE_HOME="/home/user/.humanize")
        volumes = {
            str(home): {"bind": "/home/user", "mode": "rw"},
            
            str(workspace): {"bind": "/home/user/workspace", "mode": "rw"},
        }
        volumes.update(
            {
                str(m.source.resolve()): {"bind": m.target, "mode": "ro"}
                for m in config.agent_data
            }
        )
        devices = (
            []
            if config.gpu is None
            else [
                docker.types.DeviceRequest(
                    device_ids=[config.gpu], capabilities=[["gpu"]]
                )
            ]
        )
        container = self.client.containers.create(
            config.agent_image,
            entrypoint=[],
            command=[
                "python3",
                "/opt/hma/actor_entry.py",
                "--",
                "hmz",
                "exec",
                "-f",
                "/opt/hma/actor_turn",
                "-a",
                actor.spec,
                "--",
                prompt,
            ],

            name=container_name or f"hma-{self.owner}-turn-{index:05d}",
            labels={
                LABEL: self.owner,
                "io.flowbench.execution": self.owner,
                "io.flowbench.role": "agent",
                "io.flowbench.gpus": config.gpu or "",
            },
            user=f"{config.uid}:{config.gid}",
            working_dir="/home/user/workspace",
            environment=environment,
            volumes=volumes,
            network=self.network.name,
            device_requests=devices,
            nano_cpus=int(config.cpus * 1_000_000_000),
            mem_limit=config.memory,
            shm_size=config.shm_size,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            init=True,
        )
        self.resources.append(container)
        files = {
            "opt/hma/actor_turn/__init__.py": (
                self.runtime / "actor_turn.py"
            ).read_bytes(),
            "opt/hma/actor_entry.py": (
                self.runtime / "actor_entry.py"
            ).read_bytes(),
            "opt/hma/submit_client.py": (
                self.runtime / "submit_client.py"
            ).read_bytes(),
            "run/hma/route.json": route.read_bytes(),
        }
        if config.kimi_proxy_module and "cli=kimi" in actor.spec:
            files["opt/hma/provider_proxy.py"] = (
                self.runtime / "provider_proxy.py"
            ).read_bytes()
        self.inject(container, files)
        return container

    def run(self) -> dict[str, Any]:
        'New experiment only; restart of an ambiguous controller fails closed.'
        config = self.config
        config.root.mkdir(parents=True, exist_ok=False)
        config.root.chmod(0o700)
        shutil.copytree(
            PACKAGE, self.runtime, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        if config.kimi_proxy_module:
            copy_verified(config.kimi_proxy_module, self.runtime / "provider_proxy.py")
        atomic_json(config.root / "contract.json", config.model_dump(mode="json"))
        outcome: dict[str, Any] = {"status": "failed", "turns": []}
        atomic_json(
            config.root / "state.json", {"status": "starting", "owner": self.owner}
        )
        try:
            self.network = self.client.networks.create(
                f"hma-{self.owner}", labels={LABEL: self.owner}
            )
            if config.controller_container:
                self.network.connect(config.controller_container)
            evaluator = self.evaluator()
            self.evaluator_container = evaluator
            prompt = prompt_for(
                config.task_file.read_text(),
                config.active_time_limit_seconds,
            )
            started = time.time()
            self.deadline = started + config.active_time_limit_seconds

            self.explore_deadline = self.deadline - config.review_reserve_seconds
            if self.explore_deadline <= started:
                raise ValueError("review reserve leaves no exploration time")
            atomic_json(
                config.root / "deadline.json",
                {"started_epoch": started, "deadline_epoch": self.deadline},
            )
            index = 0

            exploration_reason: str | None = None
            inflight_turn: str | None = None
            inflight_opened: dict[str, Any] | None = None
            inflight_container: Any = None
            try:
                while time.time() < self.explore_deadline:
                    home = config.root / "turns" / f"{index:05d}" / "agent"

                    workspace = self.shared_workspace
                    (home / "workspace").mkdir(parents=True, exist_ok=True)
                    turn_id = f"{self.owner}:{index}"
                    opened = self.control(
                        "open",
                        {
                            "id": turn_id,
                            "limit": config.max_valid_submissions_per_session,
                            "deadline_epoch": self.explore_deadline,
                        },
                        retries=2,
                    )
                    inflight_turn, inflight_opened = turn_id, opened
                    route = home.parent / "route.json"
                    evaluator.reload()
                    address = evaluator.attrs["NetworkSettings"]["Networks"][
                        self.network.name
                    ]["IPAddress"]
                    atomic_json(
                        route,
                        {
                            "host": address,
                            "port": 80,
                            "token": opened["token"],
                            "deadline_epoch": self.explore_deadline,
                        },
                    )
                    route.chmod(0o444)
                    container = self.actor(index, home, route, prompt)
                    atomic_json(
                        config.root / "state.json",
                        {
                            "status": "running",
                            "owner": self.owner,
                            "turn": index,
                            "container_id": container.id,
                            "deadline_epoch": self.explore_deadline,
                        },
                    )
                    inflight_container = container
                    container.start()
                    reason = "deadline"
                    while time.time() < self.explore_deadline:
                        status = self.control("status", {}, retries=2)
                        if status["exhausted"]:
                            reason = "submission_cap"
                            break
                        container.reload()
                        if not container.attrs["State"].get("Running"):
                            reason = (
                                "natural_exit"
                                if container.attrs["State"].get("ExitCode") == 0
                                else "actor_error"
                            )
                            break
                        time.sleep(config.poll_seconds)
                    self.stop(container)
                    inflight_container = None
                    (home.parent / "container.log").write_bytes(container.logs())

                    status = self.close_exploration_turn(turn_id, opened)
                    inflight_turn, inflight_opened = None, None
                    outcome["turns"].append(
                        {"turn": index, "actor": index % 2, "reason": reason, **status}
                    )
                    atomic_json(config.root / "turn-history.json", outcome["turns"])
                    if reason == "actor_error":
                        raise ActorFailure(
                            "actor failed; artifacts archived, automatic retry refused"
                        )

                    index += 1
            except ActorFailure:
                raise
            except Exception as error:  
                exploration_reason = self.exploration_failure(error)
                outcome["exploration_error"] = type(error).__name__
                if inflight_container is not None:
                    try:
                        self.stop(inflight_container)
                    except Exception:  
                        pass
                if inflight_turn is not None:

                    outcome["turns"].append(
                        {
                            "turn": index,
                            "actor": index % 2,
                            "reason": exploration_reason,
                            **self.close_exploration_turn(
                                inflight_turn, inflight_opened
                            ),
                        }
                    )
                    atomic_json(config.root / "turn-history.json", outcome["turns"])
            exploration_elapsed = time.time() - started
            if exploration_reason is None:
                exploration_reason = (
                    str(outcome["turns"][-1].get("reason"))
                    if outcome["turns"]
                    else "no_turns"
                )

            self.review_deadline = self.deadline
            try:

                review = self.review_phase(prompt, outcome["turns"])
            except Exception as error:  
                review = {
                    "reason": "review_error",
                    "error": type(error).__name__,
                    "message": str(error)[:400],
                }
            outcome["review"] = review
            atomic_json(config.root / "review.json", review)
            outcome.update(
                status="complete",
                reason="global_deadline",
                deadline_epoch=self.deadline,

                exploration_elapsed_seconds=round(exploration_elapsed, 3),
                exploration_reason=exploration_reason,
            )
            return outcome
        finally:
            errors: list[str] = []
            for resource in reversed(self.resources):
                try:
                    self.stop(resource)
                    logs = config.root / "container-logs"
                    logs.mkdir(exist_ok=True)
                    (logs / f"{resource.name}.log").write_bytes(resource.logs())
                    resource.remove()
                except (docker.errors.DockerException, OSError, RuntimeError) as error:
                    errors.append(type(error).__name__)
            if self.network is not None and not errors:
                if config.controller_container:
                    self.network.disconnect(config.controller_container)
                self.network.remove()
            outcome["cleanup_errors"] = errors
            if errors:
                outcome["status"] = "cleanup_failed"
            atomic_json(config.root / "result.json", outcome)
            atomic_json(
                config.root / "state.json",
                {
                    "status": outcome["status"],
                    "owner": self.owner,
                    "deadline_epoch": self.deadline,
                },
            )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = Config.model_validate_json(args.config.read_text())
    if args.validate_only:
        print("Configuration valid; no containers or tasks started.")
        return
    outcome = Supervisor(config).run()
    print(json.dumps(outcome, indent=2))
    if outcome["status"] != "complete":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
