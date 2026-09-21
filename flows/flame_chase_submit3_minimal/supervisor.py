"""Trusted Docker-host controller for submission-capped, artifacts-only Flame Chase.

Never run this module inside an actor or give actors the Docker socket. It uses
one prepared native evaluator and sequential, fresh HOME/container actor turns.
"""

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
NAME = "flame_chase_submit3_artifacts_only"
LABEL = "flowverse.submit3.owner"

#: Seconds of the reserve that belong to teardown rather than to the ballot: the
#: reviewer's container stop, its log write and the candidate tree it was given.
#: Measured last-turn teardown over 62 archived cells was p50 5.5 s / max 7.8 s, but
#: mbm_17 and mbm_11 carry ~3 GB candidate corpora, so the margin is headroom over a
#: measured typical case, not a proof.
TEARDOWN_MARGIN_SECONDS = 30.0
#: Seconds of the reserve that belong to /turn/finalize, which re-runs the official
#: grader synchronously. The slowest archived grader measured 30.2 s (mbh_01). Held
#: back explicitly so the ballot turn can never eat the budget the binding call needs
#: -- the failure D1 documents -- and so the finalize budget never has to be a floor
#: that would let the call run past self.deadline.
MIN_FINALIZE_SECONDS = 60.0


class ActorFailure(RuntimeError):
    """An actor turn exited non-zero: the cell is discarded, deliberately.

    Its own class so the exploration loop's control-failure handler can let it
    through. Swallowing it would let a cell that did no work report `complete` with
    a full-budget elapsed and zero submissions, which the pool would accept as clean.
    """


class Mount(BaseModel):
    """A trusted, pre-staged read-only task/control mount."""

    model_config = ConfigDict(extra="forbid")
    source: Path
    target: str


class SeedFile(BaseModel):
    """Explicit authentication/configuration file, not a previous actor HOME."""

    model_config = ConfigDict(extra="forbid")
    source: Path
    target: str


class Actor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: str
    environment_names: list[str] = Field(default_factory=list)
    seed_files: list[SeedFile] = Field(default_factory=list)


class Config(BaseModel):
    """Host-only launch contract. No pool enrollment or remote-node selection."""

    model_config = ConfigDict(extra="forbid")
    root: Path
    task_file: Path
    agent_image: str
    evaluator_image: str
    evaluator_seed_home: Path
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
    max_valid_submissions_per_session: int = Field(default=3, ge=1, strict=True)
    cpus: float = Field(default=26, gt=0)
    memory: str = "200000m"
    shm_size: str = "65536m"
    gpu: str | None = "0"
    uid: int = Field(default_factory=os.getuid, gt=0)
    gid: int = Field(default_factory=os.getgid)
    #: The closing review is one turn, so its wall IS its turn length plus teardown.
    #: The multi-round variant reserved 2700 s and never used more than 1780 s of it.
    review_turn_seconds: int = Field(default=600, gt=0, strict=True)
    #: The review is INSIDE the budget, not after it. This many seconds of the
    #: active-time limit are kept back for it: exploration turns are given a
    #: deadline this far short of the cell's, and the review turn is given the
    #: cell's own. Nothing runs past the six-hour cutoff.
    review_reserve_seconds: int = Field(default=900, gt=0, strict=True)
    poll_seconds: float = Field(default=0.25, gt=0, le=5)
    stop_seconds: int = Field(default=5, ge=0, le=30)

    @model_validator(mode="after")
    def validate_paths(self) -> Config:
        """Refuse broad paths, unsafe mount targets, and historical HOME copying."""
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
        """The reserve must actually fit everything it is reserved for.

        The sibling flow's version of this check permits wall == turns * turn_seconds,
        which is the same hole: a 900/900 contract gives the ballot turn the whole
        reserve and leaves its own close, and the finalize that BINDS it, with nothing.
        The ballot is then cast and discarded at the wall. Refuse that at load rather
        than discovering it at the cutoff of a six-hour cell.

        The wrong way to satisfy this is to raise review_turn_seconds to the reserve:
        that passes the worker adapter's receipt gate by coincidence and re-creates the
        discarded-ballot failure on 100% of cells.
        """
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
    """Adapt the budget only. The prompt is otherwise the plain Flame Chase one.

    This variant deliberately says NOTHING about the submission cap or about what a
    successor inherits, and that is the whole point of it:

    * The cap is enforced but undisclosed. The controller ends a session after its
      `cap`-th accepted submission and the actor is never told a cap exists, so the
      measurement is of behaviour under an unannounced cut rather than of planning
      around a number. The cap is NOT a parameter here: it takes no part in the text,
      and a parameter that is accepted and ignored is how a number leaks into a prompt
      later. The controller enforces it; this function cannot mention what it is not
      given. (It was a parameter until 2026-09-17, when the worker adapter -- which
      renders the same prompt into task.md before the controller ever starts -- called
      it with two arguments and every claim died in TypeError before the cell began.)
    * There is no allowlist and no export. Turns share one workspace, exactly as
      Flame Chase does, so every sentence the artifacts-only variant added about
      inherited files, dropped logs and non-inherited HOME would now be false.

    The one substitution that stays is the quota sentence. The stock task text asserts
    "no submission quota", which under a hidden cap would be a lie to the actor; it is
    replaced by the half that is still true rather than by a hint that a cap exists.
    """
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
    """The closing review turn: one ballot, no new work.

    `seconds` is this turn's actual wall and MUST be stated. Until 2026-09-18 it was
    not a parameter at all: the reviewer inherited the rendered task statement, which
    says the cell had six hours, and nothing anywhere corrected it for a turn that is
    killed at 600 s. The reviewers that paced themselves for hours were therefore
    SIGTERMed mid-thought -- 33 of glm53's 35 blank ballots died at exit 143 within
    400 ms of the 600 s wall, while gpt56sol, which happens to work fast, finished
    every one of its 9 inside the window and produced all four of the arm's medal
    flips. That reads like "one model cannot review" and is really "one model was
    never told when to stop". A turn given a wall it cannot see is not a measurement
    of judgement.

    Built on the RENDERED task statement -- what the actors were actually given -- so
    the reviewer knows what the cell was for, then contradicted where it has to be:
    the body tells an actor to keep submitting and that no wrap-up stage exists, and
    both are false for this turn.

    `task` must be the output of prompt_for, not the raw staged file. It was the raw
    file until 2026-09-17, which handed the ONE model that votes the stock "no
    submission quota" sentence this variant exists to remove -- so the undisclosed cap
    leaked, in the one turn that is the measurement, and on a short cell the duration
    was unrewritten too. The guard below makes a regression a loud review_error instead
    of a quietly off-protocol measured turn.
    """
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
    """Own only containers labelled with this new experiment's random identifier."""

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
        #: One working tree for the whole cell. Flame Chase alternates two agents in the
        #: same repository; this variant keeps that and adds only the undisclosed
        #: submission cap and the closing single-ballot review.
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
        """One control call, with a budget the CALLER sizes.

        Why the budget is per call. open/close/status are bookkeeping: the handler
        returns in microseconds and 5 s is generous. /turn/finalize is not -- it
        re-runs the official grader synchronously (evaluator.py finalize() ->
        _score() -> a `python3 mle-score-worker.py` subprocess) and sends no byte
        of the response until that finishes. With one hardcoded 5 s budget the
        client therefore raised requests.ReadTimeout on every task whose grader
        takes longer, which is nine of the ~46 archived tasks measured
        deterministically (mbh_01 30.2 s, mbm_36 14-30 s, mbm_17 15-21 s) plus a
        4.0-4.8 s cluster that crosses under contention -- while the SERVER
        finished the grade and appended the nominee anyway. review_finalized, the
        one mechanism this arm exists to measure, was unreachable on those tasks,
        and the smoke task (mbm_01, ~2.5 s) is fast enough to ship the defect
        green. Both sibling flows (evidence_countdown_nominate, postdeadline)
        carry this kwarg; the minimal fork kept the finalize call and dropped the
        budget.

        Why not a flat 900/1800 copied from those siblings. Their review ran
        AFTER the cell budget on its own wall, where a long finalize was free.
        Since 2026-09-17 the review runs INSIDE the budget, so every budget here
        is a slice of the reserve computed by the caller against self.deadline. A
        constant larger than the reserve cannot make the grader faster; it only
        converts a bounded failure into an unbounded overrun of the cutoff.

        `retries` is for the idempotent verbs only (status re-reads state; open
        re-serves an identical contract from evaluator.py's `if any(t["id"] ==
        turn_id ...)` branch). finalize and close are never retried here: finalize
        is refused a second time by the evaluator's `already finalized` guard, so
        a retry would turn a landed nomination into a 409.
        """
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
                # A 4xx/5xx is the server's considered answer and never transient;
                # only a transport failure is worth repeating.
                if attempt >= retries or getattr(error, "response", None) is not None:
                    raise
                attempt += 1
                time.sleep(0.5)

    def close_exploration_turn(
        self, turn_id: str, opened: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Close a submitting turn, and degrade without inventing anything.

        Two failures meet here. close_turn takes the evaluator's score lock, and a
        grader that started before the cutoff can still hold it -- on control()'s 5 s
        bookkeeping budget that raised. And since the loop bound moved to
        explore_deadline on 2026-09-17, the raise is no longer inside any protective
        branch: it unwinds run() and a completed six-hour cell is discarded loudly and
        receiptlessly (no termination.json, a /fail, a backoff strike).

        The obvious repair -- wrap it and fall back to {"closed": True} -- is the one
        three verifiers rejected, and rightly: that record carries no `accepted` and no
        `last_submission_id`, so attribution() cannot place the turn's submissions, the
        standing submission is unattributed and the review is skipped. It converts a
        loud cell loss into a silent review loss. So: wait a normal grade out on a
        budget taken from the reserve (never from the ballot turn), and if that expires,
        reconstruct the count from the ledger the evaluator already wrote to the host.
        """
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
        except Exception as error:  # noqa: BLE001 - degraded below, never fatal
            failure = type(error).__name__
            # Guarded: a truncated ledger line here would raise INSIDE the handler that
            # exists to stop a raise, putting the cell straight back on the path this
            # method was written to close.
            try:
                records = self.ledger_records()
            except (OSError, ValueError):
                records = []
            baseline = int((opened or {}).get("baseline") or 0)
            return {
                "id": turn_id,
                "closed": True,
                # The close did not land, so the evaluator's journal still has this
                # turn open. review_phase re-attempts it before opening the ballot
                # turn, or the open would 409 on "previous turn has not closed".
                "close_failed": failure,
                "accepted": max(0, len(records) - baseline),
                "last_submission_id": (
                    records[-1]["submission_id"] if records else None
                ),
            }

    def stop(self, container: Any) -> None:
        """Kill the entire container cgroup, never a fuzzy host process pattern."""
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
        """Copy immutable protocol files into a stopped container, with no host mounts."""
        archive = io.BytesIO()
        directories: set[str] = set()
        with tarfile.open(fileobj=archive, mode="w") as bundle:
            for path, payload in files.items():
                if (
                    not path.startswith(("opt/submit3/", "run/submit3/"))
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
        """Every accepted submission, in order. Identity and time only are used."""
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
        """Name what ended exploration when a control call raised.

        An "expired deadline" 409 on /turn/open is not a fault: the supervisor and the
        actor test the same explore_deadline epoch, so the loop can lose that race by
        a few milliseconds at a cap or hand-back boundary. It is the ordinary end of
        exploration and must not be recorded as a controller error.
        """
        response = getattr(error, "response", None)
        if response is not None and getattr(response, "status_code", None) == 409:
            if "expired deadline" in str(getattr(response, "text", "")):
                return "deadline"
        return "control_failed"

    @staticmethod
    def attribution(
        turns: list[dict[str, Any]], records: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Map every accepted submission id to the actor slot that produced it.

        A turn record is `{turn, actor, reason}` merged with the evaluator's close
        response, so what it carries about the ledger is a COUNT (`accepted`) and an
        END (`last_submission_id`) -- never a range. Submissions are appended in order,
        so walking the ledger and the turn list together recovers the boundaries
        exactly, and the end id is used to resynchronise if a count is ever short.

        This existed as a `ledger_from`/`ledger_to` lookup until 2026-09-17, when the
        first live cell of this variant skipped its review with "standing submission
        unattributed" after a healthy 905 s run: no turn record has ever carried those
        keys. They were invented by the test fixture, and the fixture was believed over
        the producer.
        """
        authors: dict[str, int] = {}
        index = 0
        for turn in turns:
            actor = int(turn.get("actor", int(turn.get("turn", 0)) % 2))
            end = turn.get("last_submission_id")
            # An ABSENT count is not a count of zero. A turn whose close degraded
            # carries no `accepted`, and treating that as "this turn took nothing"
            # leaves `index` where it was, so the NEXT turn's actor is credited with
            # this turn's submissions -- every "you"/"peer" label in ballot.json
            # inverted for that block, and the ballot handed to the model that wrote
            # the thing it is being asked to judge. Wrong data wearing the shape of
            # data, and strictly worse than the loud failure of not attributing at
            # all. The degrade path in run() supplies a ledger-derived count precisely
            # so this branch is not taken; if anything ever reaches it, fail closed and
            # let review_phase report review_skipped.
            if turn.get("accepted") is None:
                return {}
            accepted = int(turn["accepted"] or 0)
            if accepted <= 0:
                # A real zero consumed no ledger positions, so `index` is already
                # right: this turn owns nothing and the boundary does not move.
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
        """Which actor slot produced a given accepted submission, or None if unknown.

        Unknown means the review is skipped rather than handed to an arbitrary side.
        """
        return Supervisor.attribution(turns, records).get(str(submission_id))

    def review_phase(self, task_text: str, turns: list[dict[str, Any]]) -> dict[str, Any]:
        """One ballot, cast by the model that did NOT write the standing submission.

        The whole mechanism, deliberately: no rounds, no peer ballot, no convergence, no
        confidence gate. The reviewer either names the standing submission -- which
        changes nothing -- or names another accepted one, which binds.

        Why one ballot and why that side. In the multi-round variant the closing debate
        produced a 27% blank-ballot rate over 111 turns, and every one of those blanks
        came from the same actor slot (49.2% against 0.0%). Turns that produced nothing
        still consumed their share of a 2700 s wall no cell ever exhausted. Asking only
        the side with no stake in the standing submission removes the wasted turns and
        removes the "sole ballot" ambiguity with them: there is exactly one ballot by
        construction, so there is nothing to decide about what an unopposed one means.

        Silence is agreement here, not a veto and not a tie: a reviewer that writes
        nothing leaves the standing submission standing, which is what would have
        happened without the phase at all.
        """
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
        # A score-free ballot: identity, order and authorship, never a leaderboard
        # verdict. author is "you"/"peer" so the reviewer can be asked to account for
        # its own work without being told which model wrote what.
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

        # A degraded exploration close leaves that turn OPEN in the evaluator journal,
        # and open_turn refuses the next one with "previous turn has not closed" -- a
        # 409 that this phase's handler records as review_error, which is the r4
        # signature produced by a healthy cell. Re-attempt it once, on a small budget,
        # before asking for the ballot turn.
        if turns and turns[-1].get("close_failed") and turns[-1].get("id"):
            try:
                self.control(
                    "close", {"id": str(turns[-1]["id"])}, timeout=30.0
                )
            except Exception:  # noqa: BLE001 - open() below reports it if it mattered
                pass

        turn_id = f"{self.owner}:review"
        # limit 0: this turn must not be able to CREATE anything -- only to re-point at
        # a candidate already accepted. It is also what lets the turn hold the cell's
        # own deadline while every submitting turn stopped at the reserve boundary.
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
            container_name=f"submit3-{self.owner}-review",
        )
        turn_started = time.time()
        container.start()
        # The ballot's wall stops short of the cell's own by the two slices of the
        # reserve that are already owed: teardown, and the finalize that BINDS the
        # ballot. This read `min(self.review_deadline, ...)` until 2026-09-17, so a
        # reviewer that used its whole wall left nothing for its own close -- and
        # evaluator.finalize refuses an unclosed turn, so the one ballot that changed
        # the result was thrown away as review_error. Nothing here runs past
        # self.deadline: `until` is strictly inside it by construction.
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
        # Captured BEFORE the stop. It is the only honest signal that the reviewer was
        # still working when its wall arrived: a live container reports ExitCode 0,
        # which is indistinguishable from a clean, considered exit.
        timed_out = bool(container.attrs["State"].get("Running"))
        self.stop(container)
        container.reload()
        # Read AFTER the stop. With init=True tini forwards SIGTERM, so a wall-killed
        # reviewer reports 143 (or 137 after stop_seconds) rather than the live
        # container's 0. A nonzero code is therefore the NORMAL value for a reviewer
        # that used its whole wall, and it must never fail the cell.
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
        # Unconditional, and tolerant. The old `if time.time() < self.review_deadline`
        # guard protected nothing -- close_turn has no deadline check and no
        # authorize() -- it only skipped the close on exactly the cells that ran to the
        # wall. Bare, though, a transient control failure here would fail the whole
        # phase, so it is recorded and the ballot is still read.
        try:
            self.control(
                "close",
                {"id": turn_id},
                timeout=min(
                    30.0,
                    max(5.0, self.review_deadline - MIN_FINALIZE_SECONDS - time.time()),
                ),
            )
        except Exception as error:  # noqa: BLE001 - recorded, never the phase's verdict
            outcome["close_failed"] = type(error).__name__

        nominated = self.read_nomination(review, records)
        outcome["nominate"] = nominated
        if nominated is not None and nominated != standing["submission_id"]:
            return self.finalize_nomination(outcome, nominated)
        # Three outcomes wore one name until 2026-09-17: a reviewer that read the
        # ballot and endorsed it, a reviewer killed at its wall mid-thought, and a CLI
        # that died on arrival all recorded `review_noop` with no duration and no
        # session evidence. Only the first is agreement; the third is a measurement
        # that did not happen. A prompt-guard failure never reaches here -- it raises
        # inside review_prompt_for, before the container exists, and is recorded as
        # review_error by run()'s handler -- so it cannot be mistaken for a dead
        # session.
        # The wall this turn was ACTUALLY given, not a constant: `until` collapses onto
        # the review deadline on a cell whose setup ate the reserve, and a contract may
        # legitimately carry a review_turn_seconds far under a minute.
        wall_seconds = max(0.0, until - turn_started)
        if nominated == standing["submission_id"]:
            outcome["reason"] = "review_noop"
        elif not sessions:
            # Tested BEFORE timed_out on purpose. A container that HANGS is still
            # Running at the wall, so timed_out is True for a reviewer that produced
            # no session transcript at all -- and reading that as "ran its wall and
            # declined" is exactly the wrong-data-that-looks-like-data failure this
            # split exists to prevent. Absence of evidence that it ever ran outranks
            # the fact that the clock expired.
            outcome["reason"] = "review_session_died"
        elif timed_out:
            # Ran its whole wall WITH session evidence and wrote no valid ballot: a
            # decline, and silence is agreement by design. Never a retry candidate --
            # retrying a decline manufactures the opposite bias.
            #
            # Neither exit_code nor lived_seconds may be consulted on this branch, and
            # both were until 2026-09-17: exit_code is read AFTER the stop, so tini
            # reports 143 (or 137) for every wall kill, and lived_seconds is the wall
            # itself by construction. Testing either against a death signature ahead of
            # this branch made it unreachable -- every wall-killed reviewer, evidence
            # and all, was recorded as a dead session.
            outcome["reason"] = "review_silent"
        elif exit_code not in (0, None) or lived_seconds < min(60.0, wall_seconds):
            # It exited on its OWN, before its wall: a crash, or a CLI that died on
            # arrival fast enough that whatever transcript it left is a stub. The floor
            # is relative to the wall this turn was given -- a flat 60 s condemned every
            # review on any contract whose ballot turn is shorter than that.
            outcome["reason"] = "review_session_died"
        else:
            # Exited cleanly and inside its wall, with evidence, and wrote no ballot:
            # a decline, same as the wall-killed case.
            outcome["reason"] = "review_silent"
        return outcome

    @staticmethod
    def session_evidence(home: Path) -> int:
        """Count provider session transcripts under one turn HOME, BOTH backends.

        A claude-only key is a trap this campaign has already paid for: a receipt guard
        keyed to one backend's field discarded seven gold medals from codex cells. The
        reviewer is whichever side did not write the standing submission, so the backend
        is not known here.
        """
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
        """Bind the nomination, then label the result from the LEDGER, not from hope.

        Two failures were conflated before 2026-09-17, and both discarded the ballot.

        * The call went out on control()'s 5 s bookkeeping budget while the evaluator
          re-ran the official grader, so it raised ReadTimeout on every slow task --
          and the SERVER finished the grade and appended the nominee anyway. The
          receipt said review_error while the cell's measured winner had silently
          changed. So the timeout is now a real slice of the reserve (D1), and the
          ledger -- read off the host bind mount, so it is legible even with the
          evaluator dead -- is what decides what happened.
        * The raise escaped review_phase entirely and the broad handler in run()
          replaced the whole outcome dict with {"reason": "review_error", "error": ...},
          losing the reviewer identity, the nomination and the exit code. The dict is
          kept here whatever happens; the handler must never be the thing that records
          a review outcome.

        The ledger re-read also covers the partial-write window between the evaluator's
        _append and its `finalized = True`, which no drain-before-teardown protocol can
        make observable.
        """
        try:
            outcome["finalized"] = self.control(
                "finalize",
                {"submission_id": nominated},
                # Bounded by the cell deadline by construction, never a flat 900/1800
                # copied from a sibling whose review ran after the budget: a client
                # budget larger than the reserve cannot make the grader faster, it only
                # converts a bounded failure into an overrun of the cutoff. The floor
                # is reachable only if the ballot turn started so late that `until`
                # collapsed -- which validate_review_budget makes impossible for a
                # contract this controller will load, since it holds MIN_FINALIZE_SECONDS
                # back from the reserve for precisely this call.
                timeout=max(MIN_FINALIZE_SECONDS, self.deadline - time.time()),
            )
            outcome["reason"] = "review_finalized"
            return outcome
        except Exception as error:  # noqa: BLE001 - reconciled below, never re-raised
            outcome["finalize_error"] = type(error).__name__
            response = getattr(error, "response", None)
            refused = response is not None
            if refused:
                # A 409 is the evaluator's considered answer (its guards all raise
                # before the append), not a transport failure. Kept distinct so a
                # legitimate rejection is never read as a lost ballot.
                outcome["finalize_refused"] = str(getattr(response, "text", ""))[:400]
        try:
            records = self.ledger_records()
        except (OSError, ValueError):
            records = []
        # finalize APPENDS a fresh record with a NEW submission_id that re-states the
        # nominee's artifact; the ledger carries no nominated_from at all. Comparing
        # submission ids therefore never matches and the reconciliation would report
        # review_finalize_unconfirmed on every success. The artifact digest is what is
        # actually re-stated, so that is what identifies the binding.
        by_id = {str(r.get("submission_id")): r for r in records}
        nominee_artifact = (by_id.get(str(nominated)) or {}).get("artifact_sha256")
        tail_record = records[-1] if records else {}
        tail = (
            str(tail_record.get("submission_id"))
            if tail_record.get("submission_id") is not None
            else None
        )
        tail_artifact = tail_record.get("artifact_sha256")
        # The digest has to be a real one before it can prove anything: str() on both
        # sides made a ledger that carries no artifact_sha256 at all reconcile against
        # itself ("None" == "None") and report a binding that never happened -- the same
        # wrong-data-that-looks-like-data failure, one layer down.
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
        """Parse the single ballot, or return None. Never raises.

        A malformed or absent ballot is a reviewer that did not answer, which this
        variant treats as agreement with the standing submission -- so it must not be
        able to fail the cell.
        """
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
        (target / "workspace/.flowbench/submit3-state").mkdir()
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
                "/home/user/workspace/.flowbench/submit3-state/turns.json",
                "--key-file",
                "/run/submit3/admin-key",
            ],
            name=f"submit3-{self.owner}-evaluator",
            labels={
                LABEL: self.owner,
                "io.flowbench.execution": self.owner,
                "io.flowbench.role": "evaluator",
            },
            user=f"{config.uid}:{config.gid}",
            environment={"PYTHONPATH": "/opt/submit3", "HOME": "/home/user"},
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
            f"opt/submit3/{NAME}/{p.name}": p.read_bytes()
            for p in self.runtime.glob("*.py")
        }
        files["run/submit3/admin-key"] = self.key.encode()
        self.inject(container, files, private={"run/submit3/admin-key"})
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
        # The working tree is SHARED across turns and lives outside the per-turn home;
        # a second bind mount puts it back at /home/user/workspace inside the container.
        # Preparing it here rather than under home/ matters: a write to home/workspace
        # would be hidden by that mount and the actor would never see it.
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
            submit_link.symlink_to("/opt/submit3/submit_client.py")
        for tree in (home, workspace):
            for root, directories, files in os.walk(tree):
                os.chown(root, config.uid, config.gid)
                for name in directories + files:
                    os.lchown(Path(root) / name, config.uid, config.gid)
        environment = {name: os.environ[name] for name in actor.environment_names}
        environment.update(HOME="/home/user", HUMANIZE_HOME="/home/user/.humanize")
        volumes = {
            str(home): {"bind": "/home/user", "mode": "rw"},
            # Mounted OVER home/workspace so every turn writes into the same tree.
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
                "/opt/submit3/actor_entry.py",
                "--",
                "hmz",
                "exec",
                "-f",
                "/opt/submit3/actor_turn",
                "-a",
                actor.spec,
                "--",
                prompt,
            ],
            # container_name, never `name`: the chown walk above rebinds `name`, so a
            # parameter of that spelling would arrive here as the last file walked.
            name=container_name or f"submit3-{self.owner}-turn-{index:05d}",
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
            "opt/submit3/actor_turn/__init__.py": (
                self.runtime / "actor_turn.py"
            ).read_bytes(),
            "opt/submit3/actor_entry.py": (
                self.runtime / "actor_entry.py"
            ).read_bytes(),
            "opt/submit3/submit_client.py": (
                self.runtime / "submit_client.py"
            ).read_bytes(),
            "run/submit3/route.json": route.read_bytes(),
        }
        if config.kimi_proxy_module and "cli=kimi" in actor.spec:
            files["opt/submit3/wuwen_proxy.py"] = (
                self.runtime / "wuwen_proxy.py"
            ).read_bytes()
        self.inject(container, files)
        return container

    def run(self) -> dict[str, Any]:
        """New experiment only; restart of an ambiguous controller fails closed."""
        config = self.config
        config.root.mkdir(parents=True, exist_ok=False)
        config.root.chmod(0o700)
        shutil.copytree(
            PACKAGE, self.runtime, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        if config.kimi_proxy_module:
            copy_verified(config.kimi_proxy_module, self.runtime / "wuwen_proxy.py")
        atomic_json(
            config.root / "runtime-hashes.json",
            {
                str(p.relative_to(self.runtime)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in self.runtime.rglob("*.py")
            },
        )
        atomic_json(config.root / "contract.json", config.model_dump(mode="json"))
        outcome: dict[str, Any] = {"status": "failed", "turns": []}
        atomic_json(
            config.root / "state.json", {"status": "starting", "owner": self.owner}
        )
        try:
            self.network = self.client.networks.create(
                f"submit3-{self.owner}", labels={LABEL: self.owner}
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
            # Two clocks, one budget. Everything that can submit runs against
            # explore_deadline; the closing review runs against the cell's own, which is
            # later -- the evaluator admits a later deadline ONLY for a turn that cannot
            # submit, which is exactly what makes this reserve honest rather than an
            # extension. The cell still ends at self.deadline.
            self.explore_deadline = self.deadline - config.review_reserve_seconds
            if self.explore_deadline <= started:
                raise ValueError("review reserve leaves no exploration time")
            atomic_json(
                config.root / "deadline.json",
                {"started_epoch": started, "deadline_epoch": self.deadline},
            )
            index = 0
            # One transient failure among the ~86,000 status POSTs of a six-hour
            # cell used to unwind run() outright: no termination.json, no review, a
            # completed cell lost receiptless to a blip on a node that also hosts
            # 18-200 GB agent cells. A control failure now ENDS exploration and still
            # reaches the ballot and the teardown receipt. An actor failure is not a
            # control failure and still discards the cell, deliberately: swallowing it
            # would report a zero-work cell as clean.
            exploration_reason: str | None = None
            inflight_turn: str | None = None
            inflight_opened: dict[str, Any] | None = None
            inflight_container: Any = None
            try:
                while time.time() < self.explore_deadline:
                    home = config.root / "turns" / f"{index:05d}" / "agent"
                    # The tree is shared; actor() creates and prepares it. home/workspace is
                    # only a mount point and is deliberately left empty on the host.
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
                    # Do NOT stop the evaluator here. That line is inherited from a variant
                    # whose closing phase ran outside the budget; here the review talks to
                    # this container. On 2026-09-17 the same line, left in place, made every
                    # cell of the postdeadline arm open its review turn against a dead
                    # container and die on ConnectTimeout in a median of 5.2 s -- 21 of 23
                    # finished cells, with stop_reason still reading active_time_limit and
                    # the cell still reporting complete, so nothing downstream could see it.
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
                    # No export. Turns share one workspace, which is what Flame Chase does
                    # and what this variant exists to measure: the successor reads the
                    # repository the predecessor left, not an allowlisted subset of it.
                    # The per-turn home still differs -- credentials, .flowbench/task.md and
                    # the provider session live there -- so a session is still fresh; only
                    # the working tree persists. Copying the tree per turn instead would
                    # have been the obvious way to widen the allowlist to everything, but a
                    # measured cell workspace runs to 4.8 GB and a long cell takes ten
                    # turns, which is 48 GB a cell on nodes that were down to 278 GiB.
                    index += 1
            except ActorFailure:
                raise
            except Exception as error:  # noqa: BLE001 - ends exploration, not the cell
                exploration_reason = self.exploration_failure(error)
                outcome["exploration_error"] = type(error).__name__
                if inflight_container is not None:
                    try:
                        self.stop(inflight_container)
                    except Exception:  # noqa: BLE001 - the finally block re-tries
                        pass
                if inflight_turn is not None:
                    # Only ever a turn the evaluator confirmed open, with its own
                    # baseline, so the degraded record cannot claim submissions that
                    # belong to an earlier turn's actor.
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
            # The closing review runs INSIDE the budget, on the slice reserved from it,
            # so the cell's elapsed time is unchanged BY DESIGN rather than by being
            # excluded: nothing here extends the six hours. It was on its own wall until
            # 2026-09-17, when the operator required the rebuttal to be paid for out of
            # the same budget every other flow is measured on.
            self.review_deadline = self.deadline
            try:
                # The RENDERED prompt, not the raw staged file: `prompt` is assigned
                # unconditionally above, inside this same try, and it is what the
                # actors were given. Re-reading task_file here served the reviewer the
                # stock "no submission quota" sentence -- the one edit this variant
                # makes -- on the only turn that casts a ballot.
                review = self.review_phase(prompt, outcome["turns"])
            except Exception as error:  # noqa: BLE001 - a review failure is not a cell failure
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
                # Both keys the worker adapter already reads via outcome.get(): its
                # receipt gate compares the EXPLORATION span against the budget minus
                # the reserve, and its stop_reason distinguishes a dry handover from a
                # controller that died early. Neither was ever written, so the gate
                # silently fell back to the wall clock -- which is the whole budget by
                # construction and therefore clears unconditionally, exactly the guard
                # the r6 receipt-gate incident needed and did not have.
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
