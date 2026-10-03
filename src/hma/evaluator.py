"Opt-in guarded native evaluator; never modifies the original evaluator file."

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.util
import json
import math
import secrets
import threading
import time
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

from .handoff import atomic_json


def guarded_types(base: ModuleType, control_file: Path, admin_key: str) -> tuple[type, type]:
    "Wrap native scoring/ledger semantics with durable per-session admission."
    request_context = threading.local()

    class State(base.State):
        def __init__(self, config: dict[str, Any]) -> None:
            super().__init__(config)
            if config["feedback_mode"] != "blind" or config["submission_limit"] is not None:
                raise ValueError("hma requires blind evaluation and no cell-wide quota")
            self.turns = json.loads(control_file.read_text()) if control_file.exists() else []
            if not isinstance(self.turns, list):
                raise TypeError("invalid protected turn journal")
            identities: set[str] = set()
            previous_end = 0
            deadline: float | None = None
            for index, turn in enumerate(self.turns):
                if (
                    not isinstance(turn, dict)
                    or not isinstance(turn.get("baseline"), int)
                    or isinstance(turn.get("baseline"), bool)
                ):
                    raise TypeError("invalid protected turn baseline")
                if (
                    turn["baseline"] != previous_end
                    or not 0 <= turn["baseline"] <= len(self.records)
                    or turn.get("id") in identities
                ):
                    raise ValueError("turn journal does not match evaluator ledger")
                if (
                    not isinstance(turn.get("token"), str)
                    or len(turn["token"]) < 32
                    or type(turn.get("closed")) is not bool
                    or (
                        turn.get("limit") is not None
                        and (type(turn["limit"]) is not int or turn["limit"] < 0)
                    )
                ):
                    raise ValueError("invalid protected turn admission")
                if not isinstance(turn.get("deadline_epoch"), (int, float)) or not math.isfinite(
                    turn["deadline_epoch"]
                ):
                    raise ValueError("invalid protected deadline")

                if turn["limit"] != 0:
                    if deadline is not None and deadline != turn["deadline_epoch"]:
                        raise ValueError("protected deadline changed")
                    deadline = turn["deadline_epoch"]
                elif deadline is not None and turn["deadline_epoch"] < deadline:
                    raise ValueError("protected deadline moved backwards")
                identities.add(turn["id"])
                previous_end = turn.get("end_sequence", len(self.records))
                if not turn["closed"] and index != len(self.turns) - 1:
                    raise ValueError("overlapping protected turns")
                if (
                    type(previous_end) is not int
                    or not turn["baseline"] <= previous_end <= len(self.records)
                    or (
                        turn["limit"] is not None
                        and previous_end - turn["baseline"] > turn["limit"]
                    )
                ):
                    raise ValueError("invalid protected turn count")

        def current(self) -> dict[str, Any]:
            if not self.turns:
                raise ValueError("no active turn")
            return self.turns[-1]

        def authorize(self) -> dict[str, Any]:
            turn = self.current()
            token = getattr(request_context, "token", "")
            if not token or not hmac.compare_digest(token, turn["token"]) or turn["closed"]:
                raise ValueError("inactive turn")
            if time.time() >= turn["deadline_epoch"]:
                raise ValueError("global deadline")
            return turn

        def open_turn(self, turn_id: str, limit: int | None, deadline: float) -> dict[str, Any]:
            with self.lock:
                if not turn_id or (limit is not None and (type(limit) is not int or limit < 0)):
                    raise ValueError("invalid turn contract")

                if limit == 0 and not self.turns:
                    raise ValueError("selection turn before any submission turn")
                if not math.isfinite(deadline) or deadline <= time.time():
                    raise ValueError("expired deadline")
                if any(t["id"] == turn_id for t in self.turns):
                    current = self.current()
                    if (
                        current["id"] != turn_id
                        or current["limit"] != limit
                        or current["deadline_epoch"] != deadline
                    ):
                        raise ValueError("turn identity/contract reused")
                    return dict(current)
                if self.turns and not self.current()["closed"]:
                    raise ValueError("previous turn has not closed")
                if self.turns:
                    budget = min(t["deadline_epoch"] for t in self.turns if t["limit"] != 0)

                    if limit != 0 and deadline != budget:
                        raise ValueError("global deadline cannot be reset")
                    if limit == 0 and deadline < budget:
                        raise ValueError("selection deadline precedes the budget")
                turn = {
                    "id": turn_id,
                    "token": secrets.token_urlsafe(32),
                    "baseline": len(self.records),
                    "limit": limit,
                    "deadline_epoch": deadline,
                    "closed": False,
                }
                self.turns.append(turn)
                atomic_json(control_file, self.turns)
                return dict(turn)

        def close_turn(self, turn_id: str) -> dict[str, Any]:
            with self.lock:
                turn = self.current()
                if turn["id"] != turn_id:
                    raise ValueError("wrong turn")
                turn["closed"] = True
                turn["end_sequence"] = len(self.records)
                atomic_json(control_file, self.turns)
                return self.turn_status()

        def turn_status(self) -> dict[str, Any]:

            turn = dict(self.current())
            count = len(self.records) - turn["baseline"]
            return {
                "id": turn["id"],
                "accepted": count,
                "limit": turn["limit"],
                "closed": turn["closed"],
                "exhausted": turn["limit"] is not None
                and turn["limit"] > 0
                and count >= turn["limit"],
                "selection": turn["limit"] == 0,
                "last_submission_id": self.records[-1]["submission_id"] if self.records else None,
            }

        def submit(self, artifact: Path, artifact_hash: str) -> dict[str, Any]:
            with self.lock:
                turn = self.authorize()
                if (
                    turn["limit"] is not None
                    and len(self.records) - turn["baseline"] >= turn["limit"]
                ):
                    raise OverflowError("submission not accepted")
                return super().submit(artifact, artifact_hash)

        def validate(self, artifact: Path) -> dict[str, Any]:
            with self.lock:
                self.authorize()
                return super().validate(artifact)

        def _record(self, result: dict[str, Any], artifact_hash: str) -> dict[str, Any]:

            if not getattr(request_context, "finalizing", False):
                self.authorize()
            return super()._record(result, artifact_hash)

        def finalize(self, submission_id: str) -> dict[str, Any]:
            "Re-state an already-accepted candidate as the last ledger record."
            with self.lock:
                if not self.turns or not self.current()["closed"]:
                    raise ValueError("finalize requires the last turn to be closed")
                if any(turn.get("finalized") for turn in self.turns):
                    raise ValueError("already finalized")
                source = next(
                    (r for r in self.records if r["submission_id"] == submission_id),
                    None,
                )
                if source is None:
                    raise ValueError("unknown submission")
                if source is self.records[-1]:
                    raise ValueError("nominee is already the last record")
                artifact_hash = str(source["artifact_sha256"])
                artifact = base._CANDIDATES_DIR / f"{artifact_hash}.csv"
                if artifact.is_symlink() or not artifact.is_file():
                    raise ValueError("nominated artifact is not retained")
                digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
                if not hmac.compare_digest(digest, artifact_hash):
                    raise ValueError("retained artifact does not match its digest")
                request_context.finalizing = True
                try:
                    result = self._score(artifact)
                    if not result["valid"]:
                        raise ValueError(str(result.get("error", "invalid nominee")))
                    score = result.get("raw_score")
                    if (
                        isinstance(score, bool)
                        or not isinstance(score, (int, float))
                        or not math.isfinite(float(score))
                    ):
                        raise ValueError("grader returned an invalid score")
                    record = self._record(result, artifact_hash)
                finally:
                    request_context.finalizing = False
                self._append(record)
                self._append_flowbench_score(record)

                self.turns[-1]["finalized"] = True
                atomic_json(control_file, self.turns)
                return {
                    "nominated_from": submission_id,
                    "artifact_sha256": artifact_hash,
                    **self.public_record(record),
                }

    class Handler(base.Handler):
        def do_POST(self) -> None:
            if self.path.startswith("/turn/"):
                if not hmac.compare_digest(self.headers.get("X-Control-Key", ""), admin_key):
                    self._send(HTTPStatus.FORBIDDEN, {"error": "control_forbidden"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 4096:
                        raise ValueError("invalid control body")
                    payload = json.loads(self.rfile.read(length))
                    state = base._STATE
                    if self.path == "/turn/open":
                        result = state.open_turn(
                            payload["id"], payload["limit"], payload["deadline_epoch"]
                        )
                    elif self.path == "/turn/close":
                        result = state.close_turn(payload["id"])
                    elif self.path == "/turn/status":
                        result = state.turn_status()
                    elif self.path == "/turn/finalize":
                        result = state.finalize(str(payload["submission_id"]))
                    else:
                        raise ValueError("unknown control path")
                    self._send(HTTPStatus.OK, result)
                except (ValueError, KeyError, TypeError) as error:
                    self._send(HTTPStatus.CONFLICT, {"error": str(error)})
                return
            request_context.token = self.headers.get("X-Turn-Token", "")
            try:
                base._STATE.authorize()
                super().do_POST()
            except ValueError:
                self._send(HTTPStatus.FORBIDDEN, {"error": "inactive_turn"})
            finally:
                request_context.token = ""

        def do_GET(self) -> None:
            if self.path == "/healthz":
                return super().do_GET()
            if self.path == "/scores":
                self._send(HTTPStatus.FORBIDDEN, {"error": "scores_withheld"})
                return
            request_context.token = self.headers.get("X-Turn-Token", "")
            try:
                base._STATE.authorize()
                if self.path == "/session/status":
                    current = base._STATE.turn_status()
                    self._send(
                        HTTPStatus.OK,
                        {key: current[key] for key in ("accepted", "closed", "exhausted")},
                    )
                    return
                super().do_GET()
            except ValueError:
                self._send(HTTPStatus.FORBIDDEN, {"error": "inactive_turn"})
            finally:
                request_context.token = ""

    State.request_context = request_context
    return State, Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--port", type=int, default=80)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("hma_native_evaluator", args.base)
    if spec is None or spec.loader is None:
        raise ValueError("native evaluator module unavailable")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    key = args.key_file.read_text().strip()
    if len(key) < 32:
        raise ValueError("control key too short")
    state_type, handler = guarded_types(base, args.control, key)
    base._STATE = state_type(base._CONFIG)
    ThreadingHTTPServer(("0.0.0.0", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
