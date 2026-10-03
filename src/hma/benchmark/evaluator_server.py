"Restart-safe local evaluator for MLE-bench and MLE-Dojo profiles."

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import uuid
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_FLOWBENCH_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _FLOWBENCH_DIR / "mle-config.json"
_WORKER_PATH = _FLOWBENCH_DIR / "mle-score-worker.py"
_LEDGER_PATH = _FLOWBENCH_DIR / "mle-submissions.jsonl"
_SCORES_PATH = _FLOWBENCH_DIR / "scores.jsonl"
_CANDIDATES_DIR = _FLOWBENCH_DIR / "mle-candidates"
_MAX_JSON_BYTES = 4 * 1024 * 1024
_CHUNK_BYTES = 1024 * 1024
_CHAIN_GENESIS = "0" * 64


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _record_hash(record: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "record_hash"}
    payload = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_config() -> dict[str, Any]:
    config = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "experiment_id",
        "suite",
        "slug",
        "higher_is_better",
        "feedback_mode",
        "submission_limit",
        "max_artifact_bytes",
        "score_timeout_seconds",
        "upstream_path",
        "dataset_path",
        "control_path",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("MLE evaluator config violates its schema")
    if config["schema_version"] != 1 or config["feedback_mode"] not in {
        "blind",
        "interactive",
    }:
        raise ValueError("MLE evaluator config has an invalid contract")
    submission_limit = config["submission_limit"]
    if submission_limit is not None and (
        isinstance(submission_limit, bool)
        or not isinstance(submission_limit, int)
        or not 1 <= submission_limit <= 100
    ):
        raise ValueError("MLE evaluator submission limit is invalid")
    return config


class State:
    "Owns the immutable submission ledger and serialized grader calls."

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.lock = threading.RLock()
        self.records, self.head = self._restore()
        _CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    def _restore(self) -> tuple[list[dict[str, Any]], str]:
        records: list[dict[str, Any]] = []
        head = _CHAIN_GENESIS
        submission_ids: set[str] = set()
        previous_time: datetime | None = None
        try:
            lines = _LEDGER_PATH.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return records, head
        for line_number, line in enumerate(lines, start=1):
            record = json.loads(line)
            result = record.get("result") if isinstance(record, dict) else None
            submission_id = record.get("submission_id") if isinstance(record, dict) else None
            artifact_hash = record.get("artifact_sha256") if isinstance(record, dict) else None
            accepted_at = record.get("accepted_at_utc") if isinstance(record, dict) else None
            try:
                raw_score = result["raw_score"]
                score = float(raw_score)
                accepted_time = datetime.fromisoformat(accepted_at)
                accepted_time = (
                    accepted_time.replace(tzinfo=UTC)
                    if accepted_time.tzinfo is None
                    else accepted_time.astimezone(UTC)
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                raw_score = None
                score = math.nan
                accepted_time = None
            if (
                not isinstance(record, dict)
                or record.get("schema_version") != 1
                or record.get("previous_record_hash") != head
                or record.get("record_hash") != _record_hash(record)
                or record.get("experiment_id") != self.config["experiment_id"]
                or record.get("suite") != self.config["suite"]
                or record.get("sequence") != line_number
                or not isinstance(result, dict)
                or result.get("valid") is not True
                or isinstance(raw_score, bool)
                or not math.isfinite(score)
                or not isinstance(submission_id, str)
                or not submission_id
                or submission_id in submission_ids
                or not isinstance(artifact_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", artifact_hash) is None
                or accepted_time is None
                or (previous_time is not None and accepted_time < previous_time)
            ):
                raise RuntimeError(f"MLE ledger chain breaks at line {line_number}")
            records.append(record)
            submission_ids.add(submission_id)
            previous_time = accepted_time
            head = str(record["record_hash"])
        submission_limit = self.config["submission_limit"]
        if submission_limit is not None and len(records) > submission_limit:
            raise RuntimeError("MLE ledger exceeds the submission limit")
        return records, head

    def _score(self, artifact: Path) -> dict[str, Any]:
        completed = subprocess.run(
            [
                "python3",
                str(_WORKER_PATH),
                "--config",
                str(_CONFIG_PATH),
                "--submission",
                str(artifact),
            ],
            cwd=_FLOWBENCH_DIR.parent,
            capture_output=True,
            text=True,
            check=False,
            timeout=int(self.config["score_timeout_seconds"]),
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("grader returned no structured result")
        result = json.loads(lines[-1])
        if not isinstance(result, dict) or not isinstance(result.get("valid"), bool):
            raise TypeError("grader returned an invalid structured result")
        score = result.get("raw_score")
        if result["valid"] and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError("grader returned an invalid score")
        return result

    def validate(self, artifact: Path) -> dict[str, Any]:
        with self.lock:
            result = self._score(artifact)
        if result["valid"]:
            return {"valid": True}
        response: dict[str, Any] = {"valid": False, "error": "invalid_submission"}
        if self.config["feedback_mode"] == "interactive":
            response["detail"] = str(result.get("error", "validation failed"))[:512]
        return response

    def submit(self, artifact: Path, artifact_hash: str) -> dict[str, Any]:
        with self.lock:
            submission_limit = self.config["submission_limit"]
            if submission_limit is not None and len(self.records) >= submission_limit:
                raise OverflowError("submission limit exhausted")
            if any(record["artifact_sha256"] == artifact_hash for record in self.records):
                raise FileExistsError("identical artifact already submitted")
            result = self._score(artifact)
            if not result["valid"]:
                raise ValueError(str(result.get("error", "invalid submission")))
            score = result.get("raw_score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise ValueError("grader returned an invalid score")
            destination = _CANDIDATES_DIR / f"{artifact_hash}.csv"
            os.replace(artifact, destination)
            record = self._record(result, artifact_hash)
            self._append(record)
            self._append_flowbench_score(record)
            return self.public_record(record)

    def _record(self, result: dict[str, Any], artifact_hash: str) -> dict[str, Any]:
        record = {
            "schema_version": 1,
            "experiment_id": self.config["experiment_id"],
            "suite": self.config["suite"],
            "submission_id": uuid.uuid4().hex,
            "sequence": len(self.records) + 1,
            "accepted_at_utc": _utc_now(),
            "artifact_sha256": artifact_hash,
            "result": result,
            "previous_record_hash": self.head,
        }
        record["record_hash"] = _record_hash(record)
        return record

    def _append(self, record: dict[str, Any]) -> None:
        with _LEDGER_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            file.flush()
            os.fsync(file.fileno())
        self.records.append(record)
        self.head = str(record["record_hash"])

    def _append_flowbench_score(self, record: dict[str, Any]) -> None:
        score = float(record["result"]["raw_score"])
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        payload = {
            "datetime": record["accepted_at_utc"],
            "score": score,
            "metadata": {
                "experiment_id": self.config["experiment_id"],
                "submission_id": record["submission_id"],
                "artifact_sha256": record["artifact_sha256"],
                "feedback_mode": self.config["feedback_mode"],
            },
        }
        with _SCORES_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            file.flush()
            os.fsync(file.fileno())

    def public_record(self, record: dict[str, Any]) -> dict[str, Any]:
        submission_limit = self.config["submission_limit"]
        response = {
            "submission_id": record["submission_id"],
            "sequence": record["sequence"],
            "accepted_at_utc": record["accepted_at_utc"],
            "artifact_sha256": record["artifact_sha256"],
            "status": "accepted",
            "remaining_submissions": (
                None if submission_limit is None else submission_limit - len(self.records)
            ),
        }
        if self.config["feedback_mode"] == "interactive":
            response.update(record["result"])
        return response

    def find(self, submission_id: str) -> dict[str, Any] | None:
        with self.lock:
            for record in self.records:
                if record["submission_id"] == submission_id:
                    return self.public_record(record)
        return None

    def scores(self) -> list[dict[str, Any]]:
        with self.lock:
            return [self.public_record(record) for record in self.records]


_CONFIG = _load_config()
_STATE = State(_CONFIG)


class Handler(BaseHTTPRequestHandler):
    "Serves the minimal evaluator API on the private cell network."

    timeout = 60

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _content_length(self) -> int | None:
        values = self.headers.get_all("Content-Length")
        if values is None or len(values) != 1:
            self._send(HTTPStatus.LENGTH_REQUIRED, {"error": "content_length_required"})
            return None
        try:
            length = int(values[0])
        except ValueError:
            length = -1
        maximum = int(_CONFIG["max_artifact_bytes"])
        if length <= 0 or length > maximum:
            self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "invalid_body_size"})
            return None
        return length

    def _spool(self, length: int) -> tuple[Path, str] | None:
        descriptor, name = tempfile.mkstemp(prefix="mle-candidate-", dir=_FLOWBENCH_DIR)
        path = Path(name)
        digest = hashlib.sha256()
        try:
            with os.fdopen(descriptor, "wb") as file:
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise TimeoutError
                    file.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
        except (OSError, TimeoutError):
            path.unlink(missing_ok=True)
            self._send(HTTPStatus.REQUEST_TIMEOUT, {"error": "body_timeout"})
            return None
        return path, digest.hexdigest()

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send(HTTPStatus.OK, {"status": "ok"})
            return
        if self.path == "/scores":
            if _CONFIG["feedback_mode"] == "blind":
                self._send(HTTPStatus.FORBIDDEN, {"error": "scores_withheld"})
            else:
                self._send(HTTPStatus.OK, _STATE.scores())
            return
        prefix = "/submissions/"
        if self.path.startswith(prefix):
            record = _STATE.find(self.path.removeprefix(prefix))
            self._send(
                HTTPStatus.OK if record is not None else HTTPStatus.NOT_FOUND,
                record if record is not None else {"error": "submission_not_found"},
            )
            return
        self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path not in {"/validate", "/submit", "/candidates"}:
            self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        length = self._content_length()
        if length is None:
            return
        spooled = self._spool(length)
        if spooled is None:
            return
        artifact, artifact_hash = spooled
        try:
            if self.path == "/validate":
                response = _STATE.validate(artifact)
                status = HTTPStatus.OK if response["valid"] else HTTPStatus.BAD_REQUEST
            else:
                response = _STATE.submit(artifact, artifact_hash)
                status = HTTPStatus.OK
        except OverflowError:
            response, status = (
                {"error": "submission_limit_exhausted"},
                HTTPStatus.CONFLICT,
            )
        except FileExistsError:
            response, status = {"error": "duplicate_artifact"}, HTTPStatus.CONFLICT
        except (
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            response, status = {"error": "evaluation_failed"}, HTTPStatus.BAD_REQUEST
        finally:
            artifact.unlink(missing_ok=True)
        self._send(status, response)


def serve() -> None:
    "Runs the evaluator until its cell is stopped."
    ThreadingHTTPServer(("0.0.0.0", 80), Handler).serve_forever()


if __name__ == "__main__":
    serve()
