'Run one completion-bounded Humanize Goal through Kimi Code on a provider gateway.'

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_SEMANTIC_FAILURE_EXIT_CODE = 65
_PLACEHOLDER_KEY = "flowbench-loopback-proxy-credential"

_UNATTENDED_PERMISSION = "auto"
_HOP_BY_HOP = frozenset(
    {
        "accept-encoding",
        "authorization",
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")

def _nonempty_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be nonempty")
    return value

class _Proxy(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, upstream: str, api_key: str) -> None:
        super().__init__(("127.0.0.1", 0), _ProxyHandler)
        self.upstream = upstream.rstrip("/")
        self.api_key = api_key
        self.requests = 0
        self._lock = threading.Lock()

    def counted(self) -> None:
        with self._lock:
            self.requests += 1

class _ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server: _Proxy

    def log_message(self, _format: str, *args: object) -> None:
        del args

    def do_GET(self) -> None:
        self._forward(None)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self._forward(self.rfile.read(length))

    def _forward(self, body: bytes | None) -> None:
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in _HOP_BY_HOP
        }
        headers["Authorization"] = f"Bearer {self.server.api_key}"
        request = urllib.request.Request(
            self.server.upstream + self.path,
            data=body,
            headers=headers,
            method=self.command,
        )
        try:
            response = urllib.request.urlopen(request, timeout=360)
        except urllib.error.HTTPError as error:
            response = error
        except (OSError, urllib.error.URLError):
            payload = b'{"error":{"code":"HMA_PROVIDER_PROXY"}}\n'
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.server.counted()
        with response:
            self.send_response(response.status)
            for name in ("Content-Type", "x-request-id"):
                if value := response.headers.get(name):
                    self.send_header(name, value)
            self.send_header("Connection", "close")
            self.end_headers()
            while chunk := response.read(64 * 1024):
                self.wfile.write(chunk)
                self.wfile.flush()

def _submission_present(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    except OSError:
        return False

def main() -> int:
    from hmz.agents.kimi import KimiCodeCLIAgent, KimiCodeCLIAgentConfig

    objective = Path(".flowbench/task.md").read_text(encoding="utf-8").strip()
    upstream = _nonempty_environment("KIMI_MODEL_BASE_URL")
    api_key = _nonempty_environment("KIMI_MODEL_API_KEY")
    started_at = _utc_now()
    started = time.monotonic()
    wrapper_exit_code = _SEMANTIC_FAILURE_EXIT_CODE
    error_type: str | None = None
    session_id: str | None = None
    goal: Mapping[str, Any] | None = None
    proxy = _Proxy(upstream, api_key)
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    session = None
    agent = None

    def observe(_agent: object, _session: object, event: object) -> None:
        kind = str(getattr(event, "kind", ""))
        text = str(getattr(event, "text", ""))
        destination = (
            sys.stderr if kind in {"reasoning", "tool", "failed"} else sys.stdout
        )
        print(text, file=destination, flush=True)

    try:
        os.environ["KIMI_MODEL_API_KEY"] = _PLACEHOLDER_KEY
        os.environ["KIMI_MODEL_BASE_URL"] = (
            f"http://127.0.0.1:{proxy.server_address[1]}"
        )
        agent = KimiCodeCLIAgent(
            KimiCodeCLIAgentConfig(
                model=os.environ["MODEL"],
                effort=os.environ["EFFORT"],
                permission=_UNATTENDED_PERMISSION,
            )
        )
        agent.watch(observe)
        session = agent.new(Path.cwd())
        session.pursue(objective)
        session_id = session.named
        if session_id is not None:
            value = agent.server.call("GET", f"/sessions/{session_id}/goal")
            if isinstance(value, dict):
                goal = value
        submission_ready = _submission_present(Path("solution/submission.csv"))
        goal_terminal = goal is None or goal.get("status") != "active"
        wrapper_exit_code = 0 if goal_terminal and submission_ready else 65
        return wrapper_exit_code
    except BaseException as error:
        error_type = type(error).__name__
        raise
    finally:
        if session is not None:
            session.close()
        if agent is not None:
            agent.stop()
        proxy.shutdown()
        proxy.server_close()
        proxy_thread.join(timeout=5)
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete" if wrapper_exit_code == 0 else "error",
            "started_at_utc": started_at,
            "terminal_at_utc": _utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "session_id": session_id,
            "goal_status": None if goal is None else goal.get("status"),
            "submission_present": _submission_present(Path("solution/submission.csv")),
            "gateway_requests": proxy.requests,
            "wrapper_exit_code": wrapper_exit_code,
        }
        if error_type is not None:
            receipt["error_type"] = error_type
        path = Path(".flowbench/goal-terminal.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

if __name__ == "__main__":
    raise SystemExit(main())
