"Container-local fail-closed watchdog; no model work is performed here."

from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

EXIT_SUBMISSION_CAP = 75


def status(route: dict[str, Any]) -> dict[str, Any]:
    "Fetch this session's count only, never scores or protected admin state."
    connection = http.client.HTTPConnection(route["host"], route["port"], timeout=1)
    try:
        connection.request("GET", "/session/status", headers={"X-Turn-Token": route["token"]})
        response = connection.getresponse()
        payload = response.read(4096)
        if response.status != 200:
            raise RuntimeError("session gate unavailable or revoked")
        return json.loads(payload)
    finally:
        connection.close()


def supervise(command: list[str], route: dict[str, Any]) -> int:
    "Exit the container entry process on quota, timeout, or lost admission."
    remaining = route["deadline_epoch"] - time.time()
    if remaining <= 0:
        return 0
    stop_at = time.monotonic() + remaining
    process = subprocess.Popen(command)
    failure_since: float | None = None
    try:
        while time.monotonic() < stop_at:
            if (code := process.poll()) is not None:
                return code
            try:
                current = status(route)
                failure_since = None
                if current["exhausted"]:
                    return EXIT_SUBMISSION_CAP
                if current["closed"]:
                    return 0
            except (OSError, http.client.HTTPException, ValueError, RuntimeError):
                if failure_since is None:
                    failure_since = time.monotonic()
                elif time.monotonic() - failure_since >= 3:
                    return 70
            time.sleep(0.1)
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)


def main() -> None:
    command = sys.argv[1:]
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("missing actor command")
    from hma.repro.providers import bootstrap

    bootstrap()
    route = json.loads(Path("/run/hma/route.json").read_text())
    if token := os.environ.pop("CODEX_ACCESS_TOKEN", None):
        login = subprocess.run(
            ["codex", "login", "--with-access-token"],
            input=token + "\n",
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if login.returncode:
            raise RuntimeError("Codex authentication bootstrap failed")
    proxy = None
    if os.environ.get("KIMI_MODEL_API_KEY"):
        from provider_proxy import _PLACEHOLDER_KEY, _Proxy

        proxy = _Proxy(os.environ["KIMI_MODEL_BASE_URL"], os.environ["KIMI_MODEL_API_KEY"])
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        os.environ["KIMI_MODEL_API_KEY"] = _PLACEHOLDER_KEY
        os.environ["KIMI_MODEL_BASE_URL"] = f"http://127.0.0.1:{proxy.server_address[1]}"
    try:
        raise SystemExit(supervise(command, route))
    finally:
        if proxy:
            proxy.shutdown()
            proxy.server_close()


if __name__ == "__main__":
    main()
