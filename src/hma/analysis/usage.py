"""Response-level usage from pinned native log formats; never estimate tokens."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def response_usage(path: Path, cli: str, start: float, end: float) -> tuple[list[dict], list[str]]:
    events: dict[str, dict] = {}
    issues = []
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            issues.append(f"malformed JSON at line {line_number}")
            continue
        if not isinstance(row, dict):
            continue
        kind = row.get("type")
        usage, identity, stamp, actual = None, None, None, None
        if cli == "claude" and kind == "assistant":
            message = row.get("message") or {}
            actual = message.get("model")
            if actual and actual.startswith("<"):
                continue
            usage = message.get("usage")
            identity = message.get("id") or row.get("requestId")
            stamp = row.get("timestamp")
        elif cli == "codex":
            payload = row.get("payload") or {}
            if payload.get("type") != "token_count" or not payload.get("info"):
                continue
            info = payload["info"]
            usage = info.get("last_token_usage")
            # Cumulative usage identifies repeated snapshots; use per-response output.
            identity = json.dumps(
                info.get("total_token_usage") or [row.get("timestamp"), usage], sort_keys=True
            )
            stamp = row.get("timestamp")
        elif cli == "dsh" and kind == "assistant/message":
            data = row.get("data") or {}
            usage = data.get("usage")
            identity = json.dumps([data.get("turn"), data.get("step")])
            stamp = row.get("time")
            actual = ((data.get("message") or {}).get("source") or {}).get("model")
        elif cli == "kimi" and kind == "context.append_loop_event":
            event = row.get("event") or {}
            if event.get("type") != "step.end":
                continue
            usage = event.get("usage")
            identity = event.get("uuid")
            stamp = row.get("time")
        else:
            continue
        try:
            epoch = (
                datetime.fromisoformat(stamp).timestamp()
                if isinstance(stamp, str)
                else float(stamp) / 1000
            )
        except (ValueError, TypeError):
            issues.append(f"missing response timestamp at line {line_number}")
            continue
        if not start <= epoch <= end:
            continue
        key = {
            "claude": "output_tokens",
            "codex": "output_tokens",
            "dsh": "outputTokens",
            "kimi": "output",
        }[cli]
        tokens = usage.get(key) if isinstance(usage, dict) else None
        if type(tokens) is not int or tokens < 0 or identity is None:
            issues.append(f"missing response identity or output usage at line {line_number}")
            continue
        identity = str(identity)
        # Claude may stream several blocks of a message with increasing usage.
        old = events.get(identity)
        if old is None or (cli == "claude" and tokens > old["tokens"]):
            events[identity] = {
                "response_id": identity,
                "epoch": epoch,
                "tokens": tokens,
                "reported_model": actual,
            }
    return sorted(events.values(), key=lambda e: e["epoch"]), issues
