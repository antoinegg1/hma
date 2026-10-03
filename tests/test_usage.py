"""Native response counting uses provider usage and durable response identities."""

import json

import pytest

from hma.analysis.events import native_responses
from hma.analysis.usage import response_usage


def write_log(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_claude_stream_updates_are_one_response_and_subagents_excluded(tmp_path):
    def row(tokens, identity="answer", model="claude-opus-5"):
        return {
            "type": "assistant",
            "timestamp": "2026-01-01T00:00:01+00:00",
            "message": {
                "id": identity,
                "model": model,
                "usage": {"output_tokens": tokens},
                "content": [{"type": "text", "text": "x"}],
            },
        }

    root = tmp_path / ".claude/projects/work"
    write_log(root / "main.jsonl", [row(1), row(50), row(50), row(999, "fake", "<synthetic>")])
    write_log(root / "main/subagents/agent-sub.jsonl", [row(999, "child")])
    events, _ = native_responses(tmp_path, "claude", "opus", 0, 2e9)
    assert [e["tokens"] for e in events] == [50]


def test_codex_repeated_cumulative_snapshots_and_equal_sized_responses(tmp_path):
    def row(total, last):
        return {
            "type": "event_msg",
            "timestamp": "2026-01-01T00:00:01+00:00",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {"output_tokens": total},
                    "last_token_usage": {"output_tokens": last},
                },
            },
        }

    path = write_log(
        tmp_path / ".codex/sessions/rollout-test.jsonl",
        [
            {"type": "session_meta", "payload": {"id": "root", "agent_path": "/root"}},
            row(100, 100),
            row(100, 100),
            row(200, 100),
        ],
    )
    events, issues = response_usage(path, "codex", 0, 2e9)
    assert not issues
    assert [e["tokens"] for e in events] == [100, 100]
    events, _ = native_responses(tmp_path, "codex", "gpt", 0, 2e9)
    assert len(events) == 2  # Responses without separate reasoning slices still count.


@pytest.mark.parametrize("cli", ["kimi", "dsh"])
def test_step_usage_not_tool_text(tmp_path, cli):
    if cli == "kimi":
        base = tmp_path / ".kimi/sessions/work/session_a"
        write_log(
            base / "agents/main/wire.jsonl",
            [
                {
                    "time": 1000,
                    "type": "context.append_loop_event",
                    "event": {"type": "step.begin", "uuid": "one"},
                },
                {
                    "time": 2000,
                    "type": "context.append_loop_event",
                    "event": {"type": "step.end", "uuid": "one", "usage": {"output": 25}},
                },
                {
                    "time": 2000,
                    "type": "context.append_loop_event",
                    "event": {"type": "step.end", "uuid": "one", "usage": {"output": 25}},
                },
                {
                    "time": 2500,
                    "type": "context.append_loop_event",
                    "event": {"type": "tool.result", "result": {"output": "x" * 1000}},
                },
            ],
        )
        (base / "state.json").write_text(json.dumps({"agents": {"main": {"homedir": "/missing"}}}))
    else:
        write_log(
            tmp_path / ".dsh/sessions/work/root/session.jsonl",
            [
                {"type": "session", "id": "root"},
                {"time": 1000, "type": "step/start", "data": {"turn": 1, "step": 1}},
                {
                    "time": 2000,
                    "type": "assistant/message",
                    "data": {"turn": 1, "step": 1, "usage": {"outputTokens": 25}, "message": {}},
                },
                {
                    "time": 2000,
                    "type": "assistant/message",
                    "data": {"turn": 1, "step": 1, "usage": {"outputTokens": 25}, "message": {}},
                },
                {"time": 2500, "type": "tool/result", "data": {"output": "x" * 1000}},
            ],
        )
    events, _ = native_responses(tmp_path, cli, "model", 0, 10)
    assert [e["tokens"] for e in events] == [25]
    assert events[0]["epoch"] == 2


def test_missing_usage_and_malformed_records_are_exposed(tmp_path):
    path = write_log(
        tmp_path / "log.jsonl",
        [{"type": "assistant", "timestamp": "2026-01-01T00:00:01+00:00", "message": {"id": "x"}}],
    )
    with path.open("a") as f:
        f.write("{broken\n")
    events, issues = response_usage(path, "claude", 0, 2e9)
    assert not events
    assert len(issues) == 2
