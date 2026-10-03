"""Fresh-run contracts and statistics: no external API or benchmark downloads."""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hma.analysis.events import ledger
from hma.analysis.report import report, trajectory
from hma.analysis.statistics import kl_lower, mean_se, prefix_retention, summaries
from hma.benchmark.integrity import atomic_json
from hma.repro.campaign import inventory
from hma.repro.config import TASKS, load_local, load_suite, plan
from hma.repro.harness_entry import latest_candidate
from hma.repro.providers import bootstrap, credentials
from hma.supervisor import Config, Supervisor

pytest_plugins = ["test_protocol"]


def test_full_plan_and_task_splits():
    suite = load_suite()
    planned = plan(suite, [], [], [])
    assert len(suite.experiments) == 27
    assert len(planned["cells"]) == 3397
    assert len({c["id"] for c in planned["cells"]}) == 3397
    assert len(TASKS) == 75
    assert [
        sum(t.suite == "mlebench_" + s for t in TASKS.values()) for s in ["lite", "medium", "high"]
    ] == [22, 38, 15]
    assert sum(c["experiment"].startswith("goal-") for c in planned["cells"]) == 1350


@pytest.mark.parametrize(
    "groups,tasks,repeats", [(["absent"], [], []), ([], ["bad-task"], []), (["goal-gpt"], [], [3])]
)
def test_plan_rejects_unknown_selectors(groups, tasks, repeats):
    with pytest.raises(ValueError):
        plan(load_suite(), groups, tasks, repeats)


def test_plan_subset_does_not_change_other_experiments():
    p = plan(load_suite(), ["goal-gpt"], ["leaf-classification"], [0])
    assert [c["id"] for c in p["cells"]] == ["goal-gpt/mbl_09/r0"]


def test_blank_credentials_work_for_planning_but_not_execution(monkeypatch):
    local = load_local(Path(__file__).parents[1] / "configs/local.json")
    assert all(not p.api_key and not p.base_url for p in local.providers.values())
    monkeypatch.delenv("HMA_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HMA_OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="intentionally blank"):
        credentials(local, {"openai"})


def test_provider_bootstrap_does_not_write_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HMA_PROVIDER", "openai")
    monkeypatch.setenv("HMA_OPENAI_API_KEY", "synthetic-secret")
    monkeypatch.setenv("HMA_OPENAI_BASE_URL", "https://example.invalid/v1")
    bootstrap()
    content = (tmp_path / ".codex/config.toml").read_text()
    assert "synthetic-secret" not in content
    assert 'wire_api = "responses"' in content


def test_goal_actor_enables_native_goals(monkeypatch):
    monkeypatch.setenv("HMA_WORKFLOW", "goal")
    import hma.actor_turn as turn

    turn = importlib.reload(turn)
    assert turn.TaskAgent.__metadata__[0].goals is True
    monkeypatch.setenv("HMA_WORKFLOW", "hma")
    turn = importlib.reload(turn)
    assert turn.TaskAgent.__metadata__[0].goals is False


def test_unlimited_gate_is_not_review_and_keeps_deadline(gate):
    deadline = time.time() + 100
    gate.open_turn("nta-0", None, deadline)
    gate.records.extend({"submission_id": str(i)} for i in range(100))
    assert gate.turn_status()["accepted"] == 100
    assert not gate.turn_status()["exhausted"]
    assert not gate.turn_status()["selection"]
    gate.close_turn("nta-0")
    with pytest.raises(ValueError, match="cannot be reset"):
        gate.open_turn("nta-1", None, deadline + 10)
    gate.open_turn("nta-1", None, deadline)
    assert gate.turn_status()["accepted"] == 0


@pytest.mark.parametrize("cap", [1, 3, 5])
def test_cap_only_counts_accepted(gate, cap):
    gate.open_turn("capped", cap, time.time() + 100)
    assert not gate.turn_status()["exhausted"]
    gate.records.extend({"submission_id": str(i)} for i in range(cap))
    assert gate.turn_status()["exhausted"]


def test_goal_runs_one_session_and_never_reviews(config, monkeypatch):
    fields = config.model_dump()
    fields.update(
        workflow="goal",
        actors=fields["actors"][:1],
        max_valid_submissions_per_session=None,
        review_reserve_seconds=0,
    )
    goal = Config.model_validate(fields)
    network = SimpleNamespace(name="test", remove=lambda: None)
    client = SimpleNamespace(networks=SimpleNamespace(create=lambda *a, **k: network))
    evaluator = SimpleNamespace(
        reload=lambda: None,
        attrs={"NetworkSettings": {"Networks": {"test": {"IPAddress": "127.0.0.1"}}}},
    )
    opens = []

    class Offline(Supervisor):
        def evaluator(self):
            return evaluator

        def control(self, endpoint, payload, **kwargs):
            if endpoint == "open":
                opens.append(payload)
                return {"token": "fake", "baseline": 0}
            if endpoint == "status":
                return {"exhausted": False}
            raise AssertionError(endpoint)

        def actor(self, *args, **kwargs):
            return SimpleNamespace(
                id="0",
                start=lambda: None,
                reload=lambda: None,
                attrs={"State": {"Running": False, "ExitCode": 0}},
                logs=lambda: b"",
            )

        def stop(self, container):
            pass

        def close_exploration_turn(self, *args):
            return {"accepted": 0, "last_submission_id": None}

        def review_phase(self, *args):
            raise AssertionError("goal must not review")

    result = Offline(goal, client).run()
    assert len(opens) == len(result["turns"]) == 1
    assert opens[0]["limit"] is None
    assert result["review"]["reason"] == "review_disabled"


def test_mean_standard_error_and_kl_bounds():
    avg, se = mean_se([0, 50, 100])
    assert avg == 50
    assert se == pytest.approx(50 / 3**0.5)
    assert mean_se([50]) == (50, None)
    assert 0 < kl_lower(0.9, 75) < 0.9
    assert kl_lower(0.9, 150) > kl_lower(0.9, 75)


def test_summary_keeps_failed_and_excluded_tasks():
    rows = [
        {
            "experiment": "test",
            "repeat": 0,
            "status": "complete",
            "split": "lite",
            "excluded": None,
            "final": {"medal": True},
        },
        {
            "experiment": "test",
            "repeat": 0,
            "status": "failed",
            "split": "lite",
            "excluded": None,
            "final": None,
        },
        {
            "experiment": "test",
            "repeat": 0,
            "status": "complete",
            "split": "lite",
            "excluded": {"reason": "contaminated"},
            "final": {"medal": True},
        },
    ]
    row = summaries(rows)[0]
    assert row["medal"] == pytest.approx(100 / 3)
    assert row["planned"] == 3
    assert row["medal_se"] is None


def test_latest_trajectory_allows_regression_and_review_override():
    run = {
        "run": "r",
        "experiment": "hma-x",
        "workflow": "hma",
        "seconds": 180,
        "excluded": None,
        "final": {"medal": True},
    }
    data = {
        "runs": [run],
        "responses": [],
        "submissions": [
            {"run": "r", "review": False, "seconds": 10, "medal": True},
            {"run": "r", "review": False, "seconds": 80, "medal": False},
        ],
    }
    curve = trajectory(data)
    assert [r["medal_rate"] for r in curve] == [0, 100, 0, 100]


def test_prefix_aggregates_within_task_and_excludes_zero_gain():
    opts = [
        {"experiment": "nta-x", "repeat": 0, "run": "r", "task": "t", "option": i} for i in range(2)
    ]
    subs = [
        {
            "experiment": "nta-x",
            "repeat": 0,
            "run": "r",
            "review": False,
            "option": o,
            "sequence": i,
            "quality": q,
        }
        for i, (o, q) in enumerate([(0, 0.2), (0, 0.6), (1, 0.5), (1, 1)], 1)
    ]
    data = {"experiments": {"nta-x": {"workflow": "nta"}}, "options": opts, "submissions": subs}
    result = prefix_retention(data)
    assert result[0]["gain_retention"] == pytest.approx(0.2)
    assert result[1]["gain_retention"] == 1
    assert result[0]["experiment_fraction"] == 0.5


def test_missing_final_selection_uses_timestamp_not_score(tmp_path):
    a = tmp_path / "runs/a/submission.csv"
    b = tmp_path / "runs/b/submission.csv"
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    a.write_text("first")
    b.write_text("second")
    import os

    os.utime(a, (10, 10))
    os.utime(b, (20, 20))
    assert latest_candidate(tmp_path, 15) == a
    assert latest_candidate(tmp_path, 25) == b
    outside = tmp_path / "outside.csv"
    outside.write_text("x")
    (tmp_path / "runs/link").symlink_to(outside.parent.parent, target_is_directory=True)
    assert latest_candidate(tmp_path, 25) == b


def test_ledger_rejects_mutated_record(tmp_path):
    path = tmp_path / "ledger.jsonl"
    record = {"sequence": 1, "previous_record_hash": "0" * 64}
    record["record_hash"] = hashlib.sha256(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(record) + "\n")
    assert len(ledger(path)) == 1
    record["sequence"] = 2
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="chain"):
        ledger(path)


def test_report_partial_plan_without_old_archives(tmp_path):
    p = plan(load_suite(), ["goal-gpt"], ["mbl_09"], [0])
    root = tmp_path / "run"
    atomic_json(root / "plan.json", p)
    with pytest.raises(ValueError, match="incomplete"):
        report(root, tmp_path / "strict")
    result = report(root, tmp_path / "partial", partial=True)
    assert result["runs"] == 1
    assert (tmp_path / "partial/table02-main.csv").exists()
    assert inventory(root)[0]["status"] == "pending"


def test_config_cannot_silently_use_cap_for_nta(config):
    fields = config.model_dump()
    fields.update(workflow="nta", review_reserve_seconds=0)
    with pytest.raises(ValueError, match="null cap"):
        Config.model_validate(fields)


def test_harness_smoke_reduces_research_only():
    from hma.repro.cli import parser, selected_plan

    args = parser().parse_args(
        ["smoke", "--experiment", "mlevolve_no_prior-ds4", "--seconds", "600"]
    )
    selected = selected_plan(args)
    e = next(e for e in selected["suite"]["experiments"] if e["id"] == "mlevolve_no_prior-ds4")
    assert e["seconds"] == 600 and e["research_seconds"] == 420 and e["postprocess_seconds"] == 0
    original = load_suite()
    assert next(e for e in original.experiments if e.id == "mlevolve_no_prior-ds4").seconds == 44280


def test_stage_roles_and_harness_contract_are_secret_free(tmp_path, monkeypatch):
    import hma.repro.campaign as campaign

    monkeypatch.setattr(campaign.os, "getuid", lambda: 1000)
    local = load_local(Path("configs/local.json"))
    local.data_root = tmp_path / "data"
    suite = load_suite()
    for group in ["goal-gpt", "nta-gpt-kimi", "hma-gpt-kimi", "mlevolve_no_prior-ds4"]:
        cell = plan(suite, [group], ["mbl_09"], [0])["cells"][0]
        spec = TASKS[cell["task"]]
        atomic_json(local.data_root / "manifests" / f"{spec.task_name}.json", {"synthetic": True})
        launch = campaign.stage_cell(cell, suite, local, tmp_path / group, "0")
        config = json.loads(launch.read_text())
        assert len(config["agent_data"]) == 1
        assert config["agent_data"][0]["target"].endswith("/input")
        assert "api_key" not in launch.read_text()
        assert not (launch.parent / "workspace-seed/AGENTS.md").exists()
        if group.startswith("mlevolve"):
            contract = json.loads(
                (launch.parent / "workspace-seed/.flowbench/paper.json").read_text()
            )
            assert (launch.parent / "workspace-seed/.flowbench/task.md").read_text() == (
                launch.parent / "task.md"
            ).read_text()
            assert contract["require_mechanisms"] is True
            assert contract["profile"].endswith("no_prior")
            assert contract["research_budget_sec"] == 43200
        else:
            assert config["max_valid_submissions_per_session"] == (
                5 if group.startswith("hma") else None
            )


def test_frozen_adapter_bytes():
    from hma.benchmark.integrity import sha256_file
    from hma.repro.campaign import baseline_root

    for directory in baseline_root().iterdir():
        lock = json.loads((directory / "source.lock.json").read_text())
        for name, digest in lock["runtime_files"].items():
            assert sha256_file(directory / "paper_runtime" / name) == digest


def test_nta_handoffs_are_natural_unlimited_and_have_fresh_homes(config, monkeypatch):
    import hma.supervisor as controller

    fields = config.model_dump()
    fields.update(workflow="nta", max_valid_submissions_per_session=None, review_reserve_seconds=0)
    nta = Config.model_validate(fields)
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(controller.time, "time", lambda: clock.now)
    network = SimpleNamespace(name="test", remove=lambda: None)
    client = SimpleNamespace(networks=SimpleNamespace(create=lambda *a, **k: network))
    evaluator = SimpleNamespace(
        reload=lambda: None,
        attrs={"NetworkSettings": {"Networks": {"test": {"IPAddress": "127.0.0.1"}}}},
    )
    homes, limits = [], []

    class Offline(Supervisor):
        def evaluator(self):
            return evaluator

        def control(self, endpoint, payload, **kwargs):
            if endpoint == "open":
                limits.append(payload["limit"])
                return {"token": "fixture", "baseline": 0}
            if endpoint == "status":
                return {"exhausted": False}
            raise AssertionError(endpoint)

        def actor(self, index, home, *args, **kwargs):
            homes.append(home)
            return SimpleNamespace(
                id=str(index),
                start=lambda: None,
                reload=lambda: None,
                attrs={"State": {"Running": False, "ExitCode": 0}},
                logs=lambda: b"",
            )

        def stop(self, container):
            pass

        def close_exploration_turn(self, *args):
            if len(homes) == 2:
                clock.now = self.explore_deadline
            return {"accepted": 7, "last_submission_id": "candidate"}

        def review_phase(self, *args):
            raise AssertionError("NTA cannot review")

    result = Offline(nta, client).run()
    assert limits == [None, None]
    assert homes[0] != homes[1]
    assert [t["actor"] for t in result["turns"]] == [0, 1]
    assert [t["reason"] for t in result["turns"]] == ["natural_exit", "natural_exit"]
    assert result["review"]["reason"] == "review_disabled"


def test_campaign_resume_never_retries_and_rejects_changed_environment(tmp_path, monkeypatch):
    import hma.benchmark.prepare_data as prepare
    import hma.repro.campaign as campaign

    monkeypatch.setattr(campaign.os, "getuid", lambda: 1000)
    monkeypatch.setattr(campaign, "credentials", lambda *a: {})
    hardware = {"gpus": [{"uuid": "GPU-fixture", "name": "NVIDIA A10"}]}
    monkeypatch.setattr(campaign, "inspect_hardware", lambda *a: hardware)
    monkeypatch.setattr(prepare, "_verify_frozen_task", lambda *a: None)
    image = SimpleNamespace(id="sha256:original")
    client = SimpleNamespace(ping=lambda: True, images=SimpleNamespace(get=lambda name: image))
    monkeypatch.setattr(campaign.docker, "from_env", lambda: client)
    local = load_local(Path("configs/local.json"))
    local.data_root = tmp_path / "data"
    selected = plan(load_suite(), ["goal-gpt"], ["mbl_09"], [0])
    task = selected["cells"][0]["task"]
    atomic_json(local.data_root / "manifests" / f"{task}.json", {"fixture": True})
    calls = []

    def failed_stage(cell, suite, local, directory, gpu):
        calls.append(cell["id"])
        raise RuntimeError("synthetic infrastructure failure")

    monkeypatch.setattr(campaign, "stage_cell", failed_stage)
    root = tmp_path / "campaign"
    assert campaign.run_plan(selected, local, root)[0]["status"] == "failed"
    assert json.loads((root / "environment.json").read_text())["hardware"] == hardware
    assert campaign.run_plan(selected, local, root, resume=True)[0]["status"] == "failed"
    assert len(calls) == 1
    hardware["gpus"][0]["uuid"] = "GPU-changed"
    with pytest.raises(ValueError, match="environment changed"):
        campaign.run_plan(selected, local, root, resume=True)
    hardware["gpus"][0]["uuid"] = "GPU-fixture"
    image.id = "sha256:changed"
    with pytest.raises(ValueError, match="environment changed"):
        campaign.run_plan(selected, local, root, resume=True)


@pytest.mark.parametrize(
    ("running", "code", "exhausted", "expected"),
    [
        (False, 0, True, "natural_exit"),
        (False, 75, True, "submission_cap"),
        (True, 0, True, "submission_cap"),
        (False, 75, False, "actor_error"),
        (False, 1, True, "actor_error"),
        (True, 0, False, None),
    ],
)
def test_natural_completion_wins_cap_ties_without_hiding_watchdog_or_errors(
    running, code, exhausted, expected
):
    from hma.supervisor import closure_reason

    assert closure_reason({"Running": running, "ExitCode": code}, exhausted) == expected


def test_watchdog_distinguishes_natural_completion_and_quota(monkeypatch):
    import hma.actor_entry as entry

    process = SimpleNamespace(poll=lambda: None, terminate=lambda: None, wait=lambda **kwargs: 0)
    monkeypatch.setattr(entry.subprocess, "Popen", lambda command: process)
    monkeypatch.setattr(entry, "status", lambda route: {"exhausted": True, "closed": False})
    assert (
        entry.supervise(["synthetic"], {"deadline_epoch": time.time() + 10})
        == entry.EXIT_SUBMISSION_CAP
    )
    process.poll = lambda: 0
    assert entry.supervise(["synthetic"], {"deadline_epoch": time.time() + 10}) == 0


def test_completed_new_run_normalizes_and_renders_case_without_historical_data(tmp_path):
    from datetime import datetime, timezone

    selected = plan(load_suite(), ["hma-gpt-opus"], ["mbh_07"], [0])
    root = tmp_path / "campaign"
    atomic_json(root / "plan.json", selected)
    cell = selected["cells"][0]
    execution = root / "cells" / cell["id"] / "execution"
    started = 1767225600
    atomic_json(
        execution / "deadline.json", {"started_epoch": started, "deadline_epoch": started + 21600}
    )
    atomic_json(execution / "result.json", {"status": "complete"})
    atomic_json(execution / "review.json", {"reason": "review_noop"})
    atomic_json(
        execution / "turn-history.json",
        [
            {
                "turn": 0,
                "actor": 0,
                "reason": "natural_exit",
                "accepted": 2,
                "started_epoch": started,
                "ended_epoch": started + 20,
            }
        ],
    )
    flowbench = execution / "evaluator/workspace/.flowbench"
    candidate_dir = flowbench / "mle-candidates"
    candidate_dir.mkdir(parents=True)
    previous = "0" * 64
    records = []
    for sequence in [1, 2]:
        content = f"id,value\n1,{sequence}\n".encode()
        digest = hashlib.sha256(content).hexdigest()
        (candidate_dir / f"{digest}.csv").write_bytes(content)
        row = {
            "sequence": sequence,
            "submission_id": f"candidate-{sequence}",
            "artifact_sha256": digest,
            "previous_record_hash": previous,
            "accepted_at_utc": datetime.fromtimestamp(
                started + sequence * 5, tz=timezone.utc
            ).isoformat(),
            "result": {
                "raw_score": sequence * 10,
                "leaderboard": {
                    "position": sequence * 50,
                    "total": 100,
                    "any_medal": sequence == 1,
                    "gold_medal": sequence == 1,
                    "above_median": sequence == 1,
                    "gold_threshold": 15,
                },
            },
        }
        row["record_hash"] = hashlib.sha256(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        records.append(row)
        previous = row["record_hash"]
    (flowbench / "mle-submissions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    log = execution / "turns/00000/agent/.codex/sessions/rollout-test.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "main"}})
        + "\n"
        + json.dumps(
            {
                "type": "event_msg",
                "timestamp": datetime.fromtimestamp(started + 1, tz=timezone.utc).isoformat(),
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {"output_tokens": 30},
                        "total_token_usage": {"output_tokens": 30},
                    },
                },
            }
        )
        + "\n"
    )
    output = tmp_path / "report"
    outcome = report(root, output)
    assert not outcome["issues"]
    events = json.loads((output / "events.json").read_text())
    assert events["runs"][0]["final"]["score"] == 20  # Later regression remains selected.
    assert events["responses"][0]["tokens"] == 30
    assert (output / "fig05.pdf").stat().st_size > 1000
    assert (output / "fig04b.svg").exists()
    assert not json.loads((output / "coverage.json").read_text())["full_paper_plan"]
