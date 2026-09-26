"""Offline checks with temporary synthetic inputs; no experimental data."""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import hma.supervisor as controller
from hma.benchmark.stage import stage
from hma.evaluator import guarded_types
from hma.supervisor import Config, Supervisor, prompt_for, review_prompt_for


@pytest.fixture
def config(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    task = tmp_path / "task.md"
    task.write_text("Perform experiments within a 6-hour budget.")
    return Config(
        root=tmp_path / "runs/new/task",
        task_file=task,
        evaluator_seed_home=evaluator,
        agent_image="mock-agent",
        evaluator_image="mock-evaluator",
        actors=({"spec": "actor-a"}, {"spec": "actor-b"}),
        agent_data=({"source": public, "target": "/home/user/.flowbench-data/input"},),
        evaluator_data=(),
        uid=1000, gid=1000, gpu=None,
    )


def test_prompt_keeps_cap_undisclosed_and_review_wall_explicit():
    task = "Use a 6-hour budget. There is no medal-based stopping condition and no submission quota."
    rendered = prompt_for(task, 21600)
    assert "no submission quota" not in rendered.lower()
    assert "cap" not in rendered.lower()
    assert "5 accepted" not in rendered
    assert "no medal-based stopping condition" in rendered
    review = review_prompt_for(rendered, 600)
    assert "600 seconds" in review
    assert "this session cannot\nsubmit" in review
    with pytest.raises(ValueError):
        review_prompt_for(task, 600)


def test_reserve_must_fit_review_and_teardown(config):
    fields = config.model_dump()
    fields["review_reserve_seconds"] = 650
    with pytest.raises(ValueError, match="teardown"):
        Config.model_validate(fields)


def test_private_mount_refused_for_actor(config, tmp_path):
    fields = config.model_dump()
    fields["agent_data"].append({"source": tmp_path, "target": "/home/user/.flowbench-data/private"})
    with pytest.raises(ValueError, match="public input only"):
        Config.model_validate(fields)


@pytest.fixture
def gate(tmp_path):
    base = ModuleType("synthetic_evaluator")

    class BaseState:
        def __init__(self, config):
            self.config = config
            self.records = []
            self.lock = threading.RLock()

    base.State = BaseState
    base.Handler = type("BaseHandler", (), {})
    cls, _ = guarded_types(base, tmp_path / "turns.json", "temporary-test-control")
    return cls({"feedback_mode": "blind", "submission_limit": None})


def test_gate_counts_accepted_records_and_keeps_review_alive(gate):
    deadline = time.time() + 100
    gate.open_turn("first", 5, deadline)
    gate.records.extend({"submission_id": str(i)} for i in range(4))
    assert gate.turn_status()["accepted"] == 4
    assert not gate.turn_status()["exhausted"]
    gate.records.append({"submission_id": "last"})
    assert gate.turn_status()["exhausted"]
    gate.close_turn("first")
    gate.open_turn("review", 0, deadline + 20)
    status = gate.turn_status()
    assert status["selection"] and not status["exhausted"]


def test_gate_refuses_overlap_and_deadline_extension(gate):
    deadline = time.time() + 100
    gate.open_turn("first", 5, deadline)
    with pytest.raises(ValueError, match="previous turn"):
        gate.open_turn("overlap", 5, deadline)
    gate.close_turn("first")
    with pytest.raises(ValueError, match="cannot be reset"):
        gate.open_turn("extended", 5, deadline + 1)
    gate.open_turn("second", 5, deadline)
    assert gate.turn_status()["accepted"] == 0


def test_review_accepts_only_known_candidates_and_silent_fallback(tmp_path):
    records = [{"submission_id": "a"}, {"submission_id": "b"}]
    assert Supervisor.read_nomination(tmp_path, records) is None
    nomination = tmp_path / "nomination.json"
    nomination.write_text('{"nominate":"unknown"}')
    assert Supervisor.read_nomination(tmp_path, records) is None
    nomination.write_text('{"nominate":"a"}')
    assert Supervisor.read_nomination(tmp_path, records) == "a"
    nomination.write_text("invalid JSON")
    assert Supervisor.read_nomination(tmp_path, records) is None


def test_review_author_uses_actual_turn_counts():
    records = [{"submission_id": key} for key in ["a", "b", "c"]]
    turns = [
        {"actor": 0, "accepted": 2, "last_submission_id": "b"},
        {"actor": 1, "accepted": 1, "last_submission_id": "c"},
    ]
    assert Supervisor.standing_author(turns, records, "c") == 1
    assert Supervisor.standing_author(turns, records, "b") == 0
    del turns[0]["accepted"]
    assert Supervisor.standing_author(turns, records, "c") is None


def test_fresh_homes_share_exactly_one_workspace(config, monkeypatch, tmp_path):
    made = []
    def create(*args, **kwargs):
        made.append(kwargs)
        return SimpleNamespace()
    client = SimpleNamespace(containers=SimpleNamespace(create=create))
    supervisor = Supervisor(config, client)
    supervisor.network = SimpleNamespace(name="test-network")
    supervisor.runtime.mkdir(parents=True)
    for name in ["actor_turn.py", "actor_entry.py", "submit_client.py"]:
        shutil.copyfile(controller.PACKAGE / name, supervisor.runtime / name)
    monkeypatch.setattr(controller.os, "chown", lambda *a: None)
    monkeypatch.setattr(controller.os, "lchown", lambda *a: None)
    monkeypatch.setattr(supervisor, "inject", lambda *a, **k: None)
    route = tmp_path / "route.json"
    route.write_text("{}")
    for index in range(2):
        home = config.root / "turns" / str(index) / "agent"
        home.mkdir(parents=True)
        supervisor.actor(index, home, route, "synthetic prompt")
    def mounted(call, target):
        return next(source for source, value in call["volumes"].items() if value["bind"] == target)
    assert mounted(made[0], "/home/user") != mounted(made[1], "/home/user")
    assert mounted(made[0], "/home/user/workspace") == mounted(made[1], "/home/user/workspace")
    assert "actor-a" in made[0]["command"] and "actor-b" in made[1]["command"]
    assert all("docker.sock" not in key for call in made for key in call["volumes"])


def test_actual_run_loop_handles_cap_natural_return_and_review(config, monkeypatch):
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(controller.time, "time", lambda: clock.now)
    network = SimpleNamespace(name="test-network", remove=lambda: None)
    client = SimpleNamespace(networks=SimpleNamespace(create=lambda *a, **k: network))
    active = SimpleNamespace(index=-1)
    events = []
    evaluator = SimpleNamespace(
        reload=lambda: None,
        attrs={"NetworkSettings": {"Networks": {"test-network": {"IPAddress": "127.0.0.1"}}}},
    )

    class OfflineSupervisor(Supervisor):
        def evaluator(self): return evaluator
        def control(self, endpoint, payload, **kwargs):
            if endpoint == "open":
                active.index += 1
                events.append((endpoint, payload["deadline_epoch"], payload["limit"]))
                return {"token": "synthetic-token", "baseline": 0}
            if endpoint == "status":
                return {"exhausted": active.index == 0}
            raise AssertionError(endpoint)
        def actor(self, index, home, route, prompt, **kwargs):
            return SimpleNamespace(
                id=str(index), start=lambda: None, reload=lambda: None,
                attrs={"State": {"Running": False, "ExitCode": 0}}, logs=lambda: b"",
            )
        def stop(self, container): pass
        def close_exploration_turn(self, turn_id, opened):
            if active.index == 1: clock.now = self.explore_deadline
            return {"accepted": 5 if active.index == 0 else 1, "last_submission_id": str(active.index)}
        def review_phase(self, prompt, turns):
            assert clock.now == self.explore_deadline
            assert self.review_deadline == self.deadline
            assert self.deadline - self.explore_deadline == 900
            events.append(("review",))
            return {"reason": "review_noop"}

    result = OfflineSupervisor(config, client).run()
    assert result["status"] == "complete"
    assert [turn["actor"] for turn in result["turns"]] == [0, 1]
    assert [turn["reason"] for turn in result["turns"]] == ["submission_cap", "natural_exit"]
    assert events[0][1:] == events[1][1:]
    assert events[-1] == ("review",)


def test_staging_has_no_embedded_data_and_refuses_overwrite(tmp_path):
    template = Path(__file__).parents[1] / "configs/hma-opus-gpt.json"
    launch = json.loads(template.read_text())
    home = tmp_path / "evaluator"
    launch["evaluator_seed_home"] = str(home)
    launch["root"] = str(tmp_path / "runs/new/task")
    launch["task_file"] = str(tmp_path / "task.md")
    Path(launch["task_file"]).write_text("Synthetic task prompt.")
    for mount in launch["agent_data"] + launch["evaluator_data"]:
        source = tmp_path / Path(mount["source"]).name
        source.mkdir(exist_ok=True)
        mount["source"] = str(source)
    for actor in launch["actors"]:
        actor["seed_files"] = []
    config_file = tmp_path / "launch.json"
    config_file.write_text(json.dumps(launch))
    target = stage(config_file)
    assert sorted(p.name for p in target.iterdir()) == ["mle-config.json", "mle-evaluator-server.py", "mle-score-worker.py"]
    payload = json.loads((target / "mle-config.json").read_text())
    assert payload["feedback_mode"] == "blind"
    parsed = Config.model_validate_json(config_file.read_text())
    assert parsed.evaluator == payload
    assert parsed.evaluator_seed_home == home
    with pytest.raises(ValueError, match="new absolute directory"):
        stage(config_file)


def test_telemetry_cannot_be_enabled(monkeypatch):
    from hmz import telemetry
    monkeypatch.setenv("HUMANIZE_SENTRY", "on")
    assert telemetry.enabled() is False
    assert telemetry.DSN == ""
