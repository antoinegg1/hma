"""Offline contracts for dedicated-node Swarm execution; no Docker or model calls."""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hma.benchmark.integrity import atomic_json
from hma.repro import swarm
from hma.repro.config import load_local, load_suite, plan


def pinned_local(tmp_path):
    local = load_local(Path("configs/local.example.json"))
    local.data_root = tmp_path / "shared/data"
    local.agent_image = "registry.test/hma/agent@sha256:" + "a" * 64
    local.evaluator_image = "registry.test/hma/evaluator@sha256:" + "b" * 64
    local.harness_images = {
        name: f"registry.test/hma/{name}@sha256:" + "c" * 64 for name in local.harness_images
    }
    return local


def nodes(count):
    return [
        {
            "ID": f"node-{i:02}",
            "Status": {"State": "ready"},
            "Spec": {"Labels": {"hma.worker": "true"}, "Availability": "active"},
            "Description": {
                "Hostname": f"host-{i}",
                "Platform": {"OS": "linux", "Architecture": "x86_64"},
                "Resources": {"NanoCPUs": 32 * 10**9, "MemoryBytes": 256 * 1024**3},
            },
        }
        for i in range(count)
    ]


@pytest.fixture
def cluster(tmp_path):
    local = pinned_local(tmp_path)
    root = tmp_path / "shared/runs/paper"
    selected = plan(load_suite(), ["goal-gpt"], ["mbl_01", "mbl_09"], [0])
    manifest = swarm.create_manifest(selected, local, nodes(2), root, tmp_path / "shared", 2)
    atomic_json(root / "cluster.json", manifest)
    return root, local, manifest


def test_full_matrix_partition_is_complete_disjoint_and_secret_free(tmp_path):
    local = pinned_local(tmp_path)
    local.providers["openai"].api_key = "synthetic-private-key"
    selected = plan(load_suite(), [], [], [])
    manifest = swarm.create_manifest(
        selected, local, nodes(75), tmp_path / "shared/runs/paper", tmp_path / "shared"
    )
    ids = [
        cell["id"]
        for assignment in manifest["assignments"]
        for cell in swarm.shard_plan(selected, assignment["task"])["cells"]
    ]
    assert len(ids) == len(set(ids)) == 3397
    assert set(ids) == {c["id"] for c in selected["cells"]}
    assert len(manifest["assignments"]) == 75
    assert "synthetic-private-key" not in json.dumps(manifest)
    assert "providers" not in manifest["settings"]
    assert (
        manifest["assignments"]
        == swarm.create_manifest(
            selected,
            local,
            list(reversed(nodes(75))),
            tmp_path / "shared/runs/paper",
            tmp_path / "shared",
        )["assignments"]
    )


def test_plan_rejects_node_count_mutable_images_and_insufficient_capacity(tmp_path):
    local = pinned_local(tmp_path)
    selected = plan(load_suite(), ["goal-gpt"], ["mbl_09"], [0])
    args = (selected, local, nodes(1), tmp_path / "shared/runs/paper", tmp_path / "shared")
    with pytest.raises(ValueError, match="75"):
        swarm.create_manifest(*args)
    local.agent_image = "hma-agent:local"
    with pytest.raises(ValueError, match="digest"):
        swarm.create_manifest(*args, 1)
    local.agent_image = "registry.test/hma/agent@sha256:" + "a" * 64
    args[2][0]["Description"]["Resources"]["MemoryBytes"] = 220 * 1024**3
    with pytest.raises(ValueError, match="cannot fit"):
        swarm.create_manifest(*args, 1)


def test_plan_rejects_data_mount_outside_shared_root(cluster):
    root, local, manifest = cluster
    local.data_root = Path("/different-host-path")
    with pytest.raises(ValueError, match="below shared-root"):
        swarm.create_manifest(manifest["plan"], local, nodes(2), root, root.parents[1], 2)


def test_plan_rejects_tampered_protocol_even_with_rehashed_cells(cluster):
    from hma.repro.config import fingerprint

    _, _, manifest = cluster
    selected = manifest["plan"]
    selected["cells"][0]["seconds"] += 100
    selected["plan_sha256"] = fingerprint({"suite": selected["suite"], "cells": selected["cells"]})
    with pytest.raises(ValueError, match="protocol"):
        swarm.validate_plan(selected)


def test_swarm_discovery_rejects_unready_labelled_node():
    values = nodes(2)
    client = SimpleNamespace(
        info=lambda: {"Swarm": {"ControlAvailable": True}},
        nodes=SimpleNamespace(list=lambda: [SimpleNamespace(attrs=n) for n in values]),
    )
    assert len(swarm.eligible_nodes(client)) == 2
    values[1]["Status"]["State"] = "down"
    with pytest.raises(ValueError, match="ready/active"):
        swarm.eligible_nodes(client)


def test_service_is_host_local_nonroot_secret_based_and_never_automatically_restarts(cluster):
    _, _, manifest = cluster
    command = swarm.service_command(manifest, "hma-fixture", "providers-fixture", "1000:1000", 999)
    for option, value in [
        ("--mode", "global-job"),
        ("--network", "host"),
        ("--restart-condition", "none"),
        ("--entrypoint", "hma-swarm"),
        ("--group", "999"),
        ("--stop-grace-period", "120s"),
    ]:
        assert command[command.index(option) + 1] == value
    assert "HMA_NODE_ID={{.Node.ID}}" in command
    assert "type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock" in command
    assert "source=providers-fixture,target=hma-local.json,uid=1000,gid=1000,mode=0400" in command
    assert "--gpus" not in command
    assert not any("API_KEY" in arg for arg in command)
    with pytest.raises(ValueError, match="non-root"):
        swarm.service_command(manifest, "hma-fixture", "providers-fixture", "0:0", 999)
    with pytest.raises(ValueError, match="outside"):
        swarm.service_command(
            manifest, "hma-fixture", "providers-fixture", "1000:1000", 999, node_id="outsider"
        )


def test_deploy_sends_credentials_only_through_secret_stdin(cluster, monkeypatch):
    root, local, manifest = cluster
    monkeypatch.setattr(swarm, "eligible_nodes", lambda client: nodes(2))
    monkeypatch.setattr(swarm.docker, "from_env", lambda: object())
    monkeypatch.setenv("HMA_OPENAI_API_KEY", "synthetic-private-key")
    monkeypatch.setenv("HMA_OPENAI_BASE_URL", "https://fixture.invalid/v1")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="service-id\n")

    monkeypatch.setattr(swarm.subprocess, "run", run)
    result = swarm.deploy(root, local, None, "1000:1000", 999)
    assert result["service_id"] == "service-id"
    assert calls[0][0][:3] == ["docker", "secret", "create"]
    config = json.loads(calls[0][1]["input"])
    assert config["providers"]["openai"]["api_key"] == "synthetic-private-key"
    assert "synthetic-private-key" not in json.dumps([c[0] for c in calls])
    assert "synthetic-private-key" not in (root / "cluster.json").read_text()
    with pytest.raises(ValueError, match="new --name"):
        swarm.deploy(root, local, None, "1000:1000", 999, resume=True)


def test_failed_service_creation_removes_its_new_secret(cluster, monkeypatch):
    root, local, _ = cluster
    monkeypatch.setattr(swarm, "eligible_nodes", lambda client: nodes(2))
    monkeypatch.setattr(swarm.docker, "from_env", lambda: object())
    monkeypatch.setenv("HMA_OPENAI_API_KEY", "synthetic-private-key")
    monkeypatch.setenv("HMA_OPENAI_BASE_URL", "https://fixture.invalid/v1")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["service", "create"]:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout="secret-id\n")

    monkeypatch.setattr(swarm.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        swarm.deploy(root, local, None, "1000:1000", 999)
    assert calls[-1][:3] == ["docker", "secret", "rm"]


def fake_worker(cluster, monkeypatch, leftover=False):
    _, _, manifest = cluster
    node_id = manifest["assignments"][0]["node_id"]
    client = SimpleNamespace(
        info=lambda: {"Swarm": {"NodeID": node_id}},
        containers=SimpleNamespace(list=lambda **kwargs: [object()] if leftover else []),
        images=SimpleNamespace(get=lambda name: SimpleNamespace(id=name)),
    )
    monkeypatch.setattr(swarm.docker, "from_env", lambda **kwargs: client)
    monkeypatch.setattr(swarm.os, "getuid", lambda: 1000)
    monkeypatch.setenv("HMA_NODE_ID", node_id)
    monkeypatch.setenv("DOCKER_HOST", "tcp://wrong-host:2375")
    monkeypatch.setattr(swarm, "inspect_docker_hardware", lambda *args: {"fixture": True})
    return client


def test_worker_binds_shard_to_cluster_and_runs_sequentially(cluster, monkeypatch):
    root, local, manifest = cluster
    fake_worker(cluster, monkeypatch)
    calls = []

    def run(selected, config, directory, **kwargs):
        calls.append((selected, directory, kwargs))
        return [dict(c, status="complete") for c in selected["cells"]]

    monkeypatch.setattr(swarm, "run_plan", run)
    result = swarm.worker(root, local)
    assert result["complete"] is True
    assert calls[0][2] == {
        "resume": False,
        "hardware_override": {"fixture": True},
        "sequential": True,
    }
    binding = json.loads((calls[0][1] / "cluster-assignment.json").read_text())
    assert binding == manifest["assignments"][0] | {"cluster_sha256": manifest["cluster_sha256"]}
    assert swarm.os.environ["DOCKER_HOST"] == "unix:///var/run/docker.sock"
    with pytest.raises(ValueError, match="explicit resume"):
        swarm.worker(root, local)


def test_worker_refuses_orphan_containers_before_probing_or_running(cluster, monkeypatch):
    root, local, _ = cluster
    fake_worker(cluster, monkeypatch, leftover=True)
    monkeypatch.setattr(
        swarm, "inspect_docker_hardware", lambda *a: pytest.fail("must fail before probe")
    )
    with pytest.raises(ValueError, match="sibling containers"):
        swarm.worker(root, local)
    assert not (root / "shards").exists()


def test_sigterm_interrupts_worker_and_restores_handler(monkeypatch, capsys):
    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(
        swarm,
        "parser",
        lambda: SimpleNamespace(parse_args=lambda: SimpleNamespace(command="worker")),
    )

    def interrupted(args):
        signal.raise_signal(signal.SIGTERM)
        pytest.fail("signal must interrupt the synchronous worker")

    monkeypatch.setattr(swarm, "execute", interrupted)
    assert swarm.main() == 130
    assert signal.getsignal(signal.SIGTERM) == previous
    assert "will not be retried" in capsys.readouterr().out


def test_aggregate_regrading_gives_explicit_shard_instruction(tmp_path, monkeypatch):
    from hma.repro import grade

    atomic_json(tmp_path / "environment.json", {"kind": "swarm-collection"})
    monkeypatch.setattr(grade.docker, "from_env", lambda: pytest.fail("must fail before Docker"))
    with pytest.raises(ValueError, match="shards individually"):
        grade.grade(tmp_path)


def test_synchronous_campaign_stops_on_cleanup_failure_and_resume_skips_it(tmp_path, monkeypatch):
    import hma.benchmark.prepare_data as prepare
    from hma.repro import campaign

    selected = plan(load_suite(), ["goal-gpt"], ["mbl_01", "mbl_09"], [0])
    local = pinned_local(tmp_path)
    hardware = {"fixture": True}
    for cell in selected["cells"]:
        atomic_json(local.data_root / "manifests" / f"{cell['task']}.json", {})
    client = SimpleNamespace(
        ping=lambda: True, images=SimpleNamespace(get=lambda image: SimpleNamespace(id=image))
    )
    monkeypatch.setattr(campaign.os, "getuid", lambda: 1000)
    monkeypatch.setattr(campaign, "credentials", lambda *args: {})
    monkeypatch.setattr(campaign.docker, "from_env", lambda: client)
    monkeypatch.setattr(prepare, "_verify_frozen_task", lambda *args: None)
    monkeypatch.setattr(campaign.Config, "model_validate_json", lambda text: json.loads(text))
    started = []

    def stage(cell, suite, settings, directory, gpu):
        started.append(cell["id"])
        atomic_json(directory / "launch.json", {"root": str(directory / "execution")})
        return directory / "launch.json"

    class FakeSupervisor:
        def __init__(self, config):
            self.config = config

        def run(self):
            errors = ["synthetic-cleanup-failure"] if len(started) == 1 else []
            result = {
                "status": "cleanup_failed" if errors else "complete",
                "cleanup_errors": errors,
            }
            atomic_json(Path(self.config["root"]) / "result.json", result)
            return result

    monkeypatch.setattr(campaign, "stage_cell", stage)
    monkeypatch.setattr(campaign, "Supervisor", FakeSupervisor)
    root = tmp_path / "campaign"
    with pytest.raises(ValueError, match="cleanup failed"):
        campaign.run_plan(selected, local, root, hardware_override=hardware, sequential=True)
    assert len(started) == 1
    old_result = (
        root / "cells" / selected["cells"][0]["id"] / "execution/result.json"
    ).read_bytes()
    # Worker checks the actual daemon has no orphan containers before explicit resume.
    rows = campaign.run_plan(
        selected, local, root, resume=True, hardware_override=hardware, sequential=True
    )
    assert started == [c["id"] for c in selected["cells"]]
    assert [r["status"] for r in rows] == ["cleanup_failed", "complete"]
    assert (
        root / "cells" / selected["cells"][0]["id"] / "execution/result.json"
    ).read_bytes() == old_result


def test_publishing_pins_all_images_and_keeps_private_config_permissions(tmp_path, monkeypatch):
    local = pinned_local(tmp_path)
    local.providers["openai"].api_key = "synthetic-private-key"
    tagged = []

    def run(command, **kwargs):
        tagged.append(command)
        return SimpleNamespace(returncode=0)

    def inspect(command, **kwargs):
        target = command[-1]
        return json.dumps([target.rsplit(":", 1)[0] + "@sha256:" + "d" * 64])

    monkeypatch.setattr(swarm.subprocess, "run", run)
    monkeypatch.setattr(swarm.subprocess, "check_output", inspect)
    output = tmp_path / "swarm.local.json"
    result = swarm.publish_images(local, "registry.test/hma", "repro-v1", output)
    saved = json.loads(output.read_text())
    assert len(result["images"]) == 5
    assert all("@sha256:" in image for image in result["images"])
    assert saved["providers"]["openai"]["api_key"] == "synthetic-private-key"
    assert output.stat().st_mode & 0o777 == 0o600
    assert "synthetic-private-key" not in json.dumps(tagged)
    assert "synthetic-private-key" not in json.dumps(result)
    with pytest.raises(ValueError, match="already exists"):
        swarm.publish_images(local, "registry.test/hma", "repro-v1", output)
