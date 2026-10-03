"""Swarm hardware discovery uses a bounded, isolated local Docker helper."""

import ast
import copy
import json
from types import SimpleNamespace

import pytest
import requests

from hma.repro import campaign, hardware
from hma.repro.config import Local, Provider

INVENTORY = {
    "gpus": [
        {
            "index": "0",
            "uuid": "GPU-a10",
            "name": "NVIDIA A10",
            "memory_mib": 23028,
            "driver": "570.148.08",
        }
    ],
    "cpu_model": "Intel Xeon Platinum 8358",
    "logical_cpus": 30,
    "memory_bytes": 220 * 1024**3,
}


class Helper:
    def __init__(self, payload=None, status=0, error=None):
        self.payload = payload if payload is not None else {"hardware": INVENTORY, "code": "same"}
        self.status = status
        self.error = error
        self.events = []

    def start(self):
        self.events.append("start")

    def wait(self, **kwargs):
        self.events.append(("wait", kwargs))
        if self.error:
            raise self.error
        return {"StatusCode": self.status}

    def logs(self, **kwargs):
        self.events.append(("logs", kwargs))
        return (
            self.payload if isinstance(self.payload, bytes) else json.dumps(self.payload).encode()
        )

    def remove(self, **kwargs):
        self.events.append(("remove", kwargs))


@pytest.fixture
def helper_client(monkeypatch):
    monkeypatch.setattr(hardware.os, "getuid", lambda: 1000)
    monkeypatch.setattr(hardware.os, "getgid", lambda: 1001)
    monkeypatch.setattr(campaign, "code_identity", lambda: "same")

    def build(helper):
        calls = []

        def create(image, **kwargs):
            calls.append((image, kwargs))
            return helper

        return SimpleNamespace(containers=SimpleNamespace(create=create)), calls

    return build


def test_node_probe_uses_local_daemon_without_mounts_or_credentials(helper_client):
    helper = Helper()
    client, calls = helper_client(helper)
    local = Local(
        providers={"openai": Provider(api_key="secret-never-forwarded", base_url="https://local")},
        agent_image="registry.example/hma@sha256:pinned",
    )
    assert hardware.inspect_docker_hardware(local, client) == INVENTORY
    image, create = calls[0]
    assert image == local.agent_image
    assert create["entrypoint"] == []
    assert create["user"] == "1000:1001"
    assert create["environment"] == {"NVIDIA_DRIVER_CAPABILITIES": "utility"}
    assert create["device_requests"][0]["Count"] == -1
    assert create["device_requests"][0]["Capabilities"] == [["gpu"]]
    assert create["network_disabled"] and create["read_only"]
    assert "volumes" not in create and "mounts" not in create
    assert "secret-never-forwarded" not in json.dumps(create)
    assert json.loads(create["command"][-1]) == {
        "providers": {},
        "gpus": ["0"],
        "expected_gpu_model": "NVIDIA A10",
    }
    ast.parse(create["command"][2])
    assert ("wait", {"timeout": 30}) in helper.events
    assert helper.events[-1] == ("remove", {"force": True})


def test_node_probe_timeout_removes_the_helper(helper_client):
    helper = Helper(error=requests.exceptions.ReadTimeout("inventory timeout"))
    client, _ = helper_client(helper)
    with pytest.raises(ValueError, match="inventory timeout"):
        hardware.inspect_docker_hardware(Local(providers={}), client)
    assert helper.events[-1] == ("remove", {"force": True})


@pytest.mark.parametrize(
    "payload,status,message",
    [
        ({"error": "GPU 0 is NVIDIA H200; expected NVIDIA A10"}, 1, "expected NVIDIA A10"),
        ({"hardware": INVENTORY, "code": "different"}, 0, "different reproduction code"),
        (b"invalid-json", 0, "cannot inspect node hardware"),
        ([], 0, "non-object payload"),
        ({"hardware": {}, "code": "same"}, 0, "invalid inventory"),
    ],
)
def test_helper_failures_stop_preflight_and_always_cleanup(helper_client, payload, status, message):
    helper = Helper(payload=payload, status=status)
    client, _ = helper_client(helper)
    with pytest.raises(ValueError, match=message):
        hardware.inspect_docker_hardware(Local(providers={}), client)
    assert helper.events[-1] == ("remove", {"force": True})


def test_parent_rechecks_the_gpu_model_and_complete_inventory(helper_client):
    for change, message in [
        ("wrong-model", "expected NVIDIA A10"),
        ("missing-driver", "invalid GPU"),
    ]:
        inventory = copy.deepcopy(INVENTORY)
        if change == "wrong-model":
            inventory["gpus"][0]["name"] = "NVIDIA H200"
        else:
            del inventory["gpus"][0]["driver"]
        helper = Helper(payload={"hardware": inventory, "code": "same"})
        client, _ = helper_client(helper)
        with pytest.raises(ValueError, match=message):
            hardware.inspect_docker_hardware(Local(providers={}), client)
        assert helper.events[-1] == ("remove", {"force": True})


def test_helper_cleanup_failure_does_not_report_readiness(helper_client):
    helper = Helper()

    def removal_fails(**kwargs):
        raise requests.exceptions.ConnectionError("daemon unavailable")

    helper.remove = removal_fails
    client, _ = helper_client(helper)
    with pytest.raises(ValueError, match="cleanup failed"):
        hardware.inspect_docker_hardware(Local(providers={}), client)
