"""Hardware preflight must not silently label a different GPU as the paper's A10."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from docker.utils import parse_bytes

from hma.repro import hardware
from hma.repro.config import Local, load_local
from hma.supervisor import Config

INVENTORY = (
    "0, GPU-a10, NVIDIA A10, 23028, 570.148.08\n1, GPU-h200, NVIDIA H200, 143771, 570.148.08\n"
)


def test_paper_resource_defaults_and_templates_use_gib():
    # Manuscript Appendix C.1: one A10, 30 vCPUs, 220 GiB shared by alternating actors.
    local = load_local(Path("configs/local.json"))
    default = Local(providers={})
    legacy = SimpleNamespace(**json.loads(Path("configs/hma-opus-gpt.json").read_text()))
    for settings in [local, default, legacy]:
        assert settings.cpus == 30
        assert parse_bytes(settings.memory) == 220 * 1024**3
    assert local.expected_gpu_model == default.expected_gpu_model == "NVIDIA A10"
    assert Config.model_fields["cpus"].default == 30
    assert parse_bytes(Config.model_fields["memory"].default) == 220 * 1024**3


def test_inventory_selects_by_index_or_uuid_and_retains_device_evidence():
    devices = hardware.parse_gpus(INVENTORY)
    expected = {
        "index": "0",
        "uuid": "GPU-a10",
        "name": "NVIDIA A10",
        "memory_mib": 23028,
        "driver": "570.148.08",
    }
    for selector in ["0", "GPU-a10"]:
        assert hardware.check_gpus(devices, Local(providers={}, gpus=[selector])) == [expected]


@pytest.mark.parametrize("model", ["NVIDIA H200", "NVIDIA A10G"])
def test_different_gpu_is_rejected_before_running(model):
    devices = hardware.parse_gpus(INVENTORY.replace("NVIDIA A10,", f"{model},"))
    with pytest.raises(ValueError, match="expected NVIDIA A10"):
        hardware.check_gpus(devices, Local(providers={}))


def test_explicit_hardware_variant_retains_actual_gpu():
    devices = hardware.parse_gpus(INVENTORY)
    for expectation in ["NVIDIA H200", None]:
        selected = hardware.check_gpus(
            devices, Local(providers={}, gpus=["1"], expected_gpu_model=expectation)
        )
        assert selected[0]["name"] == "NVIDIA H200"
        assert selected[0]["uuid"] == "GPU-h200"


def test_missing_or_aliased_duplicate_gpu_is_rejected():
    devices = hardware.parse_gpus(INVENTORY)
    with pytest.raises(ValueError, match="not present"):
        hardware.check_gpus(devices, Local(providers={}, gpus=["7"]))
    with pytest.raises(ValueError, match="same physical device twice"):
        hardware.check_gpus(devices, Local(providers={}, gpus=["0", "GPU-a10"]))


def test_hardware_probe_records_cpu_memory_and_driver(monkeypatch):
    calls = []

    def query(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=INVENTORY)

    monkeypatch.setattr(hardware.subprocess, "run", query)
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 30)
    proc = {
        "/proc/cpuinfo": "processor : 0\nmodel name : Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz\n",
        "/proc/meminfo": "MemTotal: 230686720 kB\nMemFree: 1024 kB\n",
    }
    monkeypatch.setattr(hardware.Path, "read_text", lambda path: proc[str(path)])
    result = hardware.inspect_hardware(Local(providers={}))
    assert result["gpus"][0]["driver"] == "570.148.08"
    assert result["logical_cpus"] == 30
    assert "8358" in result["cpu_model"]
    assert result["memory_bytes"] == 220 * 1024**3
    assert calls[0][0][0] == "nvidia-smi"
    assert calls[0][1]["check"] is True


def test_doctor_blocks_gpu_mismatch_without_api_calls(monkeypatch):
    import docker
    from hma.repro import cli

    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(cli.os, "getuid", lambda: 1000)
    monkeypatch.setattr(docker, "from_env", lambda: SimpleNamespace(ping=lambda: True))

    def mismatch(local):
        raise ValueError("GPU 0 is NVIDIA H200; expected NVIDIA A10")

    monkeypatch.setattr(hardware, "inspect_hardware", mismatch)
    result = cli.doctor(Local(providers={}), offline=False)
    assert result["gpu_model_matches"] is False
    assert result["ready"] is False
    assert "NVIDIA H200" in result["hardware_error"]


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("nvidia-smi"),
        subprocess.CalledProcessError(9, "nvidia-smi"),
        subprocess.TimeoutExpired("nvidia-smi", 15),
    ],
)
def test_unavailable_gpu_probe_is_a_clear_preflight_error(monkeypatch, error):
    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(hardware.subprocess, "run", unavailable)
    with pytest.raises(ValueError, match="cannot inspect GPU hardware"):
        hardware.inspect_hardware(Local(providers={}))
