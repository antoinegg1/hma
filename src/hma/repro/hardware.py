"""Record actual host hardware and check the declared GPU model before execution."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import requests

import docker
from hma.repro.config import Local


def parse_gpus(output: str) -> list[dict]:
    devices = []
    for row in csv.reader(output.splitlines()):
        if len(row) != 5:
            raise ValueError("unexpected nvidia-smi GPU inventory format")
        index, uuid, name, memory, driver = [field.strip() for field in row]
        devices.append(
            {
                "index": index,
                "uuid": uuid,
                "name": name,
                "memory_mib": int(memory),
                "driver": driver,
            }
        )
    return devices


def check_gpus(devices: list[dict], local: Local) -> list[dict]:
    selected = []
    for requested in local.gpus:
        found = next((d for d in devices if requested in (d["index"], d["uuid"])), None)
        if found is None:
            raise ValueError(f"configured GPU is not present: {requested}")
        if any(device["uuid"] == found["uuid"] for device in selected):
            raise ValueError(f"GPU {requested} selects the same physical device twice")
        actual = found["name"].removeprefix("NVIDIA ").casefold()
        expected = (local.expected_gpu_model or "").removeprefix("NVIDIA ").casefold()
        if expected and actual != expected:
            raise ValueError(
                f"GPU {requested} is {found['name']}; expected {local.expected_gpu_model}. "
                "Use the paper's A10 hardware, or explicitly change expected_gpu_model "
                "for a documented hardware variant."
            )
        selected.append(found)
    return selected


def inspect_hardware(local: Local) -> dict:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"cannot inspect GPU hardware with nvidia-smi: {error}") from error
    devices = check_gpus(parse_gpus(completed.stdout), local)
    cpu_lines = Path("/proc/cpuinfo").read_text().splitlines()
    model = next(
        (line.split(":", 1)[1].strip() for line in cpu_lines if line.startswith("model name")), None
    )
    mem_lines = Path("/proc/meminfo").read_text().splitlines()
    memory_kib = next(int(line.split()[1]) for line in mem_lines if line.startswith("MemTotal:"))
    return {
        "gpus": devices,
        "cpu_model": model,
        "logical_cpus": os.cpu_count(),
        "memory_bytes": memory_kib * 1024,
    }


def inspect_docker_hardware(local: Local, client) -> dict:
    """Inspect a Swarm node through its daemon without mounting GPUs in the controller."""
    from hma.repro.campaign import code_identity

    if os.getuid() == 0:
        raise ValueError("run the hardware helper as a non-root Docker user")
    settings = {
        "providers": {},
        "gpus": local.gpus,
        "expected_gpu_model": local.expected_gpu_model,
    }
    probe = """import json, sys
from hma.repro.config import Local
from hma.repro.campaign import code_identity
from hma.repro.hardware import inspect_hardware
try:
    hardware = inspect_hardware(Local.model_validate_json(sys.argv[1]))
    print(json.dumps({"hardware": hardware, "code": code_identity()}), flush=True)
except Exception as error:
    print(json.dumps({"error": str(error)}), flush=True)
    raise SystemExit(1)
"""
    container = None
    try:
        container = client.containers.create(
            local.agent_image,
            entrypoint=[],
            command=["python3", "-c", probe, json.dumps(settings)],
            user=f"{os.getuid()}:{os.getgid()}",
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
            environment={"NVIDIA_DRIVER_CAPABILITIES": "utility"},
            network_disabled=True,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            read_only=True,
            pids_limit=64,
            labels={"io.flowbench.role": "hardware-preflight"},
        )
        container.start()
        status = container.wait(timeout=30)
        payload = json.loads(container.logs(stdout=True, stderr=False))
        if not isinstance(status, dict):
            raise ValueError("hardware helper returned an invalid exit status")
        if not isinstance(payload, dict):
            raise ValueError("hardware helper returned a non-object payload")
        if status.get("StatusCode") != 0:
            raise ValueError(payload.get("error") or "hardware helper exited unsuccessfully")
        if payload.get("code") != code_identity():
            raise ValueError("hardware helper and controller use different reproduction code")
        hardware = payload.get("hardware")
        required = {"gpus", "cpu_model", "logical_cpus", "memory_bytes"}
        if not isinstance(hardware, dict) or set(hardware) != required:
            raise ValueError("hardware helper returned an invalid inventory")
        if (
            not isinstance(hardware["gpus"], list)
            or not hardware["gpus"]
            or not isinstance(hardware["cpu_model"], (str, type(None)))
            or type(hardware["logical_cpus"]) is not int
            or hardware["logical_cpus"] <= 0
            or type(hardware["memory_bytes"]) is not int
            or hardware["memory_bytes"] <= 0
        ):
            raise ValueError("hardware helper returned an invalid inventory")
        for device in hardware["gpus"]:
            if (
                not isinstance(device, dict)
                or set(device) != {"index", "uuid", "name", "memory_mib", "driver"}
                or any(
                    not isinstance(device[key], str) or not device[key]
                    for key in ("index", "uuid", "name", "driver")
                )
                or type(device["memory_mib"]) is not int
                or device["memory_mib"] <= 0
            ):
                raise ValueError("hardware helper returned an invalid GPU inventory")
        hardware["gpus"] = check_gpus(hardware["gpus"], local)
        return hardware
    except (
        docker.errors.DockerException,
        requests.RequestException,
        ValueError,
        TypeError,
        KeyError,
    ) as error:
        raise ValueError(f"cannot inspect node hardware through Docker: {error}") from error
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except (docker.errors.DockerException, requests.RequestException) as error:
                raise ValueError(
                    "hardware helper cleanup failed; inspect the local daemon"
                ) from error
