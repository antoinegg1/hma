"""Record actual host hardware and check the declared GPU model before execution."""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

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
