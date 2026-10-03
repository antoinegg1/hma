"""Build public-base Docker images and frozen upstream harness layers."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from hma.repro.campaign import baseline_root
from hma.repro.config import ASSETS, Local
from hma.repro.sources import checkout, verify_baseline


def repository() -> Path:
    root = Path(__file__).resolve().parents[3]
    if not (root / "docker/agent.Dockerfile").is_file():
        raise ValueError("image builds require the source checkout; run from an editable install")
    return root


def build(local: Local, names: list[str]) -> None:
    root = repository()
    selected = names or ["agent", "evaluator", "ml_master_v2", "mlevolve_no_prior", "scienceflow"]
    valid = {"agent", "evaluator", "ml_master_v2", "mlevolve_no_prior", "scienceflow"}
    if set(selected) - valid:
        raise ValueError("unknown image name")
    for name in selected:
        if name in {"agent", "evaluator"}:
            image = local.agent_image if name == "agent" else local.evaluator_image
            subprocess.run(
                [
                    "docker",
                    "build",
                    "-f",
                    str(root / "docker" / f"{name}.Dockerfile"),
                    "-t",
                    image,
                    str(root),
                ],
                check=True,
            )
            continue
        upstream = checkout(local.data_root / "sources", name)
        baseline = baseline_root() / name
        verify_baseline(upstream, baseline / "source.lock.json", baseline / "paper_runtime")
        context = root / ".cache" / "build" / name
        context.mkdir(parents=True, exist_ok=True)
        # Exact lock allowlist avoids shipping cached runs/credentials or unrelated upstream assets.
        lock = json.loads((baseline / "source.lock.json").read_text())
        source_dir = context / "upstream"
        if source_dir.exists():
            shutil.rmtree(source_dir)
        for relative in lock["files"]:
            target = source_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(upstream / relative, target)
        shutil.copytree(baseline / "paper_runtime", context / "paper_runtime", dirs_exist_ok=True)
        shutil.copyfile(baseline / "source.lock.json", context / "source.lock.json")
        shutil.copyfile(root / "docker/harness.Dockerfile", context / "Dockerfile")
        shutil.copyfile(
            root / "docker/requirements-harness.lock", context / "requirements-harness.lock"
        )
        (context / "model-assets.json").write_text(
            json.dumps(
                json.loads((ASSETS / "sources.lock.json").read_text()).get("model_assets", {})
            )
        )
        subprocess.run(
            [
                "docker",
                "build",
                "--build-arg",
                f"HMA_AGENT_IMAGE={local.agent_image}",
                "--build-arg",
                f"HARNESS={name}",
                "-t",
                local.harness_images[name],
                str(context),
            ],
            check=True,
        )
