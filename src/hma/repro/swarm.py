"""Freeze, deploy and collect one benchmark-task shard per dedicated Swarm node."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
from pathlib import Path

from docker.utils import parse_bytes

import docker
from hma.benchmark.integrity import atomic_json
from hma.repro.campaign import code_identity, run_plan
from hma.repro.config import Local, Suite, fingerprint, load_local, load_suite, plan
from hma.repro.hardware import inspect_docker_hardware
from hma.repro.providers import credentials, names
from hma.supervisor import LABEL

NODE_LABEL = "hma.worker"
IMAGE_DIGEST = re.compile(r"[^\s,]+@sha256:[0-9a-f]{64}\Z")


def image_refs(local: Local) -> list[str]:
    return sorted({local.agent_image, local.evaluator_image, *local.harness_images.values()})


def validate_plan(selected: dict) -> None:
    suite = Suite.model_validate(selected["suite"])
    canonical = {c["id"]: c for c in plan(suite, [], [], [])["cells"]}
    cells = selected["cells"]
    if not cells or len({c["id"] for c in cells}) != len(cells):
        raise ValueError("empty or duplicate cells in plan")
    if any(canonical.get(c["id"]) != c for c in cells):
        raise ValueError("plan cells differ from the declared experiment protocol")
    if selected["plan_sha256"] != fingerprint({"suite": selected["suite"], "cells": cells}):
        raise ValueError("plan fingerprint mismatch")
    if selected["max_gpu_hours"] != sum(c["seconds"] for c in cells) / 3600:
        raise ValueError("plan budget mismatch")


def shard_plan(selected: dict, task: str) -> dict:
    cells = [c for c in selected["cells"] if c["task"] == task]
    return {
        "schema_version": 1,
        "suite": selected["suite"],
        "cells": cells,
        "plan_sha256": fingerprint({"suite": selected["suite"], "cells": cells}),
        "max_gpu_hours": sum(c["seconds"] for c in cells) / 3600,
    }


def eligible_nodes(client) -> list[dict]:
    if not client.info().get("Swarm", {}).get("ControlAvailable"):
        raise ValueError("this command needs a Swarm manager")
    nodes = [
        n.attrs
        for n in client.nodes.list()
        if n.attrs.get("Spec", {}).get("Labels", {}).get(NODE_LABEL) == "true"
    ]
    for node in nodes:
        if node["Status"]["State"] != "ready" or node["Spec"]["Availability"] != "active":
            raise ValueError(f"labelled node is not ready/active: {node['ID']}")
        platform = node["Description"]["Platform"]
        if platform["OS"] != "linux" or platform["Architecture"] not in {"x86_64", "amd64"}:
            raise ValueError("Swarm workers must run Linux x86-64")
    return sorted(nodes, key=lambda n: n["ID"])


def create_manifest(
    selected: dict,
    local: Local,
    nodes: list[dict],
    root: Path,
    shared_root: Path,
    expected_nodes: int = 75,
) -> dict:
    validate_plan(selected)
    tasks = sorted({c["task"] for c in selected["cells"]})
    if len(nodes) != expected_nodes or len(tasks) != expected_nodes:
        raise ValueError(
            f"need exactly {expected_nodes} labelled nodes and distinct benchmark tasks"
        )
    if len({n["ID"] for n in nodes}) != len(nodes):
        raise ValueError("duplicate Swarm node IDs")
    if len(local.gpus) != 1:
        raise ValueError("Swarm requires exactly one selected GPU per node")
    if any(not IMAGE_DIGEST.fullmatch(ref) for ref in image_refs(local)):
        raise ValueError(
            "pin every image to a registry @sha256 digest using python -m hma.repro.swarm images"
        )
    required = {
        e.harness
        for e in Suite.model_validate(selected["suite"]).experiments
        if e.harness and any(c["experiment"] == e.id for c in selected["cells"])
    }
    if required - local.harness_images.keys():
        raise ValueError("missing external harness images")
    root, shared_root = root.resolve(), shared_root.resolve()
    if shared_root == Path("/") or any("," in str(p) for p in (root, shared_root)):
        raise ValueError("use a dedicated shared directory without commas, such as /srv/hma")
    if not root.is_relative_to(shared_root) or not local.data_root.is_relative_to(shared_root):
        raise ValueError("run-root and absolute data_root must be below shared-root")
    for node in nodes:
        capacity = node["Description"]["Resources"]
        if capacity["NanoCPUs"] < (local.cpus + local.evaluator_cpus) * 1e9 or capacity[
            "MemoryBytes"
        ] < parse_bytes(local.memory) + parse_bytes(local.evaluator_memory):
            raise ValueError(f"node {node['ID']} cannot fit agent plus evaluator resource limits")
    assignments = [
        {"node_id": n["ID"], "node_hostname": n["Description"]["Hostname"], "task": task}
        for n, task in zip(sorted(nodes, key=lambda n: n["ID"]), tasks, strict=True)
    ]
    manifest = {
        "schema_version": 1,
        "plan": selected,
        "assignments": assignments,
        "settings": local.model_dump(mode="json", exclude={"providers"}),
        "code": code_identity(),
        "root": str(root),
        "shared_root": str(shared_root),
    }
    manifest["cluster_sha256"] = fingerprint(manifest)
    return manifest


def load_manifest(root: Path) -> dict:
    manifest = json.loads((root / "cluster.json").read_text())
    if manifest["cluster_sha256"] != fingerprint(
        {k: v for k, v in manifest.items() if k != "cluster_sha256"}
    ):
        raise ValueError("cluster fingerprint mismatch")
    validate_plan(manifest["plan"])
    if Path(manifest["root"]) != root.resolve():
        raise ValueError("cluster run-root must retain its frozen absolute path")
    assignments = manifest["assignments"]
    if (
        len({a["node_id"] for a in assignments}) != len(assignments)
        or len({a["task"] for a in assignments}) != len(assignments)
        or {a["task"] for a in assignments} != {c["task"] for c in manifest["plan"]["cells"]}
    ):
        raise ValueError("invalid node-to-task partition")
    return manifest


def publish_images(local: Local, repository: str, tag: str, output: Path) -> dict:
    if output.exists():
        raise ValueError("output config already exists; choose a new ignored *.local.json")
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._/-]*(?::[0-9]+)?(?:/[A-Za-z0-9._/-]+)?", repository
    ):
        raise ValueError("invalid image repository prefix")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag):
        raise ValueError("invalid image tag")
    sources = {
        "agent": local.agent_image,
        "evaluator": local.evaluator_image,
        **local.harness_images,
    }
    pinned = {}
    for role, source in sources.items():
        target = f"{repository.rstrip('/')}/{role}:{tag}"
        subprocess.run(["docker", "tag", source, target], check=True)
        subprocess.run(["docker", "push", target], check=True)
        raw = subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", target], text=True
        )
        matches = [ref for ref in json.loads(raw) if ref.startswith(target.rsplit(":", 1)[0] + "@")]
        if len(matches) != 1 or not IMAGE_DIGEST.fullmatch(matches[0]):
            raise ValueError(f"cannot determine published digest for {role}")
        pinned[role] = matches[0]
    result = local.model_dump(mode="json")
    result.update(
        agent_image=pinned.pop("agent"),
        evaluator_image=pinned.pop("evaluator"),
        harness_images=pinned,
    )
    atomic_json(output, result)  # atomic_json creates user-only files, including credentials.
    return {"config": str(output), "images": image_refs(Local.model_validate(result))}


def check_settings(manifest: dict, local: Local) -> None:
    if manifest["settings"] != local.model_dump(mode="json", exclude={"providers"}):
        raise ValueError("local resource/image/data settings differ from frozen cluster")
    if manifest["code"] != code_identity():
        raise ValueError("code differs from frozen cluster; rebuild images and use a new root")


def service_command(
    manifest: dict,
    name: str,
    secret: str,
    user: str,
    docker_gid: int,
    resume: bool = False,
    node_id: str | None = None,
) -> list[str]:
    if not re.fullmatch(r"[1-9][0-9]*:[0-9]+", user) or docker_gid < 0:
        raise ValueError("use a non-root numeric UID:GID and the common Docker socket GID")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
        raise ValueError("invalid service name")
    uid, gid = user.split(":")
    shared = manifest["shared_root"]
    command = [
        "docker",
        "service",
        "create",
        "--detach",
        "--name",
        name,
        "--mode",
        "global-job",
        "--constraint",
        f"node.labels.{NODE_LABEL}==true",
        "--network",
        "host",
        "--restart-condition",
        "none",
        "--stop-grace-period",
        "120s",
        "--init",
        "--user",
        user,
        "--group",
        str(docker_gid),
        "--with-registry-auth",
        "--env",
        "HMA_NODE_ID={{.Node.ID}}",
        "--env",
        "DOCKER_HOST=unix:///var/run/docker.sock",
        "--env",
        "HUMANIZE_SENTRY=off",
        "--workdir",
        shared,
        "--mount",
        f"type=bind,src={shared},dst={shared}",
        "--mount",
        "type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock",
        "--secret",
        f"source={secret},target=hma-local.json,uid={uid},gid={gid},mode=0400",
        "--entrypoint",
        "python3",
    ]
    if node_id:
        if node_id not in {a["node_id"] for a in manifest["assignments"]}:
            raise ValueError("node is outside the frozen assignment")
        command += ["--constraint", f"node.id=={node_id}"]
    command += [
        manifest["settings"]["agent_image"],
        "-m",
        "hma.repro.swarm",
        "worker",
        "--root",
        manifest["root"],
        "--config",
        "/run/secrets/hma-local.json",
    ]
    if resume:
        command += ["--resume"]
    return command


def deploy(
    root: Path,
    local: Local,
    name: str | None,
    user: str,
    docker_gid: int,
    resume: bool = False,
    node_id: str | None = None,
) -> dict:
    manifest = load_manifest(root)
    check_settings(manifest, local)
    nodes = eligible_nodes(docker.from_env())
    if {n["ID"] for n in nodes} != {a["node_id"] for a in manifest["assignments"]}:
        raise ValueError("Swarm membership changed since plan; restore the original labelled nodes")
    if not resume and (root / "shards").exists():
        raise ValueError("existing shards require an explicit --resume deployment")
    if resume and not name:
        raise ValueError("resume requires a new --name; never update or force-restart the old job")
    name = name or "hma-" + manifest["cluster_sha256"][:12]
    secret = name + "-providers"
    command = service_command(manifest, name, secret, user, docker_gid, resume, node_id)
    suite = Suite.model_validate(manifest["plan"]["suite"])
    selected = {c["experiment"] for c in manifest["plan"]["cells"]}
    required = {
        suite.models[a].provider for e in suite.experiments if e.id in selected for a in e.actors
    }
    values = credentials(local, required)
    config = local.model_dump(mode="json")
    for provider in required:
        key, base = names(provider)
        config["providers"][provider] = {"api_key": values[key], "base_url": values[base]}
    # Credentials travel via stdin into a Swarm secret, never argv, env or the manifest.
    subprocess.run(
        ["docker", "secret", "create", secret, "-"],
        input=json.dumps(config),
        text=True,
        capture_output=True,
        check=True,
    )
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True)
    except BaseException:
        subprocess.run(["docker", "secret", "rm", secret], capture_output=True, check=False)
        raise
    return {
        "service": name,
        "service_id": result.stdout.strip(),
        "secret": secret,
        "cluster_sha256": manifest["cluster_sha256"],
    }


def worker(root: Path, local: Local, resume: bool = False) -> dict:
    manifest = load_manifest(root)
    check_settings(manifest, local)
    if os.getuid() == 0:
        raise ValueError("worker must use the configured non-root UID")
    # Never use a remote DOCKER_HOST: sibling mounts and hardware must refer to this host.
    os.environ["DOCKER_HOST"] = "unix:///var/run/docker.sock"
    client = docker.from_env(timeout=30)
    node_id = client.info().get("Swarm", {}).get("NodeID")
    if node_id != os.environ.get("HMA_NODE_ID"):
        raise ValueError("controller node identity does not match the local Docker daemon")
    assigned = next((a for a in manifest["assignments"] if a["node_id"] == node_id), None)
    if assigned is None:
        raise ValueError("this node has no frozen assignment")
    lock_root = Path(manifest["shared_root"]) / "node-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    with (lock_root / f"{node_id}.lock").open("a+") as node_lock:
        fcntl.flock(node_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if client.containers.list(all=True, filters={"label": LABEL}):
            raise ValueError(
                "node has HMA sibling containers; inspect/clean the original owner before resume"
            )
        for image in image_refs(local):
            client.images.get(image)  # Preload all pinned images on every node before deploying.
        hardware = inspect_docker_hardware(local, client)
        shard = root / "shards" / assigned["task"]
        binding = {"cluster_sha256": manifest["cluster_sha256"], **assigned}
        assignment_file = shard / "cluster-assignment.json"
        if shard.exists():
            if (
                not resume
                or not assignment_file.exists()
                or json.loads(assignment_file.read_text()) != binding
            ):
                raise ValueError(
                    "existing shard requires matching cluster assignment and explicit resume"
                )
        else:
            shard.mkdir(parents=True)
            atomic_json(assignment_file, binding)
        rows = run_plan(
            shard_plan(manifest["plan"], assigned["task"]),
            local,
            shard,
            resume=resume and (shard / "plan.json").exists(),
            hardware_override=hardware,
            sequential=True,
        )
        return {
            "task": assigned["task"],
            "cells": len(rows),
            "complete": all(r["status"] == "complete" for r in rows),
        }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    images = commands.add_parser(
        "images", help="publish built images and write a digest-pinned config"
    )
    images.add_argument("--config", type=Path, default=Path("configs/local.json"))
    images.add_argument("--repository", required=True)
    images.add_argument("--tag", required=True)
    images.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("plan", help="freeze node-to-task assignments on the manager")
    prepare.add_argument("--config", type=Path, default=Path("configs/swarm.local.json"))
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--shared-root", type=Path, default=Path("/srv/hma"))
    prepare.add_argument("--expected-nodes", type=int, default=75)
    prepare.add_argument("--plan-file", type=Path)
    launch = commands.add_parser("deploy", help="launch one controller job per frozen node")
    launch.add_argument("--root", type=Path, required=True)
    launch.add_argument("--config", type=Path, default=Path("configs/swarm.local.json"))
    launch.add_argument("--name")
    launch.add_argument("--user", default="1000:1000")
    launch.add_argument("--docker-gid", type=int, required=True)
    launch.add_argument("--resume", action="store_true")
    launch.add_argument("--node-id", help="optionally target one assigned node during recovery")
    run = commands.add_parser("worker", help="internal Swarm job entry point")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    collect = commands.add_parser(
        "collect", help="validate completed shards for the existing reporter"
    )
    collect.add_argument("--root", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument(
        "--allow-partial",
        action="store_true",
        help="diagnose pending/interrupted cells in initialized shards",
    )
    return root


def execute(args: argparse.Namespace) -> dict:
    if args.command == "collect":
        from hma.repro.swarm_collect import collect

        return collect(args.root.resolve(), args.output.resolve(), args.allow_partial)
    local = load_local(args.config)
    if args.command == "images":
        return publish_images(local, args.repository, args.tag, args.output)
    root = args.root.resolve()
    if args.command == "plan":
        selected = (
            json.loads(args.plan_file.read_text())
            if args.plan_file
            else plan(load_suite(), [], [], [])
        )
        manifest = create_manifest(
            selected,
            local,
            eligible_nodes(docker.from_env()),
            root,
            args.shared_root,
            args.expected_nodes,
        )
        if root.exists():
            raise ValueError("cluster root already exists; use a new root")
        root.mkdir(parents=True)
        atomic_json(root / "cluster.json", manifest)
        (root / "images.txt").write_text("\n".join(image_refs(local)) + "\n")
        return {
            "root": str(root),
            "nodes": len(manifest["assignments"]),
            "cells": len(selected["cells"]),
            "cluster_sha256": manifest["cluster_sha256"],
        }
    if args.command == "deploy":
        return deploy(root, local, args.name, args.user, args.docker_gid, args.resume, args.node_id)
    return worker(root, local, args.resume)


def main() -> int:
    args = parser().parse_args()
    previous = None
    if args.command == "worker":

        def terminate(signum, frame):
            raise KeyboardInterrupt("Swarm worker stopping; preserving started attempts")

        previous = signal.signal(signal.SIGTERM, terminate)
    try:
        result = execute(args)
        print(json.dumps(result, indent=2))
        return 1 if result.get("complete") is False else 0
    except KeyboardInterrupt:
        print(json.dumps({"error": "worker interrupted; started cells will not be retried"}))
        return 130
    except (
        ValueError,
        OSError,
        KeyError,
        docker.errors.DockerException,
        subprocess.SubprocessError,
    ) as error:
        # Never print a provider config, subprocess stdin, or a Docker request body.
        print(json.dumps({"error": str(error)}))
        return 1
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
