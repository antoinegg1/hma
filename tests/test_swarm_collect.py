"""A collected report must represent exactly one frozen, finished cluster."""

from __future__ import annotations

import fcntl
import json
import os
import shutil

import pytest

from hma.analysis.events import normalize
from hma.benchmark.integrity import atomic_json
from hma.repro.config import Local, Suite, fingerprint, load_suite, plan
from hma.repro.swarm_collect import collect


def read(path):
    return json.loads(path.read_text())


def rewrite(path, update):
    value = read(path)
    update(value)
    atomic_json(path, value)


def rehash_environment(environment):
    environment["identity"] = fingerprint(
        {key: environment[key] for key in ("settings", "images", "code", "data", "hardware")}
    )


@pytest.fixture
def frozen_cluster(tmp_path):
    root = tmp_path / "cluster"
    selected = plan(load_suite(), ["goal-gpt"], ["mbl_01", "mbl_09"], [0])
    tasks = sorted({cell["task"] for cell in selected["cells"]})
    settings = Local(providers={}, data_root=tmp_path / "data").model_dump(
        mode="json", exclude={"providers"}
    )
    cluster = {
        "schema_version": 1,
        "plan": selected,
        "settings": settings,
        "code": fingerprint({"source": "same-frozen-code"}),
        "assignments": [
            {"task": task, "node_id": f"node-{index}", "node_hostname": f"host-{index}"}
            for index, task in enumerate(tasks)
        ],
    }
    cluster["cluster_sha256"] = fingerprint(cluster)
    atomic_json(root / "cluster.json", cluster)
    for index, assignment in enumerate(cluster["assignments"]):
        task = assignment["task"]
        shard = root / "shards" / task
        shard.mkdir(parents=True)
        (shard / ".campaign.lock").touch()
        atomic_json(
            shard / "cluster-assignment.json",
            assignment | {"cluster_sha256": cluster["cluster_sha256"]},
        )
        cells = [cell for cell in selected["cells"] if cell["task"] == task]
        atomic_json(
            shard / "plan.json",
            {
                "schema_version": 1,
                "suite": selected["suite"],
                "cells": cells,
                "plan_sha256": fingerprint({"suite": selected["suite"], "cells": cells}),
                "max_gpu_hours": sum(cell["seconds"] for cell in cells) / 3600,
            },
        )
        data_hash = fingerprint({"task": task})
        environment = {
            "settings": settings,
            "code": cluster["code"],
            "images": {
                "hma-agent:local": "sha256:agent",
                "hma-evaluator:local": "sha256:evaluator",
            },
            "data": {task: data_hash},
            "hardware": {
                "gpus": [
                    {
                        "index": "0",
                        "uuid": f"GPU-{index}",
                        "name": "NVIDIA A10",
                        "memory_mib": 23028,
                        "driver": "570.148.08",
                    }
                ],
                "cpu_model": "fixture CPU",
                "logical_cpus": 40,
                "memory_bytes": 256 * 1024**3,
            },
        }
        rehash_environment(environment)
        atomic_json(shard / "environment.json", environment)
        for cell in cells:
            directory = shard / "cells" / cell["id"]
            atomic_json(directory / "cell.json", cell | {"data_manifest_sha256": data_hash})
            atomic_json(
                directory / "execution/result.json",
                {"status": "complete" if index == 0 else "failed"},
            )
    return root


def first_shard(root):
    return root / "shards" / read(root / "cluster.json")["assignments"][0]["task"]


def first_cell(shard):
    return shard / "cells" / read(shard / "plan.json")["cells"][0]["id"]


def test_collect_preserves_failed_coverage_hardware_and_source_evidence(frozen_cluster, tmp_path):
    before = {str(path): path.read_bytes() for path in frozen_cluster.rglob("*") if path.is_file()}
    output = tmp_path / "report-root"
    result = collect(frozen_cluster, output)
    assert result["statuses"] == {"complete": 1, "failed": 1}
    assert result["tasks"] == result["cells"] == 2
    cluster = read(frozen_cluster / "cluster.json")
    assert read(output / "plan.json") == cluster["plan"]
    assert read(output / "cluster.json") == cluster
    environment = read(output / "environment.json")
    assert environment["kind"] == "swarm-collection"
    assert "hardware" not in environment
    assert {
        row["environment"]["hardware"]["gpus"][0]["uuid"] for row in environment["shards"].values()
    } == {
        "GPU-0",
        "GPU-1",
    }
    for cell in cluster["plan"]["cells"]:
        link = output / "cells" / cell["id"]
        assert link.is_symlink()
        assert link.readlink().is_absolute()
        assert link.resolve() == frozen_cluster / "shards" / cell["task"] / "cells" / cell["id"]
    normalized = normalize(output)
    assert len(normalized["runs"]) == 2
    assert sorted(row["status"] for row in normalized["runs"]) == ["complete", "failed"]
    assert before == {
        str(path): path.read_bytes() for path in frozen_cluster.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("state", [None, "pending", "interrupted", "cleanup_failed", "unknown"])
def test_collect_requires_terminal_result_for_every_cell(frozen_cluster, tmp_path, state):
    result = first_cell(first_shard(frozen_cluster)) / "execution/result.json"
    if state is None:
        result.unlink()
    else:
        atomic_json(result, {"status": state})
    output = tmp_path / "collected"
    with pytest.raises(ValueError):
        collect(frozen_cluster, output)
    assert not output.exists()


@pytest.mark.parametrize("state", ["pending", "missing-result", "interrupted", "cleanup_failed"])
def test_partial_collection_preserves_all_denominators_and_original_states(
    frozen_cluster, tmp_path, state
):
    source = first_cell(first_shard(frozen_cluster))
    if state == "pending":
        shutil.rmtree(source)
    elif state == "missing-result":
        (source / "execution/result.json").unlink()
    else:
        atomic_json(source / "execution/result.json", {"status": state})
    before = {str(path): path.read_bytes() for path in frozen_cluster.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        collect(frozen_cluster, tmp_path / "strict")
    output = tmp_path / "collected"
    result = collect(frozen_cluster, output, allow_partial=True)
    expected_state = "interrupted" if state == "missing-result" else state
    assert result["statuses"] == {expected_state: 1, "failed": 1}
    assert result["cells"] == 2
    assert result["linked_cells"] == (1 if state == "pending" else 2)
    assert result["partial"] is True
    cluster = read(frozen_cluster / "cluster.json")
    assert read(output / "plan.json") == cluster["plan"]
    link = output / "cells" / read(first_shard(frozen_cluster) / "plan.json")["cells"][0]["id"]
    assert link.is_symlink() is (state != "pending")
    environment = read(output / "environment.json")
    assert environment["partial"] is True
    assert environment["statuses"] == result["statuses"]
    assert sorted(row["status"] for row in normalize(output)["runs"]) == sorted(
        [expected_state, "failed"]
    )
    assert before == {
        str(path): path.read_bytes() for path in frozen_cluster.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("status", [None, "unknown", "success", "running"])
def test_partial_mode_rejects_invalid_result_status(frozen_cluster, tmp_path, status):
    atomic_json(
        first_cell(first_shard(frozen_cluster)) / "execution/result.json", {"status": status}
    )
    with pytest.raises(ValueError, match="invalid shard cell outcome"):
        collect(frozen_cluster, tmp_path / "collected", allow_partial=True)


@pytest.mark.parametrize("name", ["plan.json", "environment.json", "cluster-assignment.json"])
def test_partial_mode_still_requires_every_shard_to_be_initialized(frozen_cluster, tmp_path, name):
    (first_shard(frozen_cluster) / name).unlink()
    with pytest.raises(ValueError, match="missing regular collection input"):
        collect(frozen_cluster, tmp_path / "collected", allow_partial=True)


def test_partial_collection_remains_subject_to_report_coverage_checks(frozen_cluster, tmp_path):
    from hma.analysis.report import report

    shutil.rmtree(first_cell(first_shard(frozen_cluster)))
    output = tmp_path / "collected"
    collect(frozen_cluster, output, allow_partial=True)
    with pytest.raises(ValueError, match="incomplete runs"):
        report(output, tmp_path / "report")
    assert read(tmp_path / "report/coverage.json")["issues"]


@pytest.mark.parametrize(
    "field", ["plan", "code", "settings", "images", "data", "hardware", "identity"]
)
def test_collect_rejects_mixed_protocol_or_environment(frozen_cluster, tmp_path, field):
    shard = first_shard(frozen_cluster)
    if field == "plan":

        def change(value):
            value["cells"][0]["seconds"] += 1
            value["plan_sha256"] = fingerprint({"suite": value["suite"], "cells": value["cells"]})

        rewrite(shard / "plan.json", change)
    else:

        def change(value):
            if field == "code":
                value["code"] = fingerprint({"different": "source"})
            elif field == "settings":
                value["settings"]["cpus"] = 29
            elif field == "images":
                value["images"]["hma-agent:local"] = "sha256:different"
            elif field == "data":
                value["data"]["unassigned-task"] = "different"
            elif field == "hardware":
                value["hardware"]["gpus"][0]["name"] = "NVIDIA H200"
            rehash_environment(value)
            if field == "identity":
                value["identity"] = "invalid"

        rewrite(shard / "environment.json", change)
    output = tmp_path / "collected"
    with pytest.raises(ValueError):
        collect(frozen_cluster, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "field", ["missing-task", "duplicate-node", "global-plan-hash", "cluster-hash"]
)
def test_collect_rejects_invalid_global_manifest(frozen_cluster, tmp_path, field):
    def change(value):
        if field == "missing-task":
            value["assignments"].pop()
        elif field == "duplicate-node":
            value["assignments"][1]["node_id"] = value["assignments"][0]["node_id"]
        elif field == "global-plan-hash":
            value["plan"]["plan_sha256"] = "invalid"
        value["cluster_sha256"] = fingerprint(
            {k: v for k, v in value.items() if k != "cluster_sha256"}
        )
        if field == "cluster-hash":
            value["cluster_sha256"] = "invalid"

    rewrite(frozen_cluster / "cluster.json", change)
    with pytest.raises(ValueError):
        collect(frozen_cluster, tmp_path / "collected")


def test_collect_rejects_unrelated_campaign_even_with_identical_plan(frozen_cluster, tmp_path):
    rewrite(
        first_shard(frozen_cluster) / "cluster-assignment.json",
        lambda value: value.update(cluster_sha256="different-frozen-cluster"),
    )
    with pytest.raises(ValueError, match="different frozen cluster"):
        collect(frozen_cluster, tmp_path / "collected")


def test_collect_accepts_task_specific_harness_image_sets(frozen_cluster, tmp_path):
    cluster = read(frozen_cluster / "cluster.json")
    task = cluster["assignments"][0]["task"]
    suite = cluster["plan"]["suite"]
    harness = next(row.copy() for row in suite["experiments"] if row["harness"] == "scienceflow")
    harness.update(id="extra-harness", tasks=[task])
    suite["experiments"].append(harness)
    cluster["plan"] = plan(
        Suite.model_validate(suite), ["goal-gpt", "extra-harness"], ["mbl_01", "mbl_09"], [0]
    )
    cluster["settings"]["harness_images"] = {"scienceflow": "hma-scienceflow:local"}
    cluster["cluster_sha256"] = fingerprint(
        {k: v for k, v in cluster.items() if k != "cluster_sha256"}
    )
    atomic_json(frozen_cluster / "cluster.json", cluster)
    for assignment in cluster["assignments"]:
        shard = frozen_cluster / "shards" / assignment["task"]
        cells = [cell for cell in cluster["plan"]["cells"] if cell["task"] == assignment["task"]]
        atomic_json(
            shard / "plan.json",
            {
                "schema_version": 1,
                "suite": suite,
                "cells": cells,
                "plan_sha256": fingerprint({"suite": suite, "cells": cells}),
                "max_gpu_hours": sum(cell["seconds"] for cell in cells) / 3600,
            },
        )
        atomic_json(
            shard / "cluster-assignment.json",
            assignment | {"cluster_sha256": cluster["cluster_sha256"]},
        )
        environment = read(shard / "environment.json")
        environment["settings"] = cluster["settings"]
        if assignment["task"] == task:
            environment["images"]["hma-scienceflow:local"] = "sha256:scienceflow"
            cell = next(cell for cell in cells if cell["experiment"] == "extra-harness")
            atomic_json(
                shard / "cells" / cell["id"] / "execution/result.json", {"status": "failed"}
            )
        rehash_environment(environment)
        atomic_json(shard / "environment.json", environment)
    output = tmp_path / "collected"
    result = collect(frozen_cluster, output)
    assert result["cells"] == 3
    assert (
        read(output / "environment.json")["images"]["hma-scienceflow:local"] == "sha256:scienceflow"
    )


def test_collect_checks_cell_data_evidence_and_existing_lock(frozen_cluster, tmp_path):
    shard = first_shard(frozen_cluster)
    cell = first_cell(shard)
    rewrite(cell / "cell.json", lambda value: value.update(data_manifest_sha256="different"))
    with pytest.raises(ValueError, match="cell evidence"):
        collect(frozen_cluster, tmp_path / "collected")
    (shard / ".campaign.lock").unlink()
    with pytest.raises(ValueError, match="missing shard campaign lock"):
        collect(frozen_cluster, tmp_path / "collected")
    assert not (shard / ".campaign.lock").exists()


def test_collect_refuses_active_lock_then_releases_other_locks(frozen_cluster, tmp_path):
    shards = sorted((frozen_cluster / "shards").iterdir())
    with (shards[1] / ".campaign.lock").open("r+") as active:
        fcntl.flock(active.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="still running or locked"):
            collect(frozen_cluster, tmp_path / "collected")
        with (shards[0] / ".campaign.lock").open("r+") as available:
            fcntl.flock(available.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert collect(frozen_cluster, tmp_path / "collected")["cells"] == 2


def test_collector_uses_nfs_compatible_shared_locks_on_read_only_descriptors(
    frozen_cluster, tmp_path, monkeypatch
):
    from hma.repro import swarm_collect

    original = fcntl.flock
    calls = []

    def nfs_lock(descriptor, operation):
        mode = fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
        assert mode == os.O_RDONLY
        assert operation == fcntl.LOCK_SH | fcntl.LOCK_NB
        calls.append(descriptor)
        return original(descriptor, operation)

    # An existing reader can coexist with collection. A real campaign's exclusive
    # lock still blocks it, as the separate active-worker test verifies.
    with (first_shard(frozen_cluster) / ".campaign.lock").open("r") as reader:
        original(reader.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        monkeypatch.setattr(swarm_collect.fcntl, "flock", nfs_lock)
        collect(frozen_cluster, tmp_path / "collected")
    assert len(calls) == 2


@pytest.mark.parametrize("kind", ["directory", "file", "broken-symlink", "inside-shard"])
def test_collect_refuses_existing_output_or_source_mutation(frozen_cluster, tmp_path, kind):
    output = tmp_path / "collected"
    if kind == "directory":
        output.mkdir()
    elif kind == "file":
        output.write_text("preserve")
    elif kind == "broken-symlink":
        output.symlink_to(tmp_path / "missing")
    else:
        output = first_shard(frozen_cluster) / "collected"
    with pytest.raises(ValueError):
        collect(frozen_cluster, output)
    if kind == "file":
        assert output.read_text() == "preserve"
    elif kind == "broken-symlink":
        assert output.is_symlink()


def test_collection_failure_does_not_publish_partial_directory(
    frozen_cluster, tmp_path, monkeypatch
):
    from hma.repro import swarm_collect

    calls = []

    def fail_midway(path, value):
        calls.append(path)
        if len(calls) == 3:
            raise OSError("synthetic write failure")
        atomic_json(path, value)

    monkeypatch.setattr(swarm_collect, "atomic_json", fail_midway)
    output = tmp_path / "collected"
    with pytest.raises(OSError, match="synthetic write"):
        collect(frozen_cluster, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".collected.*"))


def test_collection_retains_shard_exclusions_for_normal_artifact_validation(
    frozen_cluster, tmp_path
):
    shard = first_shard(frozen_cluster)
    cell_id = read(shard / "plan.json")["cells"][0]["id"]
    exclusion = {cell_id: {"reason": "fixture evidence", "artifact_sha256": "a" * 64}}
    atomic_json(shard / "exclusions.json", exclusion)
    output = tmp_path / "collected"
    collect(frozen_cluster, output)
    assert read(output / "exclusions.json") == exclusion
    # Collection preserves the claim; export still refuses a nonexistent candidate.
    with pytest.raises(ValueError, match="final candidate digest"):
        normalize(output)
