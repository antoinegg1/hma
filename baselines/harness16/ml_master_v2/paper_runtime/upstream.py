"""Small, recorded compatibility patches around unchanged upstream algorithms."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import runpy
import sys
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, digest


def replace_once(path: Path, before: str, after: str) -> None:
    """Reject source drift instead of silently applying a partial patch."""
    text = path.read_text()
    if text.count(before) != 1:
        raise ValueError(f"upstream patch anchor mismatch: {path.name}")
    path.write_text(text.replace(before, after))


def credentials(role: str) -> tuple[str, str]:
    """Read provider credentials only at execution time, never from CLI argv."""
    prefix = "PAPER_" + role.upper()
    key = os.environ.get(prefix + "_API_KEY", "")
    url = os.environ.get(prefix + "_BASE_URL", "")
    if not key or not url:
        raise ValueError(f"{prefix}_API_KEY and {prefix}_BASE_URL are required")
    return key, url


def write_yaml(path: Path, value: Any) -> None:
    """Restrict resolved credential-bearing configuration to the current user."""
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    path.chmod(0o600)


def configure(
    root: Path, workspace: Path, contract: dict[str, Any], seconds: int, port: int
) -> list[str]:
    """Configure the selected upstream entry point and return nonsecret argv."""
    source = contract["source"]
    code_key, code_url = credentials("code")
    feedback_key, feedback_url = credentials("feedback")
    run_root = root.parent
    public = workspace / "input"
    # Describe the task separately from the harness's finalization instructions.
    description = run_root / "description.md"
    description.write_text(
        (public / "description.md").read_text()
        + "\n"
        + (workspace / ".flowbench/task.md").read_text()
        if (workspace / ".flowbench/task.md").exists()
        else (public / "description.md").read_text()
    )
    if source == "ml_master_v1":
        # Preserve these clauses from dataset/full_instructions/instruction_template.txt.
        # The benchmark's local task description replaces that original template.
        with description.open("a") as handle:
            handle.write(
                "\nYou must solve the task by training a model and running the model on the test set to produce a submission file. "
                "You may not hand-label the data to produce a submission file.\n"
                "You are not allowed to view others' solutions to this competition. "
                "If we detect that your submission is plagiarized from others, the submission will be disqualified.\n"
            )
    agent_code = {
        "model": contract["code_model"],
        "temp": contract["temperature"],
        "base_url": code_url,
        "api_key": code_key,
    }
    agent_feedback = {
        "model": contract["feedback_model"],
        "temp": contract.get("feedback_temperature", contract["temperature"]),
        "base_url": feedback_url,
        "api_key": feedback_key,
    }
    cpus = sorted(os.sched_getaffinity(0))[: contract["cpu"]]
    if not cpus:
        raise ValueError("no available CPUs")
    if source in {"mlevolve", "ml_master_v1"}:
        path = root / (
            "config/config.yaml" if source == "mlevolve" else "utils/config_mcts.yaml"
        )
        cfg = yaml.safe_load(path.read_text())
        cfg.update(
            data_dir=str(public),
            dataset_dir=str(public),
            desc_file=str(description),
            exp_name=contract["slug"],
            log_dir=str(run_root / "runs"),
            workspace_dir=str(run_root / "runs"),
            preprocess_data=False,
            copy_data=False,
            start_cpu_id=cpus[0],
            cpu_number=len(cpus),
        )
        cfg["agent"].update(
            code=agent_code, feedback=agent_feedback, time_limit=seconds
        )
        cfg["exec"]["timeout"] = min(cfg["exec"]["timeout"], seconds)
        if source == "mlevolve":
            cfg.update(exp_id=contract["slug"])
            cfg["agent"]["seed"] = contract["agent_seed"]
            cfg["agent"]["search"]["num_gpus"] = contract["gpu"]
            cfg["agent"]["memory_embedding_device"] = (
                "cuda" if contract["gpu"] else "cpu"
            )
            if contract.get("require_mechanisms"):
                cfg["agent"]["memory_embedding_model_path"] = (
                    "/opt/paper/models/bge-base-en-v1.5"
                )
            replace_once(
                root / "engine/validation/format_client.py",
                'os.getenv("GRADING_SERVER_PORT", "5005")',
                f'os.getenv("GRADING_SERVER_PORT", "{port}")',
            )
            entry = "run.py"
        else:
            cfg["agent"]["steerable_reasoning"] = contract.get(
                "steerable_reasoning", True
            )
            replace_once(
                root / "utils/server_utils.py",
                '"http://127.0.0.1:5001"',
                f'"http://127.0.0.1:{port}"',
            )
            entry = "main_mcts.py"
        write_yaml(path, cfg)
        return [str(root / entry)]
    if source == "scienceflow":
        cfg_path = root / "paper-config.yaml"
        write_yaml(
            cfg_path,
            {
                "include": [str(root / "scienceflow/config/default.yaml")],
                "agent": {"code": agent_code, "feedback": agent_feedback},
            },
        )
        manifest = {
            "max_concurrent": 1,
            "time_limit": seconds,
            "resume": False,
            "resume_budget_policy": "remaining",
            "lnr": {
                "wall_clock_budget_sec": max(1, seconds - 5),
                "num_workers": contract.get("workers", 2),
                "seed": contract["agent_seed"],
                "resource_control_mode": "resource_smart_llm",
            },
            "defaults": {
                "config": str(cfg_path),
                "workspace_base": str(run_root / "runs"),
            },
            "tasks": [
                {
                    "exp_id": contract["slug"],
                    "run_id": f"paper-seed-{contract['agent_seed']}",
                    "input_data_dir": str(public),
                    "cpu_list": ",".join(map(str, cpus)),
                    "gpu_list": ",".join(str(i) for i in range(contract["gpu"]))
                    if contract["gpu"]
                    else "cpu",
                }
            ],
        }
        path = root / "paper-manifest.yaml"
        write_yaml(path, manifest)
        return ["-m", "scienceflow.cli", "parallel", "-m", str(path), "-j", "1"]
    path = root / "configs/ml_master_2/deepseek-v3.2-example.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg.update(
        competition_id=contract["slug"],
        is_lower_better=contract["lower_is_better"],
        data_root=str(public),
        grading_servers=[f"http://127.0.0.1:{port}"],
    )
    original_llm = dict(cfg["llm"][cfg["llm"]["default"]])
    cfg["llm"] = {
        "default": "paper_code",
        "paper_code": {
            **original_llm,
            "model": contract["code_model"],
            "api_key": code_key,
            "base_url": code_url,
            "temperature": contract["temperature"],
        },
        "paper_feedback": {
            **original_llm,
            "model": contract["feedback_model"],
            "api_key": feedback_key,
            "base_url": feedback_url,
            "temperature": contract["temperature"],
        },
    }
    for name, agent in cfg["agents"].items():
        agent["llm"] = (
            "paper_feedback"
            if name in {"knowledge_promotion", "wisdom_promotion"}
            else "paper_code"
        )
    cfg["session"]["type"] = "local"
    cfg["session"]["local"].update(
        working_dir=str(run_root / "runs"),
        gpu_devices=[str(i) for i in range(contract["gpu"])],
        cpu_devices=cpus,
        symlinks={str(public): "input"},
    )
    if contract["profile"].endswith("no_prior"):
        replace_once(
            root / "playground/ml_master_2/core/exp/prefetch_exp.py",
            "        if self.prefetch_agent:\n",
            "        return data_knowledge, model_knowledge, prefetch_descriptor\n\n        if self.prefetch_agent:\n",
        )
    else:
        prior = workspace / contract["prior_dir"]
        for old, new in [
            (
                'os.path.join(os.getcwd(), "playground/ml_master_2/example_wisdom")',
                repr(str(prior)),
            ),
            (
                'os.path.join(os.getcwd(), "playground/ml_master_2/example_wisdom/db.json")',
                repr(str(prior / "db.json")),
            ),
        ]:
            replace_once(root / "playground/ml_master_2/core/playground.py", old, new)
        key, url = credentials("embedding")
        cfg["embedding"]["openai"].update(
            api_key=key,
            base_url=url,
            model=contract.get("embedding_model", "text-embedding-3-large"),
        )
        os.environ.update(OPENAI_API_KEY=key, GPT_BASE_URL=url)
    replace_once(
        root / "playground/ml_master_2/core/utils/watch_dog.py",
        "RUN_TIMEOUT_SECONDS = 86400",
        f"RUN_TIMEOUT_SECONDS = {seconds}",
    )
    write_yaml(path, cfg)
    return [
        str(root / "run.py"),
        "--agent",
        "ml_master_2",
        "--config",
        str(path),
        "--run-dir",
        str(run_root / "runs"),
        "--task",
        str(description),
    ]


def install_mlevolve_audit(output: Path) -> None:
    """Observe real mechanism calls and reject a disabled required mechanism."""
    import functools
    import threading
    import time
    import uuid
    from collections.abc import Iterator
    from datetime import UTC, datetime

    from agents import (
        aggregation_agent,
        code_review_agent,
        evolution_agent,
        fusion_agent,
    )
    from agents.memory.global_memory import GlobalMemoryLayer
    from engine import coldstart
    from engine.agent_search import AgentSearch
    from engine.coldstart import knowledge
    from openai import Stream
    from openai.resources.chat.completions import Completions

    lock = threading.Lock()

    def event(kind: str, **fields: Any) -> None:
        row = {"at_utc": datetime.now(UTC).isoformat(), "event": kind, **fields}
        with lock, output.open("a") as stream:
            stream.write(json.dumps(row, default=str) + "\n")

    def observe(owner: Any, name: str, label: str) -> None:
        original = getattr(owner, name)

        @functools.wraps(original)
        def observed(*args: Any, **kwargs: Any) -> Any:
            event(label + "_started")
            try:
                result = original(*args, **kwargs)
            except Exception as error:
                event(label + "_error", error_type=type(error).__name__)
                raise
            event(
                label + "_returned",
                result_count=len(result) if isinstance(result, list) else None,
                saved=result if label == "memory_save" else None,
                memory_records=len(args[0].records) if label == "memory_save" else None,
            )
            return result

        setattr(owner, name, observed)

    original_guidance = knowledge.build_guidance_description

    def guidance(*args: Any, **kwargs: Any) -> str:
        result = original_guidance(*args, **kwargs)
        if (
            result == "None model"
            and args[0].exp_id == "multi-modal-gesture-recognition"
        ):
            event(
                "knowledge_no_match",
                task=args[0].exp_id,
                reason="upstream category Others has no model templates; original draft fallback retained",
            )
            return result
        if not result or result == "None model":
            raise RuntimeError("required cold-start knowledge is unavailable")
        event("knowledge_loaded", characters=len(result))
        return result

    knowledge.build_guidance_description = guidance
    coldstart.build_guidance_description = guidance
    original_init = AgentSearch.__init__

    def initialize(agent: Any, *args: Any, **kwargs: Any) -> None:
        original_init(agent, *args, **kwargs)
        if agent.global_memory is None:
            raise RuntimeError("required global memory failed to initialize")
        cfg = agent.acfg
        if not (
            cfg.use_evolution
            and cfg.use_fusion
            and cfg.use_global_memory
            and cfg.use_aggregation
            and cfg.check_data_leakage
            and cfg.use_diff_mode
            and cfg.use_stepwise_generation
            and agent.cfg.coldstart.use_coldstart
        ):
            raise RuntimeError("required search mechanism is disabled")
        if cfg.initial_drafts != 3 or cfg.search.parallel_search_num != 3:
            raise RuntimeError("required three-way search is misconfigured")
        event(
            "mechanisms_initialized",
            parallel_search_num=3,
            initial_drafts=3,
            memory_directory=str(agent.cfg.workspace_dir / "global_memory"),
            fusion_eligible_after_sec=cfg.time_limit / 2,
            fusion_probability=cfg.fusion_vs_evolution_prob,
        )

    AgentSearch.__init__ = initialize
    for owner, name, label in (
        (GlobalMemoryLayer, "save_node", "memory_save"),
        (GlobalMemoryLayer, "retrieve_similar_records", "memory_retrieve"),
        (code_review_agent, "run", "code_review"),
        (fusion_agent, "run", "cross_branch_fusion"),
        (evolution_agent, "run", "intra_branch_evolution"),
        (aggregation_agent, "run", "multi_branch_aggregation"),
    ):
        observe(owner, name, label)
    original_create = Completions.create

    @functools.wraps(original_create)
    def create(client: Any, *args: Any, **kwargs: Any) -> Any:
        body = kwargs.get("extra_body") or {}
        if (
            kwargs.get("model") != "deepseek-v4-flash"
            or body.get("reasoning_effort") != "max"
            or body.get("thinking") != {"type": "enabled"}
        ):
            raise RuntimeError(
                "model or thinking parameters violate the frozen contract"
            )
        call_id = uuid.uuid4().hex
        started = time.monotonic()
        event(
            "llm_request",
            call_id=call_id,
            model=kwargs["model"],
            reasoning_effort="max",
            stream=bool(kwargs.get("stream")),
            tools=bool(kwargs.get("tools")),
            response_format=kwargs.get("response_format"),
            max_tokens=kwargs.get("max_tokens"),
        )
        try:
            result = original_create(client, *args, **kwargs)
        except Exception as error:
            event(
                "llm_error",
                call_id=call_id,
                error_type=type(error).__name__,
                elapsed_sec=time.monotonic() - started,
            )
            raise
        usage = getattr(result, "usage", None)
        if isinstance(result, Stream):
            result._paper_call_id = call_id
        event(
            "llm_response",
            call_id=call_id,
            elapsed_sec=time.monotonic() - started,
            usage=usage.model_dump() if usage else None,
            finish_reasons=[c.finish_reason for c in getattr(result, "choices", [])],
        )
        return result

    Completions.create = create
    original_iter = Stream.__iter__

    def iterate(stream: Any) -> Iterator[Any]:
        usage, reasons = None, []
        for chunk in original_iter(stream):
            if getattr(chunk, "usage", None):
                usage = chunk.usage.model_dump()
            reasons.extend(
                c.finish_reason
                for c in getattr(chunk, "choices", [])
                if c.finish_reason
            )
            yield chunk
        event(
            "llm_stream_complete",
            call_id=getattr(stream, "_paper_call_id", None),
            usage=usage,
            finish_reasons=reasons,
        )

    Stream.__iter__ = iterate


def main() -> None:
    """Run upstream in a separate process group owned by the deadline supervisor."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    contract = json.loads((args.workspace / ".flowbench/paper.json").read_text())
    random.seed(contract["agent_seed"])
    try:
        import numpy as np

        np.random.seed(contract["agent_seed"])
    except ImportError:
        pass
    argv = configure(args.root, args.workspace, contract, args.seconds, args.port)
    atomic_json(
        args.root.parent / "adapter-files.json",
        {str(p.relative_to(args.root)): digest(p) for p in args.root.rglob("*.py")},
    )
    os.chdir(args.root)
    sys.path.insert(0, str(args.root))
    sys.argv = argv
    if args.check_config:
        if contract["source"] in {"mlevolve", "ml_master_v1"}:
            module = (
                "config" if contract["source"] == "mlevolve" else "utils.config_mcts"
            )
            config_module = importlib.import_module(module)
            cfg = config_module.load_cfg()
            config_module.prep_agent_workspace(cfg)
            runpy.run_path(argv[0], run_name="__paper_import_check__")
        elif contract["source"] == "scienceflow":
            importlib.import_module("scienceflow.cli")
            importlib.import_module("scienceflow.config.settings").load_cfg(
                args.root / "paper-config.yaml", cli_args=False
            )
            importlib.import_module("scienceflow.core.parallel_runner").ParallelRunner(
                args.root / "paper-manifest.yaml"
            )
        else:
            module = importlib.import_module("playground.ml_master_2.core.playground")
            playground = module.MLMaster2Playground(
                config_path=args.root / "configs/ml_master_2/deepseek-v3.2-example.yaml"
            )
            playground.set_run_dir(args.root.parent / "runs", task_id="probe")
            try:
                playground.setup()
            finally:
                if playground.session is not None:
                    playground.session.close()
        print(
            json.dumps(
                {"profile": contract["profile"], "status": "config_import_passed"}
            )
        )
        return
    if contract["source"] == "mlevolve" and contract.get("require_mechanisms"):
        install_mlevolve_audit(args.root.parent / "mechanisms.jsonl")
    if argv[0] == "-m":
        sys.argv = [argv[1], *argv[2:]]
        runpy.run_module(argv[1], run_name="__main__")
    else:
        runpy.run_path(argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
