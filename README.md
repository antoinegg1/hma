# HMA: reproduce the paper

Run all experiments from fresh model sessions and MLE-bench data. The full plan
has 75 tasks, 27 configurations, and 3,397 task-runs; analysis uses your new results.

## 1. Configure

Each task uses **1 NVIDIA A10, 30 vCPUs, and 220 GiB RAM**, shared by its
alternating agents. Provision at least 32 vCPUs and 256 GiB
per machine for the separate 2-CPU/16-GiB evaluator and controller overhead.
The H200 listed in paper Table 5 describes a cited baseline's original hardware.

Use Linux x86-64, rootful Docker, NVIDIA Container Toolkit, and a driver supporting
CUDA 12.8. Mount the same shared POSIX filesystem at `/srv/hma` on every node;
it must support cross-node `flock`. Use the same non-root UID/GID and Docker
socket group GID everywhere. See the [Swarm setup](docs/swarm.md) for commands.

```bash
cp configs/local.example.json configs/local.json
chmod 600 configs/local.json
```

Set `data_root` to `/srv/hma/data`, `gpus` to `["0"]`, and keep
`expected_gpu_model: "NVIDIA A10"`, `cpus: 30`, `memory: "220g"`.
**API keys and base URLs are blank in the template.** Fill them in the ignored
`configs/local.json`; deployment delivers this configuration as a Docker secret.
The publication command below creates a private config with pinned image digests.

## 2. Install requirements and prepare

On the manager, from this checkout:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-repro.lock
python -m pip install --no-deps -e .
hma-repro doctor --offline
hma-repro build
hma-swarm images --config configs/local.json --repository ghcr.io/ORG/hma \
  --tag repro-v1 --output configs/swarm.local.json
```

Replace `ORG`, authenticate to your registry, and make sure every node can pull
these images. Configure Kaggle access and accept competition rules, then prepare
the shared data once:

```bash
hma-repro data prepare --config configs/swarm.local.json --suite paper
hma-repro data verify --config configs/swarm.local.json --suite paper
```

## 3. Execute on 75 nodes

We recommend **75 A10 machines managed by Docker Swarm**, with one benchmark task
assigned to each machine.

After the [Swarm setup](docs/swarm.md), plan on its manager:

```bash
hma-swarm plan --config configs/swarm.local.json --root /srv/hma/runs/paper
```

On **every execution node**, authenticate to the registry if private, then pull:

```bash
while IFS= read -r image; do docker pull "$image"; done < /srv/hma/runs/paper/images.txt
```

Back on the manager, substitute your numeric user/group and Docker socket group:

```bash
hma-swarm deploy --root /srv/hma/runs/paper --config configs/swarm.local.json \
  --user 1000:1000 --docker-gid DOCKER_GID
docker service ls
# Inspect the service name printed by deploy:
docker service ps --no-trunc SERVICE_NAME
```

Each node runs all configurations/repeats for its assigned task sequentially.
The 75-node mapping is frozen; completed or failed attempts are never silently
rerun. The full plan declares up to 21,767.6 GPU hours, plus preparation overhead.

## 4. Collect and report

After the workers finish:

```bash
hma-swarm collect --root /srv/hma/runs/paper --output /srv/hma/runs/paper-report
hma-repro report --run-root /srv/hma/runs/paper-report --output outputs/rerun
```

Reports contain new CSV tables and PDF/SVG/PNG figures. Incomplete evidence is
reported explicitly. See [experiment coverage](docs/experiments.md), the
[detailed single-host runbook](docs/local-run.md), and [AGENTS.md](AGENTS.md).

**Verification:** actual Docker/GPU/API and multi-node Swarm execution have not
been validated; see [test status](docs/verification.md). The repository uses
pinned public answer keys; [the manuscript's corrected-key difference](docs/local-run.md#5-prepare-and-verify-data)
remains unresolved. No historical scores are inserted into new results.
