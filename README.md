# HMA: reproduce the paper

Fresh runs of the paper's experiments: 75 tasks and 27 configurations.

## Configuration

Each task uses **1 NVIDIA A10, 30 vCPUs, and 220 GiB RAM**.

```bash
cp configs/local.example.json configs/local.json
```

Set `data_root` to your prepared data directory and fill in your API keys and
base URLs. These fields are blank in the template; experiment settings are
preconfigured.

## Requirements and preparation

- Download and prepare the 75 MLE-bench datasets. See [data preparation](docs/local-run.md#5-prepare-and-verify-data).
- Prepare **75 A10 machines with Docker Swarm**, one benchmark task per machine.
- Set up Python 3.12 and the dependencies in [requirements-repro.lock](requirements-repro.lock).

See the [setup guide](docs/swarm.md) for data paths, machines, and Docker environments.

## Run

After completing the cluster setup, launch from the manager:

```bash
python -m hma.repro.swarm deploy --root /srv/hma/runs/paper \
  --config configs/swarm.local.json --docker-gid DOCKER_GID
```

Each machine runs its task's configurations and repeats sequentially.

## Results

After all workers finish:

```bash
python -m hma.repro.swarm collect --root /srv/hma/runs/paper --output /srv/hma/runs/paper-report
python -m hma.repro.cli report --run-root /srv/hma/runs/paper-report --output outputs/rerun
```

[Experiment map](docs/experiments.md) · [Single-machine guide](docs/local-run.md) ·
[AGENTS.md](AGENTS.md) · [Validation and data limitations](docs/verification.md)
