# HMA: reproduce the paper

Fresh runs of the paper's experiments: 75 tasks and 27 configurations.

## Configuration

Each task uses **1 NVIDIA A10, 30 vCPUs, and 220 GiB RAM**.

```bash
cp configs/local.example.json configs/local.json
```

Set `data_root` to `/srv/hma/data` and fill in your API keys and base URLs.
These fields are blank in the template; other experiment settings are preconfigured.

## Requirements and preparation

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements-repro.lock
pip install --no-deps -e .
hma-repro build
hma-swarm images --config configs/local.json --repository ghcr.io/ORG/hma \
  --tag repro-v1 --output configs/swarm.local.json
```

Replace `ORG` with your registry namespace. After configuring Kaggle access and
accepting the competition rules, prepare the data:

```bash
hma-repro data prepare --config configs/swarm.local.json --suite paper
hma-repro data verify --config configs/swarm.local.json --suite paper
```

## Run with Docker Swarm

We recommend **75 A10 machines**, one benchmark task per machine. Each machine
runs that task's configurations and repeats sequentially.
[Set up the cluster](docs/swarm.md), then run on the manager:

```bash
hma-swarm plan --config configs/swarm.local.json --root /srv/hma/runs/paper
```

Pull the images on each execution node:

```bash
while IFS= read -r image; do docker pull "$image"; done < /srv/hma/runs/paper/images.txt
```

Launch from the manager using the IDs from your cluster setup:

```bash
hma-swarm deploy --root /srv/hma/runs/paper --config configs/swarm.local.json \
  --user 1000:1000 --docker-gid DOCKER_GID
```

## Results

After all workers finish:

```bash
hma-swarm collect --root /srv/hma/runs/paper --output /srv/hma/runs/paper-report
hma-repro report --run-root /srv/hma/runs/paper-report --output outputs/rerun
```

Outputs include CSV tables and PDF/SVG/PNG figures.

[Experiment map](docs/experiments.md) · [Single-machine guide](docs/local-run.md) ·
[AGENTS.md](AGENTS.md) · [Validation and data limitations](docs/verification.md)
