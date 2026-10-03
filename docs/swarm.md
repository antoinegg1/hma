# Run the paper on 75 A10 nodes

The recommended full campaign assigns each of the 75 MLE-bench tasks to one
machine. That machine executes every selected configuration and repeat for its
task sequentially. This distributes the full 3,397-cell matrix without changing
per-task budgets, model order, caps, or repeat counts.

Actual multi-node Swarm execution has not yet been validated. Complete a real
native/harness smoke using the [local runbook](local-run.md#6-run-small-end-to-end-checks)
before committing to the full campaign; offline tests do not exercise GPUs or APIs.

## Host and storage setup

Use 75 Linux x86-64 machines, each with one NVIDIA A10. The per-task agent limits
are 30 vCPUs and 220 GiB RAM; provision at least 32 vCPUs and 256 GiB per machine
to accommodate the private 2-CPU/16-GiB evaluator and controller overhead.
Install rootful Docker Engine and NVIDIA Container Toolkit on every node, with a
driver supporting CUDA 12.8. Use a Docker Engine version supporting `global-job`
services. This workflow uses local `/var/run/docker.sock`; rootless Docker and
remote Docker daemons are unsupported.

Mount the **same shared POSIX filesystem at `/srv/hma` on every node**. It must
support cross-node `flock`, atomic rename, and normal file permissions. Use one
fixed non-root UID/GID for the controller and actors across all nodes; this user
must own the run directories. The Docker socket's group must have a common numeric
GID on all nodes. `--docker-gid` adds that group to the controller service.
Ensure the cluster can read prepared data and write `/srv/hma/runs` as that user.
Data, image layers, retained submissions, and workspaces need multiple TB of disk.

Nodes must reach the registry and model endpoints. Docker requires TCP 2377,
TCP/UDP 7946, and UDP 4789 between the relevant Swarm hosts; see
[Docker's network prerequisites](https://docs.docker.com/engine/swarm/swarm-tutorial/).

## Create and label the swarm

On the manager, replacing the example IP with its reachable address:

```bash
docker swarm init --advertise-addr MANAGER_IP
docker swarm join-token worker
```

Run the printed `docker swarm join --token ... MANAGER_IP:2377` command on each
other machine. See Docker's [create](https://docs.docker.com/engine/swarm/swarm-tutorial/create-swarm/)
and [join](https://docs.docker.com/engine/swarm/swarm-tutorial/add-nodes/)
instructions. A manager with an A10 can also count among the 75 execution nodes;
an additional dedicated manager need not be labeled.

On the manager, label each execution node, substituting its node ID or hostname:

```bash
docker node ls
docker node update --label-add hma.worker=true NODE_ID
```

Repeat the label command for exactly 75 nodes. All must be `Ready` and `Active`.
Keep these node identities and labels fixed throughout the campaign. Planning
requires exactly 75 eligible nodes by default (`--expected-nodes 75`).

## Images and data

Install the locked requirements on the manager as shown in the [README](../README.md).
Edit the supplied `configs/local.json` directly: set `data_root` to
`/srv/hma/data` and configure all required providers. Keep the
per-task defaults: `gpus: ["0"]`, `expected_gpu_model: "NVIDIA A10"`,
`cpus: 30`, `memory: "220g"`.

Build the agent, evaluator, and three harness images once:

```bash
python -m hma.repro.cli build
python -m hma.repro.swarm images --config configs/local.json --repository ghcr.io/ORG/hma \
  --tag repro-v1 --output configs/swarm.local.json
```

Replace `ORG` and log into your registry before publishing. `python -m hma.repro.swarm images`
pushes all five built images and writes their `NAME@sha256:DIGEST` references to
`configs/swarm.local.json`. This ignored, mode-0600 file preserves the provider
values from the input configuration. Use it for all subsequent commands.
The configured `agent_image` also supplies the controller; deployment has no
separate image override. Do not rebuild between planning and running.

Set `KAGGLE_API_TOKEN` or the manager user's `~/.kaggle/access_token`, accept
competition rules, and prepare the shared dataset once:

```bash
python -m hma.repro.cli data prepare --config configs/swarm.local.json --suite paper
python -m hma.repro.cli data verify --config configs/swarm.local.json --suite paper
```

Provider keys and URLs must stay blank in the committed `configs/local.json`.
Use uncommitted local values or the existing [provider environment variables](local-run.md#3-local-configuration-and-credentials).
The generated `configs/swarm.local.json` remains ignored and mode 0600.
Deployment packages the configuration as
a Docker secret rather than putting credentials in service environment variables.
The public plan excludes credentials. Keep raw run directories private: native
tools can persist credential-bearing state. Kaggle credentials remain on the
preparation host. See the [data and answer-key scope](local-run.md#5-prepare-and-verify-data).

## Plan and deploy

Run these commands on the manager, with the shared filesystem mounted:

```bash
python -m hma.repro.swarm plan --config configs/swarm.local.json --root /srv/hma/runs/paper
```

Planning freezes the task-to-node mapping, full scientific plan, source identity,
and non-secret configuration. `data_root` and the campaign root must be absolute
paths beneath `--shared-root` (default `/srv/hma`). Planning does not run models.

The plan writes `images.txt` with the five pinned references. On **every execution
node**, run `docker login` for a private registry, then pre-pull all images:

```bash
while IFS= read -r image; do docker pull "$image"; done < /srv/hma/runs/paper/images.txt
```

Workers use these cached images; sibling actor/evaluator containers do not inherit
the manager's registry credentials. Worker hosts do not need a Python/HMA install.
Back on the manager, launch with the numeric IDs configured consistently across
the cluster:

```bash
python -m hma.repro.swarm deploy --root /srv/hma/runs/paper --config configs/swarm.local.json \
  --user 1000:1000 --docker-gid DOCKER_GID
```

`--user` defaults to `1000:1000`; substitute your chosen non-root IDs. Set
`--docker-gid` to the common socket group ID. Deployment uses registry auth for
the controller service itself and starts paid model work. Its default name is
`hma-<hash12>`; `--name` can choose another name. Deployment uses `--detach` and
returns the service ID immediately; it does not wait for the long experiments.

The service uses Docker's [global-job mode](https://docs.docker.com/reference/cli/docker/service/create/#running-as-a-job),
[host networking](https://docs.docker.com/engine/network/drivers/host/), and no automatic restart. One
controller per labeled node opens that node's Docker socket and creates the
ordinary actor/evaluator containers locally. GPU access belongs to these local
containers; no Swarm GPU flag or default NVIDIA runtime is needed when ordinary
Docker GPU containers already work through the toolkit. Each worker
probes the selected GPU through a Docker helper and checks the A10 expectation
before running its frozen shard at `shards/<task>`.

Use the service name printed by deployment:

```bash
docker service ls
docker service ps --no-trunc SERVICE_NAME
docker service logs SERVICE_NAME
```

Do not use `docker service update --force` to retry scientific results, relabel
nodes mid-run, or start another campaign on the same GPU. Use dedicated execution
nodes. A worker rejects a node outside the frozen assignment. Preserve failures
and partial outputs. After the original job has stopped on a node, continue only
its never-started cells using an explicitly named new job:

```bash
python -m hma.repro.swarm deploy --root /srv/hma/runs/paper --config configs/swarm.local.json \
  --user 1000:1000 --docker-gid DOCKER_GID \
  --resume --name hma-paper-resume-1 --node-id ORIGINAL_NODE_ID
```

Use an assigned node ID from the frozen manifest. Omit `--node-id` to resume on
all original nodes once their jobs have stopped. Resume requires a new `--name`;
it never restarts the old service. Completed, failed, and interrupted attempts
are not retried; code, settings, images, data, and recorded hardware must remain
unchanged. Inspect and clean up any containers left by the stopped original
controller before recovery; the worker refuses to overlap with them. `SIGTERM`
supports orderly cleanup. A `SIGKILL` can leave actor/evaluator containers behind,
and resume refuses to proceed until the original owner's leftovers are explicitly
cleaned up. Preserve its logs and failure records; cleanup must not erase or
rewrite the historical error.

## Optional one-node pilot

Before labeling the remaining nodes, use one labeled A10 to exercise Swarm with
a single task/repeat:

```bash
python -m hma.repro.cli plan --experiment goal-gpt --task leaf-classification --repeat 0 \
  --output outputs/swarm-pilot.json
python -m hma.repro.swarm plan --config configs/swarm.local.json --root /srv/hma/runs/pilot \
  --plan-file outputs/swarm-pilot.json --expected-nodes 1
```

Pre-pull the pilot's `images.txt` on that node and deploy using its pilot root.
This retains the full six-hour goal budget and calls a real model. For shorter
local checks, use `python -m hma.repro.cli smoke` in the [local runbook](local-run.md).
Create a separate full-paper root after completing the pilot and labeling all
75 execution nodes.

## Collect and analyze

After all workers finish, create a separate aggregate report root. Collection
requires **every planned cell to have a `complete` or `failed` terminal result**;
it rejects running/locked shards, missing cells, and interrupted/nonterminal
cells. Failed results remain failures and retain their planned denominators.
Collection acquires shared file locks compatible with NFS; those locks conflict
with each active worker's exclusive campaign lock.

Optional regrading runs on each original shard before collection, not on the
aggregate root. For example, substitute a task ID from the manifest:

```bash
python -m hma.repro.cli grade --run-root /srv/hma/runs/paper/shards/TASK_ID
```

The private evaluator already records scores, so regrading is unnecessary unless
you want an independent check. Then collect and report:

```bash
python -m hma.repro.swarm collect --root /srv/hma/runs/paper --output /srv/hma/runs/paper-report
python -m hma.repro.cli report --run-root /srv/hma/runs/paper-report --output outputs/rerun
```

For diagnosis after the workers stop, partial collection also accepts pending,
interrupted, and `cleanup_failed` cells in already initialized shards:

```bash
python -m hma.repro.swarm collect --root /srv/hma/runs/paper \
  --output /srv/hma/runs/paper-partial --allow-partial
python -m hma.repro.cli report --run-root /srv/hma/runs/paper-partial \
  --output outputs/rerun-partial --allow-partial
```

Every assigned shard must still have a matching frozen plan, environment, and
assignment, and no active worker lock. `--allow-partial` does not bypass integrity
checks, admit uninitialized shards, or turn an incomplete run into a completed one.

Collection validates and links the source evidence; it does not copy it.
**Keep every source shard at its frozen absolute path.** The aggregate uses
symlinks, so deleting or relocating the original shards breaks it. Use a fresh
output directory for each collection; existing outputs are never overwritten.

Reporting generates new CSV tables and PDF/SVG/PNG figures. Successful collection
does not imply a complete reproduction: strict reporting rejects failed runs or
missing evidence. Use `--allow-partial` on `python -m hma.repro.cli report` only for diagnosis,
and never describe that report as full-paper results. See the
[experiment map](experiments.md) for definitions and [verification record](verification.md)
for completed checks and remaining real-runtime validation.
