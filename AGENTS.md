# Instructions for maintaining and running this repository

This is the single entry point for an agent operating this repository. Before
running or changing experiments, read the [README](README.md),
[experiment map](docs/experiments.md), and [validation record](docs/verification.md).
Follow the linked local/Swarm runbooks at each stage below; they contain the full
commands. This file applies to the source repository. It is **not an experiment
prompt** and must not be staged into an evaluated agent's workspace. The scope is
fresh reruns; historical archives are unnecessary.

## Required inputs and execution scope

Use the user's existing instructions and configuration. Continue work already
authorized; do not ask again merely because a run consumes API credits. A request
limited to repository edits is not a request to publish images, deploy jobs, or
run paid experiments. Follow the execution scope already authorized in the
conversation. If a required input is missing, identify it and continue independent
preparation; never invent values.

| Input | Required for execution |
| --- | --- |
| Scope and paths | Full `paper` suite or an explicitly selected subset; fresh run/output roots. Smoke results never replace formal runs. |
| Machines | A10 execution hosts with working NVIDIA Docker; 30 vCPU/220 GiB for each actor, plus evaluator/controller overhead (recommend 32 vCPU/256 GiB hosts). |
| Model access | Keys and base URLs for the selected providers; full paper requires `openai`, `anthropic`, `glm`, `deepseek`, `kimi`. Exact models and API formats are in the README and [credential guide](docs/local-run.md#3-local-configuration-and-credentials). |
| Data access | Kaggle credentials and accepted competition rules; shared `/srv/hma/data` for Swarm. |
| Cluster access | Manager/node access, shared `/srv/hma` with cross-node locking, common non-root UID/GID and Docker socket GID. |
| Image registry | A repository/tag to publish all five images and credentials allowing every node to pull them. |

Edit the supplied `configs/local.json` directly; no template copy is needed.
Keep committed API keys/base URLs blank. Environment overrides take precedence;
`.env` is not loaded automatically. For Swarm, set `data_root` to `/srv/hma/data`
before preparing data. Replace `ORG`, `DOCKER_GID`, `NODE_ID`, and other documented
placeholders with the actual environment values.

## Execution runbook

1. **Install and check offline.** Work from the checkout root using the Python
   3.12 environment from [installation](README.md#installation). Install
   `requirements-repro.lock`, then `pip install --no-deps -e .`. Run:

   ```bash
   python -m pytest -q
   python -m hma.repro.cli doctor --config configs/local.json --offline
   python -m hma.repro.cli plan --suite paper --output outputs/paper-plan.json
   ```

   Expect 27 configurations, 3,397 cells, 75 tasks (22/38/15), and 21,767.6 maximum
   GPU hours. A cell is one configuration/task/repeat, with its own workspace.
   Investigate any unexplained difference. Offline checks do not call models.

2. **Build and prepare one task.** On a non-root Docker/NVIDIA execution host,
   follow [builds](docs/local-run.md#4-build-environments) and
   [data preparation](docs/local-run.md#5-prepare-and-verify-data): build all five
   images, prepare and verify `leaf-classification`, then run live `doctor`.
   A non-GPU manager cannot pass the host GPU check; do this on an A10 host with
   the same checkout/configuration. `doctor` checks configuration and hardware,
   but does not test API authentication or connectivity.

3. **Validate real execution.** Run the [smoke commands](docs/local-run.md#6-run-small-end-to-end-checks)
   for all six native models, HMA, NTA, and all three external harnesses in
   separate fresh roots. Check the declared model, nonempty parsed main-response
   usage, and an accepted candidate with a finite private score. HMA/NTA must
   actually hand off; inspect `turn-history.json` and fresh session/HOME evidence.
   HMA needs review or a justified fallback; NTA must have no review. A short
   smoke that produces no submission or handoff is inconclusive: increase its
   budget in a new root. HMA smoke must exceed its 900-second review reserve.
   Use items 1–6 of the [real-host checklist](docs/verification.md#real-host-acceptance-checklist)
   for these local checks; complete its campaign and Swarm acceptance checks
   during the later stages below.

4. **Prepare Swarm and publish images.** Follow the [host setup](docs/swarm.md#host-and-storage-setup)
   and [cluster creation](docs/swarm.md#create-and-label-the-swarm), initially
   labeling only one A10 for the pilot below. Follow [images and data](docs/swarm.md#images-and-data)
   to publish the five built images and generate `configs/swarm.local.json`.
   Use that generated configuration for subsequent commands; its image references
   must include digests. It captures local provider values at generation time;
   later edits to `configs/local.json` do not update it. Do not rebuild between
   planning and execution. Single-host execution instead follows
   [the local campaign](docs/local-run.md#7-run-the-paper-matrix).

5. **Run a one-node pilot before labeling all 75 nodes.** Follow the
   [pilot commands](docs/swarm.md#optional-one-node-pilot) with `--expected-nodes 1`
   and a separate `/srv/hma/runs/pilot` root. Pre-pull its `images.txt` and deploy
   that root. This is a real six-hour goal run, not a short smoke. Validate
   submission, grading and reporting, and exercise controlled shutdown/recovery
   on a separate pilot campaign. Preserve interrupted attempts.

6. **Run the full frozen matrix.** Once the pilot passes, label exactly 75 Ready,
   Active A10 nodes; prepare and verify all 75 tasks. Follow
   [plan and deploy](docs/swarm.md#plan-and-deploy) using a fresh
   `/srv/hma/runs/paper` root. Freeze the task-to-node assignments, pre-pull the
   manifest's `images.txt` on every execution node, then deploy. Each node runs
   all configurations/repeats for its assigned task sequentially. The agent image
   also runs the controller, which creates local actor/evaluator containers via
   the Docker socket. Credentials travel in a Docker secret. Do not overlap other
   campaigns on these GPUs. Deployment returns immediately; monitor until the
   workers finish rather than treating a returned service ID as completion.

7. **Monitor and recover without retrying evidence.** Use the service name from
   deployment for `docker service ps --no-trunc SERVICE_NAME` and
   `docker service logs SERVICE_NAME`. Inspect an individual shard, for example:

   ```bash
   python -m hma.repro.cli status --run-root /srv/hma/runs/paper/shards/mbl_09
   ```

   Do not pass the cluster root to `cli status`; it is not a campaign shard.
   `deploy --resume` requires a new `--name`, optionally `--node-id` for one
   original node, and runs only never-started cells. The old job must have stopped.
   Never force service restarts or silently retry attempts. SIGTERM supports
   cleanup; SIGKILL leftovers block resume until the original owner's containers
   are explicitly cleaned up. Preserve logs and errors. Resume cannot repair a
   failed/interrupted cell into complete coverage. Diagnose in a new root; a new
   full cohort, if required, must remain separate and within the authorized scope.

8. **Collect and report.** Follow [collection](docs/swarm.md#collect-and-analyze).
   All workers must have stopped, all shards must be initialized, and every cell
   must have a complete/failed terminal result. Optional regrading targets each
   original shard before collection. Then run:

   ```bash
   python -m hma.repro.swarm collect --root /srv/hma/runs/paper --output /srv/hma/runs/paper-report
   python -m hma.repro.cli report --run-root /srv/hma/runs/paper-report --output outputs/rerun
   ```

   Use a fresh collection output. Keep source shards at their original paths:
   collection links to them. Collection holds shared locks that conflict with
   active worker locks. For diagnosis, `collect --allow-partial` permits pending,
   interrupted and cleanup-failed cells, but not uninitialized shards; pair it
   with `report --allow-partial`. Successful collection may still contain failures.

## Completion and handoff

Only describe a campaign as a complete full-paper rerun when strict `report`
succeeds without `--allow-partial` and `outputs/rerun/coverage.json` has:

- `planned == 3397` and `full_paper_plan == true`;
- `partial == false` and `issues == []`;
- `complete == planned` for every entry in `by_experiment`.

A selected/smoke plan is not full coverage, even if all its cells succeed. A valid
run with no accepted candidate can be a counted miss; do not select replacements
to improve the score. Check generated tables/figures against the
[experiment map](docs/experiments.md#paper-output--computation--generated-files).
Deliver the frozen plan, actual hardware/image/data provenance, original shard
and aggregate paths, report directory, coverage, and remaining limitations.
Raw native HOME/log directories may contain credentials; do not publish them
uninspected or expose secret values in a handoff.

Documentation and offline checks do not prove live execution: Docker/GPU/API and
multi-node validation are still outstanding in the recorded environment. The
paper's complete corrected-answer-key patch is unavailable, so this repository
uses pinned public upstream keys. Report that data version; complete new-run
coverage does not establish exact agreement with the paper's corrected-key scores.

## Protocol invariants

- Appendix C.1 specifies one NVIDIA A10, 30 vCPUs, and 220 GiB RAM per task,
  shared by the alternating agents. Use the same configured limits for native
  and external 16-task runs. Table 5's H200/other hardware describes cited
  baselines' original papers, not this execution environment. Keep
  `expected_gpu_model="NVIDIA A10"` for the paper setting; an explicit different
  model or null is a hardware variant. Use `python -m hma.repro.cli` for hardware preflight;
  low-level `hma-run` has no such check. The 64 GiB shared-memory limit and
  separate 2-CPU/16-GiB evaluator are implementation choices absent from the
  manuscript. Provision their overhead separately.
- The immutable matrix is `src/hma/repro/assets/experiments.json`. No fallback
  models or undocumented changes to caps, timing, prompts, order or repeat count.
- Goal: exactly one fresh native session with native goal pursuit enabled; no
  outer session restart, cap or review. HMA: alternate fresh sessions/HOMEs over
  one shared workspace; accepted submission cap 5 (1/3 only in named ablations).
  NTA: alternate only on natural completion, explicit null cap, no final review.
- Native runs have a 21,600-second total window. HMA reserves 900 seconds, with a
  600-second review call and time for shutdown/finalization. The 900-second
  reserve is specified by the paper; the 600-second call limit is an implementation
  choice. NTA uses one repeat in this fresh-run matrix. The historical cap-1/3 and
  reverse-order archives had 21,700-second terminal windows; fresh runs enforce
  the nominal 21,600-second budget. Never reset time on handoff or silently extend
  a deadline. Harness research and total windows are
  separate fields; smoke shortening cannot modify the default scientific plan.
- The admission gate counts accepted submissions only; invalid submissions do
  not exhaust a cap. Null means unlimited. Zero is the internal review-only gate,
  not a substitute for NTA. Natural completion wins ties with reaching the cap;
  distinguish quota-watchdog exit from a successful native return. A failed
  native actor is a recorded failure.
- Agents see public data and acceptance/format feedback only. Private labels,
  scores, ranks, medal thresholds, evaluator admin keys and root instructions
  must never enter agent mounts or prompts.
- The standing candidate is the latest accepted candidate. Final review may
  nominate only an already accepted artifact, with standing fallback. Never
  substitute the best hidden test score. Repeated review receipts do not create
  new exploratory experiments. Agents cannot submit during review.
- External harnesses use the fixed adapters in `baselines/harness16/`. Preserve
  their bytes and source hashes. Prior knowledge is disabled. If an official
  final is missing, use the declared latest pre-deadline CSV rule, never hidden
  score selection. MLEvolve memory/mechanism checks remain enabled; the memory
  encoder is a pinned generic pretrained model, not competition prior knowledge.
- Resume schedules only never-started cells under exactly the same plan, code,
  image IDs, resources, recorded hardware and data manifests. GPU UUID/name,
  memory/driver and host CPU model/count/RAM are recorded; CPU model and driver
  version need not match the historical host. No automatic retries, score-based
  exclusions, deletion of failed cells, or combining independently sampled runs.
- Table metrics use fixed planned denominators, per-repeat rates then mean/SE.
  One repeat has undefined SE. Final disqualifications require evidence, reason,
  and exact final artifact digest; retain the task denominator and raw history.
- Responses are native main-agent model responses with provider output usage;
  exclude subagents/coordinators/tool returns, include tool-requesting model
  responses, deduplicate IDs/cumulative snapshots. Never estimate missing usage.
  Keep empty windows undefined and expose decoding failures in coverage.
- Figure 3 is offline prefix analysis of first-repeat goal/NTA runs. Do not call
  it an online stopping guarantee or insert historical curve values.
- Cases use predefined tasks/repeat 0 and all accepted exploratory scores. Never
  select a better repeat after seeing results or describe historical behaviors as
  observed in new runs without new trajectory evidence.

## Code map and verification

`repro/` owns CLI/configuration/scheduling/provider injection/builds. `supervisor.py`
and `evaluator.py` own execution/admission. `benchmark/` owns data integrity and
grading. `analysis/` owns normalized records and calculations. `hmz/` is the
vendored native runtime. See docs/provenance.md for imported material.

Use synthetic fixtures for protocol changes, not expensive inference in unit
tests. Required offline checks after relevant code changes:

```bash
python -m pytest -q
python -m hma.repro.cli doctor --config configs/local.json --offline
python -m hma.repro.cli plan --suite paper
```

For lint/build development tools, install `ruff` and `build` separately, then use:

```bash
ruff check src/hma tests
ruff format --check src/hma tests
python -m build --wheel
```

Do not format frozen external adapters or casually reformat vendored `hmz/`.
If editing imported sources, record the change and update provenance. Regenerate
locks intentionally, never weaken checksum validation to make a build pass.
Check runtime/model/source versions against the lock. A changed build must receive
new image IDs and a new campaign root. Keep host-only dependencies out of model
workspaces and secrets/data/logs out of Git, Docker build context and wheel.

Update README and the experiment map with interface/protocol changes. Record
exactly what was tested in docs/verification.md: offline tests do not prove Docker,
GPU, API, all-task preparation, or full scientific reproducibility. Retain the
explicit public-answer-key limitation until a versioned correction is supplied.
