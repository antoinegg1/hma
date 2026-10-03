# Detailed local execution

For the recommended 75-node Docker Swarm campaign, start with the
[main README](../README.md) and [Swarm runbook](swarm.md). This page preserves the
complete single-host workflow; run its commands from the repository root.

This repository runs the experiments for **HMA: Humanize MLE Agents** from new
model sessions and newly prepared MLE-bench data. It includes the six native goal
baselines, six HMA pairs, five natural-termination alternation (NTA) pairs, cap and
starting-order ablations, four case studies, and three external harnesses on the
16-task comparison. Analysis reads your new runs; no historical result archive is
required or used to fill missing results.

**Start here:** install → configure → build → prepare one task → smoke → run →
report. [AGENTS.md](../AGENTS.md) is the maintenance/runbook for coding agents;
[the experiment map](experiments.md) connects commands to paper figures and
tables. [Validation status](verification.md) distinguishes offline checks
from real runs. This checkout has not yet passed a Docker/GPU/provider smoke.

## 1. Machine and access requirements

Use Linux x86-64, Python 3.12, Git/Git LFS, Docker Engine with the NVIDIA Container
Toolkit, and a non-root account allowed to run Docker. For this single-host
workflow, the controller runs directly on the Docker host and must reach container
IPs. Rootless Docker, Docker Desktop, remote daemons, and arbitrary containerized
controllers are unsupported. The supported containerized path is the
[Swarm controller with host networking](swarm.md#plan-and-deploy). The host must
have `python3.12-venv` or equivalent available.

The paper's execution environment is **one NVIDIA A10, 30 vCPUs, and 220 GiB
RAM**, shared by the alternating agents within a task (Section 4.1 and Appendix
C.1). The default configuration uses these limits for native and external
16-task runs. Set `gpus` to one device ID per worker; limits apply to each worker.
The H200 in Table 5 belongs to MLEvolve's original published configuration, not
the paper's HMA environment. See [environment provenance](provenance.md)
for the reported CPU, GPU memory, driver, and implementation-specific settings.

ScienceFlow's pinned environment uses CUDA 12.8 PyTorch, so the host driver must
support CUDA 12.8. Different hardware changes the experimental setting. Budget
multiple TB of disk for data, build layers, workspaces, and accepted candidates;
disk use is workload-dependent. Use an adequately sized local data/run volume.

You need Kaggle access and acceptance of each selected competition's rules, plus
API access to the exact model IDs in
[`experiments.json`](../src/hma/repro/assets/experiments.json). Endpoints must expose
the APIs used by the declared native CLIs. There is no automatic substitute when
a model is unavailable. Standard package, source, model-weight, and data downloads
also need network access.

## 2. Install and check without credentials

Run these commands from the repository root, keeping the virtual environment
active for subsequent commands:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-repro.lock
python -m pip install --no-deps -e .
python -m pytest -q
python -m hma.repro.cli doctor --offline
python -m hma.repro.cli plan --suite paper --output outputs/paper-plan.json
```

The default plan contains **3,397 task-runs**, with a maximum of **21,767.6 GPU
hours** of declared task windows, before data preparation/builds and teardown.
This is a budget preview, not an estimate of API charges or a promise of actual
runtime. No model is contacted by `plan`, `doctor`, or offline tests.

## 3. Local configuration and credentials

Edit the supplied `configs/local.json` directly to set `data_root`, image tags,
and GPU IDs. The agent
defaults are 30 vCPUs and `220g` (220 GiB). `expected_gpu_model` defaults to
`"NVIDIA A10"`; `doctor` and campaign execution check every selected GPU's exact
model name. For a hardware variant or smoke on another GPU, explicitly change
this field to that model or `null` to disable the model check, and label the run
as a hardware variant.

The 64 GiB shared-memory limit and separate private evaluator defaults of 2 CPUs
and 16 GiB are implementation choices, not values specified in the paper. Size
the host for evaluator and controller overhead beyond the agent limits; adjust
`evaluator_cpus` / `evaluator_memory` if needed. Relative paths resolve from your
current working directory. Run the commands from the checkout root.

**Every committed `api_key` and `base_url` in `configs/local.json` must remain
empty.** Fill these locally without committing the values, or use the existing
environment variables below. Environment variables take precedence:

| Provider | Key variable | Base URL variable | API |
| --- | --- | --- | --- |
| OpenAI | `HMA_OPENAI_API_KEY` | `HMA_OPENAI_BASE_URL` | Responses |
| Anthropic | `HMA_ANTHROPIC_API_KEY` | `HMA_ANTHROPIC_BASE_URL` | Anthropic Messages |
| GLM | `HMA_GLM_API_KEY` | `HMA_GLM_BASE_URL` | Anthropic-compatible Messages |
| DeepSeek | `HMA_DEEPSEEK_API_KEY` | `HMA_DEEPSEEK_BASE_URL` | OpenAI-compatible chat |
| Kimi | `HMA_KIMI_API_KEY` | `HMA_KIMI_BASE_URL` | Kimi Code-compatible API |

Both fields are required for selected providers. `.env` files are ignored by Git
but are **not automatically loaded** by the campaign command. Source your own
environment explicitly if you use one. Keep local configuration readable only by
your user (`chmod 600 configs/local.json`). Keys are injected into the selected
actor container, not experiment manifests or evaluator environments. Native and
external tools can persist credential-bearing state in their run directories;
treat raw run directories as private and inspect before sharing them.

## 4. Build environments

```bash
python -m hma.repro.cli build
```

This builds `agent`, `evaluator`, `ml_master_v2`, `mlevolve_no_prior`, and
`scienceflow` in that order. Upstream revisions and file hashes are verified
before building external harness images. No private image registry is required.
For only the native workflows:

```bash
python -m hma.repro.cli build --image agent --image evaluator
```

Then build selected external layers when needed:

```bash
python -m hma.repro.cli build --image ml_master_v2 --image mlevolve_no_prior --image scienceflow
```

Images record resolved Python dependencies. ScienceFlow retains its upstream
`uv.lock`; the other two harnesses use this repository's constrained runtime
requirements and common training stack. These are documented fresh-run
environments, not reconstructions of old private images. See
[provenance](provenance.md).

## 5. Prepare and verify data

Configure Kaggle with `KAGGLE_API_TOKEN` or a user-only
`~/.kaggle/access_token` file. Accept competition rules on Kaggle before
preparation. No Kaggle credential is copied into actor containers.

```bash
python -m hma.repro.cli data prepare --task leaf-classification
python -m hma.repro.cli data verify --task leaf-classification
python -m hma.repro.cli doctor
```

`doctor` checks prerequisites, selected GPU models, and whether provider settings
are present; it does not test API connectivity. `run` additionally checks selected
images and data manifests. Each campaign records GPU UUIDs, names, memory and
drivers, plus host CPU model/count and RAM in `environment.json`; CPU model and
driver version are recorded rather than required to match the historical host.
Data preparation pins MLE-bench, downloads the Kaggle archive, invokes its
preparer, checks upstream MD5s, and freezes SHA-256
inventories for public/private/control trees. Some tasks use separate Python 3.11
preparation environments managed by the pinned `uv` executable.

Prepare the complete suite, or just the external comparison:

```bash
python -m hma.repro.cli data prepare --suite paper
python -m hma.repro.cli data verify --suite paper
# Alternative subset:
python -m hma.repro.cli data prepare --suite harness16
```

Re-running preparation verifies already frozen tasks. A checksum mismatch is an
error; do not edit the manifest to suppress it. `--prune-working-data` removes
redundant extraction intermediates after successful preparation, keeping the
frozen inputs and original downloads.

**Answer-key scope:** this repository uses the pinned public upstream answer keys.
The manuscript mentions corrected keys, but a complete versioned patch was not
identified in the supplied local materials. No undocumented correction is
invented. Fresh scores must therefore be labeled with this data version; exact
agreement with the paper's corrected-key numbers is not claimed.

## 6. Run small end-to-end checks

These commands call real models after you configure credentials:

```bash
python -m hma.repro.cli smoke --experiment goal-gpt --task leaf-classification --seconds 300 --run-root runs/smoke-gpt
python -m hma.repro.cli report --run-root runs/smoke-gpt --output outputs/smoke-gpt --allow-partial
```

A short run may produce no submission. To exercise submission and actual grading,
verify that `submissions.csv` is nonempty and contains a finite score; increase
the smoke budget if necessary. A completed controller alone is not proof of a
successful benchmark solution. Use unique run roots for each new attempt.

Repeat with `goal-opus`, `goal-glm`, `goal-ds4`, `goal-ds41`, and `goal-kimi`
for every native model/backend. Check alternation separately:

```bash
python -m hma.repro.cli smoke --experiment hma-gpt-kimi --task leaf-classification --seconds 1800 --run-root runs/smoke-hma
python -m hma.repro.cli smoke --experiment nta-gpt-kimi --task leaf-classification --seconds 600 --run-root runs/smoke-nta
python -m hma.repro.cli smoke --experiment ml_master_v2-ds4 --task leaf-classification --seconds 600 --run-root runs/smoke-master
python -m hma.repro.cli smoke --experiment mlevolve_no_prior-ds4 --task leaf-classification --seconds 600 --run-root runs/smoke-evolve
python -m hma.repro.cli smoke --experiment scienceflow-ds4 --task leaf-classification --seconds 600 --run-root runs/smoke-science
```

HMA smoke must leave more than its 900-second review reserve. Harness smoke uses
`seconds - 180` for research and disables postprocessing; it is a startup check,
not a scientific result. All smoke plans have distinct fingerprints and labels.

## 7. Run the paper matrix

```bash
python -m hma.repro.cli run --suite paper --run-root runs/paper
```

The scheduler assigns one task at a time per configured GPU. The full matrix
includes all experimental arms and reference runs needed by the report. Selection
commands are available for development or smaller campaigns:

```bash
python -m hma.repro.cli plan --suite hma
python -m hma.repro.cli run --suite goal --run-root runs/goal-only
python -m hma.repro.cli run --experiment hma-gpt-opus --task mbh_07 --repeat 0 --run-root runs/ingv
```

Selectors accept task IDs, slugs, or catalog names; `--experiment`, `--task`, and
`--repeat` can be repeated. Repeats are zero-indexed. Available suites are
`paper`, `goal`, `hma`, `nta`, `ablation`, `harness16`, and `cases`. The `cases`
suite adds the single GLM–Kimi Pawpularity run; the other three case studies come
from main HMA repeat 0. The `harness16` suite selects the 96 external runs; its
native references come from the goal suite. `--experiments PATH` accepts a
modified copy of the JSON suite and records the full definition. Changing any
model, prompt, cap, timing, data, or environment creates a new protocol.

## 8. Status, interruption, grading, and reports

```bash
python -m hma.repro.cli status --run-root runs/paper
python -m hma.repro.cli run --suite paper --run-root runs/paper --resume
python -m hma.repro.cli export --run-root runs/paper --output outputs/events
python -m hma.repro.cli grade --run-root runs/paper
python -m hma.repro.cli report --run-root runs/paper --output outputs/rerun
```

Resume requires the exact original selection and unchanged code, image IDs,
resource settings, recorded hardware, and data manifests. It schedules **only
never-started cells**; complete, failed, or interrupted attempts are preserved.
There is no silent retry, additional budget, or best-of-attempts selection. To investigate a failure,
run the selected cell in a new root and retain the original status. Reports do
not automatically merge independent campaigns. Resume can finish pending cells,
but cannot turn failed or interrupted cells into complete coverage. A complete
report requires complete evidence from one frozen campaign; do not splice in
replacement attempts from other roots.

`grade` independently rescoring retained accepted CSVs is optional: the blind
evaluator already stores private scores during execution. It writes an offline
`regraded.json` without rewriting the hash-chained acceptance ledger. Never edit
evaluator data between execution and regrading. `report` emits normalized JSON,
CSV tables, and PDF/SVG/PNG figures. It fails on incomplete runs or missing
response usage and writes `coverage.json` describing the issues. To inspect
incomplete evidence, explicitly use `--allow-partial`; plots are labeled partial,
and failed/no-submission tasks remain in the planned denominator. Empty response
windows and single-repeat standard errors are undefined, not zero.

Artifact-bound disqualifications can be documented in `runs/paper/exclusions.json`:

```json
{
  "experiment/task/r0": {
    "reason": "Describe the observed disqualifying evidence",
    "artifact_sha256": "Exact SHA-256 of the final candidate"
  }
}
```

Use actual planned IDs/digests. This zeroes that returned candidate's task metrics
without removing the task or rewriting its exploration history. Do not copy
historical exclusion lists into new experiments.

## Repository layout

| Path | Responsibility |
| --- | --- |
| `src/hma/repro/` | CLI, validated plans, scheduling, providers, builds and regrading |
| `src/hma/repro/assets/` | Experiment matrix, 16-task set, upstream/version locks |
| `src/hma/supervisor.py` | Fresh sessions, budgets, alternation and final review |
| `src/hma/benchmark/` | 75-task catalog, data preparation, blind evaluator and grader |
| `src/hma/analysis/` | Native usage normalization, statistics, new-run tables/figures |
| `src/hmz/` | Vendored native agent runtime |
| `baselines/harness16/` | Frozen external adapters and source file locks |
| `tests/` | Offline protocol, statistics and integration fixtures |

Use `configs/local.json` and `python -m hma.repro.cli` for experiment configuration,
the complete matrix and hardware preflight. The low-level `hma-run` and
`hma-stage-evaluator` commands are retained for controller development; `hma-run`
does not perform that hardware check. The root AGENTS.md is for repository
maintenance and is not staged into benchmark workspaces.
