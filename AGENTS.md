# Instructions for maintaining and running this repository

Read README.md, docs/experiments.md, and docs/verification.md before changing the
reproduction protocol. This file applies to the source repository. It is **not an
experiment prompt** and must not be staged into an evaluated agent's workspace.
The user's task scope is fresh reruns; historical archives are unnecessary.

## Execution runbook

1. Work from the checkout root in Python 3.12. Install `requirements-repro.lock`,
   then `pip install --no-deps -e .`. Copy the example local JSON only if the local
   file does not exist. Keep committed API keys/base URLs blank.
2. Run `pytest -q`, `hma-repro doctor --offline`, and `hma-repro plan --suite paper`.
   The default plan has 27 configurations, 3,397 cells, 75 tasks (22/38/15), and
   21,767.6 maximum GPU hours. Investigate any unexplained difference.
3. On a suitable non-root Docker/NVIDIA host, follow README sections 3–6: supply
   provider/Kaggle settings, build images, prepare/verify leaf-classification,
   and run a real native smoke. Then check the other backends, HMA/NTA, and each
   external harness in separate smoke roots. Do not claim an actual submission
   passed unless an accepted candidate and finite grader result exist.
4. Prepare/verify all required tasks, freeze the `paper` plan, and use
   `hma-repro run --suite paper --run-root runs/paper` for the full campaign.
   Real smoke/full runs consume paid inference; preparation alone does not imply
   the user requested those runs. Follow the authorization in the conversation.
5. Inspect `status`, preserve failures, optionally `grade`, and run `report`.
   Use `--allow-partial` for diagnosis only. A partial/smoke report must never be
   described as full-paper results. `coverage.json` is the machine-readable record.

## Protocol invariants

- Appendix C.1 specifies one NVIDIA A10, 30 vCPUs, and 220 GiB RAM per task,
  shared by the alternating agents. Use the same configured limits for native
  and external 16-task runs. Table 5's H200/other hardware describes cited
  baselines' original papers, not this execution environment. Keep
  `expected_gpu_model="NVIDIA A10"` for the paper setting; an explicit different
  model or null is a hardware variant. Use `hma-repro` for hardware preflight;
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
hma-repro doctor --config configs/local.example.json --offline
hma-repro plan --suite paper
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
