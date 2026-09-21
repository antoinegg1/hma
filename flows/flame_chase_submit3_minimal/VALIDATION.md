> **Stale: this record is the ancestor's, dated 2026-09-12.** It validates a cap of
> three, an allowlist export and a phase that ran after the cutoff -- none of which
> this variant has. The current shape is an undisclosed cap of five, one closing
> ballot by the non-author, and a review paid for out of the budget (exploration is
> 20700 s of a 21600 s cell). The live pins for THIS flow are
> KaggleBench's `tests/flowbench/flows/test_submit3_minimal.py`, which
> since 2026-09-17 also stands the controller up against a stub Docker client and a
> fake control plane rather than only asserting on source text.

# Validation — 2026-09-12

## Pool integration follow-up

- Final protocol/original-flow/staging test run: **50 passed**; Ruff passes.
- Real Docker smoke also passed from inside the production worker image, with
  the existing cluster isolation guard active. This uses fake actors/scorer,
  no model API and no GPU, and verifies three fresh containers, quota, natural
  exit, deadline, immutable protocol injection and cleanup.
- Smoke audit:
  `/tmp/submit3-worker-smoke-KkZkNF/submit3-smoke-gjieq8bk/workspace/cells/mbl_05/submit3/test/0/submit3-runtime`
- Approved production cohort: 20 GPT-first/Kimi tasks, 6h, dedicated workers on
  the frozen 48-host allowlist. Two startup incompatibilities were repaired:
  the pinned Humanize loader's parent-directory cleanup and the cluster guard's
  strict host-mount contract. Neither the guard nor isolated-host permissions
  was changed. Failed startup attempts remain separate infrastructure evidence.
- First live probe confirmed actual Codex assistant responses and running
  agent/evaluator containers for `mbh_04` and `mbl_11` on two fleet nodes. This is startup evidence, not final benchmark completion or
  verification of every task's scoring path or the subsequent real Kimi turn.
- Deployment/config/state/probe evidence is under KaggleBench's
  `experiments/mlebench_mh_goal6h/submit3_artifacts20/`.

## Initial implementation validation

No model API, Kaggle submission, production experiment, pool entry or GPU was
used. The original Flame Chase and native evaluator source were not modified.

- New protocol/allowlist/watchdog tests plus original Flame Chase tests:
  **31 passed**.
- Existing KaggleBench Flame staging/flow regression tests: **18 passed**.
- Ruff formatting and checks pass for every newly added Python file.
- Real Docker CPU smoke, using fake actor processes and a fake scorer with the
  actual native evaluator receipt implementation: **passed**.

Latest smoke audit directory:
`/tmp/submit3-smoke-ke2vp4ew/runs/experiment`

Observed sequence:

1. Actor 0: exactly three successful submissions, forced cap exit; fourth rejected.
2. Actor 1: one successful submission, natural exit; no attempt to fill its quota.
3. Actor 0: new container and HOME, then stopped at the immutable global deadline.

Five total accepted receipts, no exposed score feedback, no inherited analysis,
session history or Git metadata. Allowed code and the accepted candidate were
inherited. Dummy daemonized background processes were created inside the test
containers; container teardown completed with no cleanup errors. All four
containers and the private test network were removed; small audit files remain.

This validates the controller, native evaluator adapter and container boundary,
not live provider credentials, production task-specific grader dependencies,
large-artifact disk capacity, or semantic removal of analysis inside code or
opaque binary files. Those are not claimed as tested.
