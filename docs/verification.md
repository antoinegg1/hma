# Validation record

This repository prepares fresh reruns. It does not claim the expensive paper
matrix has already been rerun or that public upstream labels reproduce the
manuscript's unspecified corrected-key numbers.

## Checked locally

- Python 3.12 environment installed from the reproduction dependency set.
- Offline tests exercise cap 1/3/5, explicit unlimited NTA admission, single-session
  goal behavior, private/public separation, blind feedback, final-review rules,
  immutable submission evidence, planned denominators, SE/KL/prefix calculations,
  score regressions, partial reports, external candidate fallback, adapter hashes,
  smoke budget contracts, natural-completion precedence, watchdog reason codes,
  and response deduplication for four native log formats. A complete synthetic
  new-run fixture also verifies normalization, score regression retention and
  actual PDF/SVG case/trajectory rendering.
- Full matrix resolves to 27 configurations and 3,397 task-runs; all 75 task IDs
  resolve with 22 lite / 38 medium / 15 high tasks. Provider templates are blank.
- All four public upstream Git revisions were successfully fetched. The upstream
  allowlists and fixed adapter hashes for all three external harnesses matched.
- All three frozen adapters also passed configuration-only execution against
  copies of the verified upstream sources with synthetic credentials/task text.
  This does not exercise their model loops or training dependencies.
- Dependency resolution checks cover the host/native and common ML/harness locks.
  This is not evidence that every large training package imports on a GPU host.

Recorded on 2026-10-03: **50 offline tests passed**, Ruff lint and formatting
passed, and `hma-1.0.0-py3-none-any.whl` built successfully. A separate clean
Python 3.12 environment installed the locked dependencies and wheel, passed
`uv pip check`, and ran the same test suite. The combined host/ML/harness locks
resolved without conflicts. A blank-credential launch failed before Docker or
inference, as intended. The installed wheel contains the experiment and adapter
assets and no datasets, runs, credentials or local configuration.

## Not run here

The implementation machine has no Docker CLI/daemon. No real model credentials
were configured for this work. Consequently none of the following is marked as
passed: image builds, NVIDIA container execution, Kaggle task preparation, actual
private grading, native provider authentication, external harness inference, or
full-paper experiments. The fresh-clone Docker/data/API path still needs the
real-host checks below; offline fixtures are not substitutes for those checks.

## Real-host acceptance checklist

Follow README installation/configuration and then, in order:

1. `hma-repro build` succeeds for all five images. Check native CLI versions,
   `pip check`, and the recorded environment freeze. Confirm the ScienceFlow
   interpreter runs as the configured non-root user and its CUDA 12.8 stack can
   access the chosen GPU.
2. `hma-repro data prepare --task leaf-classification` and `data verify` succeed;
   `doctor` reports ready after all provider settings are configured.
3. Run each README native smoke in its own root. Each log must identify the
   declared backend/model and decode nonempty main-response usage. For grading
   acceptance, at least one finite-scored accepted candidate is required.
4. Run HMA/NTA smokes. Inspect `turn-history.json`, native session IDs/HOMEs,
   accepted counts and unchanged shared workspace. A smoke too short to reach a
   cap/natural exit does not validate a real handoff. HMA must show final review
   or a justified no-candidate fallback; NTA must have no final review.
5. Run all three external harness smokes, then a sufficiently long selected task
   for real candidate export. MLEvolve must record enabled memory/search
   mechanisms and disabled cold-start prior. A short smoke may have no final.
6. `export`, optional `grade`, and `report` must agree on accepted candidate
   identities, scores and denominators. Missing usage or incomplete cells must
   appear in `coverage.json`; deliberately altered artifacts/ledgers must fail.
7. Prepare and verify all 75 tasks before the full campaign. Preserve the exact
   image IDs, data manifests and plan. Launch `--suite paper`; after interruption,
   use the same command plus `--resume`. Verify already-started cells are not
   silently retried and the original attempt remains visible.
8. Only call the final output a complete new-run reproduction when coverage is
   complete and all scientific inputs match the documented protocol. Record the
   actual GPU, driver, provider endpoints/model versions, images and data revision
   alongside any publication of new results. Do not include secret values.
