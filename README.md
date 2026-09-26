# HMA

HMA alternates Claude Opus 5 and GPT-5.6-sol in fresh agent sessions while
preserving one shared task workspace. Each option ends on natural completion,
five accepted experiments, or the exploration deadline. Final Review selects
among accepted candidates within the same six-hour budget.

## Contents

- `configs/hma-opus-gpt.json`: the sole experiment and evaluator configuration.
- `src/hma/`: controller, evaluation bridge, and provider adapter.
- `src/hmz/`: supporting native-agent runtime.
- `docker/`: overlays for externally prepared agent and evaluator images.
- `tests/`: offline protocol tests with temporary synthetic inputs.
- `LICENSE`: license terms and component notices.

## Install and test

Use Python 3.12 or later:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-core.lock
python -m pip install --no-deps -e .
python -m pytest -q
```

The tests exercise controller transitions, session isolation, evaluator
admission, and Review handling without model inference or benchmark data.

## Run

Prepare the MLE-bench task, upstream grader, public input, private evaluator
data, native-agent CLIs, and provider credentials separately. The agent image
needs `requirements-runtime.lock` and this package; the evaluator image needs
the task's grader dependencies. The Dockerfiles add HMA to these base images.

Edit `configs/hma-opus-gpt.json`: replace `/REPRO_ROOT/...`, image names, and
credential-file paths; set non-root `uid`/`gid` to match file ownership. Fill in
the nested `evaluator` block with the task slug, metric direction, and paths as
seen inside the evaluator container. Then run on the Docker host:

```sh
hma-stage-evaluator --config configs/hma-opus-gpt.json
hma-run --config configs/hma-opus-gpt.json --validate-only
hma-run --config configs/hma-opus-gpt.json
```

Both the run directory and evaluator home must be new. The controller uses a
21,600-second budget with 900 seconds reserved for Review; the Review call is
limited to 600 seconds to leave time for teardown and finalization. A missing
or invalid nomination keeps the standing candidate. Agents cannot access
private evaluator data or hidden test scores.

This anonymous code supplement contains no experimental results, datasets,
trajectories, transcripts, or Git metadata. Telemetry is disabled. The dependency
lists describe the supplementary code's validation environment; external model
CLIs and grader images are supplied by the reproducer.
