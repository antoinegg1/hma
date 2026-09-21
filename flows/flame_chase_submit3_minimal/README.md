# Flame Chase: submit3-minimal

Two alternating actors, each in a **fresh Docker container and HOME**, sharing
**one working tree** for the whole cell. This is plain Flame Chase transfer with
exactly two additions:

1. An **undisclosed cap** on accepted submissions per session. The controller
   ends the session after the cap-th accepted submission; no sentence of the
   prompt says a cap exists.
2. One **closing ballot**, cast by the model that did *not* write the standing
   submission, which may re-point the cell's result at another accepted
   submission.

The ballot is paid for **out of** the cell budget, not added to it. Nothing runs
past the cell's cutoff.

> **Naming.** The `submit3` in the package name is historical -- it comes from
> the 2026-09-12 ancestor `flame_chase_submit3_artifacts_only`, whose cap was
> three. Here the cap is a config value and the production arm used five.

## What this variant is *not*

The ancestor built an artifacts-only handoff: a deterministic allowlist, an
export step, and a per-handoff manifest, so that each actor inherited a filtered
copy of its predecessor's files. **None of that runs here.** Turns share one
workspace directly, exactly as Flame Chase does.

`handoff.py` still contains that machinery (`export_workspace`, `rule_for`, the
`POLICY` table), but this supervisor imports only `atomic_json` and
`copy_verified` from it. The export path is dead code in this variant; it is
retained so the module stays diffable against the ancestor.

Only the **per-turn HOME** is fresh. OS processes, installed packages, shell
environment, caches and HOME changes do not carry over, so actors must keep
checkpoints and dependency declarations in the workspace.

## Architecture

A **host-supervised multi-container workflow**, not an in-place
`hmz exec -f flame_chase` replacement. The trusted supervisor runs on the Docker
host and drives `actor_turn.py` -- a one-agent Humanize `@flow` with
`goals=False` -- inside each new container. Actors never receive the Docker
socket, the controller directory, the evaluator HOME, a previous HOME, or the
admin key. They keep ordinary network access.

`evaluator.py` wraps a copy of the native MLE evaluator at startup. The original
evaluator source, grader, blind public feedback, candidate-hash deduplication
and hash-chained receipts are unchanged. Protected per-turn tokens bind
admission to a session, and counts come from committed native receipts rather
than tool calls or model statements. Invalid submits, validations, duplicates
and retries do not increase the count. A result scored after the deadline is
rejected before a receipt is appended. Score and medal information stays hidden.

`actor_entry.py` is a container-local, model-free watchdog: it polls
`/session/status` and exits when the session is exhausted or closed, so a
container still terminates if the host supervisor disappears. It does not
schedule successors.

`submit_client.py` is the actor's blind submit helper, authenticated only by the
current turn's token from `/run/submit3/route.json`.

## The undisclosed cap

The controller enforces `max_valid_submissions_per_session` and the actor is
never told a cap exists. The measurement is of behaviour under an *unannounced*
cut, not of planning around a number.

Two things follow, and both are load-bearing:

**`prompt_for(task, seconds)` takes no cap parameter at all.** It cannot mention
what it is not given. This is stricter than "don't put the number in the text":
a parameter that is accepted and ignored is how a number leaks into a prompt
later. (The cap *was* a parameter until 2026-09-17, when the worker adapter --
which renders the same prompt into `task.md` before the controller starts --
called it with two arguments and every claim died in `TypeError` before the cell
began.)

**The one substitution that stays is the quota sentence.** The stock task text
asserts "There is no medal-based stopping condition and no submission quota."
Under a hidden cap the second half is a claim the actor cannot check, so it is
replaced by the half that is still true rather than by a hint that a cap exists:

```
There is no medal-based stopping condition.
```

If the phrase `no submission quota` survives rendering, `prompt_for` raises
rather than run an off-protocol cell.

**`/session/status` withholds `limit` and `selection`.** It projects only
`accepted`, `closed` and `exhausted`. `accepted` is not a leak -- an actor can
count its own accepted submits -- but `limit` is the number itself, and
`route.json` is injected `0o444` inside the agent's own container, so any bearer
of its token could otherwise read the cap straight out of the endpoint. The
supervisor reads full status over the separate control plane with
`X-Control-Key`.

If the task text has no `## Test-time information policy` section, the standard
anti-lookup clause is appended. Note what that clause does and does not cover:
it forbids seeking *external* solutions, notebooks, write-ups, recovered labels
or answer files. It says nothing about answers that ship inside the task's own
data.

## The closing ballot

One turn, one ballot, cast by the actor slot that did **not** author the
standing submission. No rounds, no peer ballot, no convergence, no confidence
gate.

Why that side, and why only one: in the multi-round variant the closing debate
produced a 27% blank-ballot rate over 111 turns, and every blank came from the
same actor slot (49.2% against 0.0%), while turns that produced nothing still
consumed their share of a wall no cell ever exhausted. Asking only the side with
no stake in the standing submission removes the wasted turns, and removes the
"what does an unopposed ballot mean" ambiguity with them: there is exactly one
ballot by construction.

The reviewer gets, in the shared workspace under `.flowbench/review/`:

| Path | Contents |
|---|---|
| `ballot.json` | Every accepted submission in order: id, sequence, accepted-at, `author` as `"you"`/`"peer"`, and which is `standing` |
| `candidates/N.csv` | The artifact for sequence `N` |

`ballot.json` carries identity, order and authorship -- never a score or a
leaderboard verdict. Authorship is rendered as `you`/`peer` so the reviewer can
be asked to account for its own work without being told which model wrote what.

The reviewer writes `.flowbench/review/nomination.json`:

```json
{"nominate": "<submission_id>"}
```

Naming the standing submission changes nothing. Naming another **accepted**
submission binds it as the cell's result via `/turn/finalize`. **Silence leaves
the standing submission in place** -- it is agreement by design, not a veto and
not a tie, and is exactly what would have happened without the phase.

The turn is opened with `limit 0`, so it cannot create anything, only re-point
at a candidate already accepted. That is also what lets it hold the cell's own
deadline while every submitting turn stopped at the reserve boundary.

Authorship is recovered by walking the turn records and the evaluator ledger
together: a record carries a **count** (`accepted`) and an **end**
(`last_submission_id`), never a range. An *absent* count is not a count of zero
-- if any turn's close degraded without one, attribution fails closed and the
phase reports `review_skipped`, because crediting the wrong actor would invert
every `you`/`peer` label and hand the ballot to the model being judged.

### Review outcomes

| `reason` | Meaning |
|---|---|
| `review_noop` | Reviewer named the standing submission, or there were fewer than two candidates |
| `review_silent` | Ran with session evidence and wrote no valid ballot -- a decline. Never retried; retrying a decline manufactures the opposite bias |
| `review_session_died` | No provider transcript at all, or exited on its own well before its wall |
| `review_skipped` | Standing submission could not be attributed to an actor |
| `review_error` | The phase raised; recorded, never a cell failure |

`review_silent` and `review_session_died` are deliberately separated: a
container that *hangs* is still running at the wall, so "ran its wall" alone
cannot distinguish a considered decline from a session that never started.
Absence of transcript evidence outranks the clock. Neither branch may consult
the exit code, which is read after the stop: with `init=True` tini forwards
SIGTERM, so **143 (or 137) is the normal exit for a reviewer that used its whole
wall** and must never fail the cell.

## Budget and deadlines

The review runs **inside** the budget, on a slice reserved from it. Elapsed cell
time is unchanged by design rather than by being excluded.

```
deadline         = start + active_time_limit_seconds        # e.g. 21600
explore_deadline = deadline - review_reserve_seconds        # e.g. 21600 - 900 = 20700
```

Every submitting turn is given `explore_deadline`. Only the zero-limit ballot
turn holds `deadline` itself. The ballot's own wall stops short of that by the
slices already owed -- `stop_seconds`, a 30 s teardown margin, and 60 s for the
`finalize` that binds the ballot -- or by `review_turn_seconds`, whichever comes
first. A reviewer that consumed the whole cell deadline would otherwise leave
nothing for its own close, and `finalize` refuses an unclosed turn.

`review_prompt_for(task, seconds)` **must** be told the turn's real wall, and
states it in the prompt. It is built on the *rendered* prompt -- what the actors
were actually given -- and guards both invariants: it raises if the text still
contains `no submission quota` (which would leak the cap on the one turn that is
the measurement) and if `seconds` is not a positive `int`.

This was not always so, and the cost was a measurement rather than a crash.
Until 2026-09-18 the reviewer inherited the rendered task statement, which
advertises six hours, with nothing correcting it for a turn killed at 600 s.
Reviewers that paced themselves for hours were SIGTERMed mid-thought: 33 of
glm53's 35 blank ballots died at exit 143 within 400 ms of the wall, while
gpt56sol, which happens to work fast, finished all 9 of its ballots inside the
window and produced all four of that arm's medal flips. That reads as "one model
cannot review" and is really "one model was never told when to stop".

## Configuration

Launch config is a single JSON file; see `example.json` (its `/ABS/...` values
are placeholders, not a runnable campaign). Unknown keys are rejected.

| Key | Default | Notes |
|---|---|---|
| `root` | -- | New, explicit, absolute campaign directory (>= 5 path parts, not a symlink) |
| `task_file` | -- | Base blind MLE task prompt |
| `agent_image`, `evaluator_image` | -- | Pinned local images |
| `evaluator_seed_home` | -- | Freshly prepared native evaluator HOME |
| `actors` | -- | Exactly two, in order |
| `agent_data`, `evaluator_data` | -- | Read-only mounts. The actor sees only the public input; private data must be under `evaluator_data` |
| `active_time_limit_seconds` | `21600` | Whole cell |
| `max_valid_submissions_per_session` | `3` | **The undisclosed cap.** The production arm set this to `5`; `example.json` leaves it at `3` |
| `review_reserve_seconds` | `900` | Held back from the budget for the ballot. Must exceed `review_turn_seconds` plus teardown, and be under the active limit |
| `review_turn_seconds` | `600` | The ballot turn's own length |
| `cpus`, `memory`, `shm_size`, `gpu` | `26`, `200000m`, `65536m`, `"0"` | Actor container resources |
| `uid`, `gid` | caller's | Must be non-root |
| `controller_container` | `null` | Connects a trusted containerized worker to the execution's private bridge |

`actors` defines alternation order. Only explicitly named environment variables
and provider auth/config files are injected; never seed a whole historical HOME.

## Usage

Put this repository's `flows` directory on `PYTHONPATH` and run the supervisor
**on the Docker host**:

```bash
PYTHONPATH=/path/to/humanize2/src:/path/to/flowverse/flows \
  python -m flame_chase_submit3_minimal.supervisor \
    --config /absolute/path/launch.json --validate-only

PYTHONPATH=/path/to/humanize2/src:/path/to/flowverse/flows \
  python -m flame_chase_submit3_minimal.supervisor \
    --config /absolute/path/launch.json
```

Do **not** run the supervisor inside an actor container, and do not enroll this
as an ordinary shared-HOME flow. The package deliberately has no in-place
Humanize entry point.

Runtime Python sources are copied and frozen per launch with their SHA-256
recorded. Launching against an existing root is **refused**: crash recovery is
deliberately fail-closed, not an automatic resume, so an old actor is never
overlapped and no budget is silently restarted.

## Outputs

Under `root`:

| Path | Contents |
|---|---|
| `contract.json` | The resolved config |
| `runtime-hashes.json` | SHA-256 of the frozen runtime sources |
| `deadline.json`, `state.json` | Deadline and live status |
| `workspace/` | The shared working tree, including `.flowbench/review/` |
| `turns/` | Per-turn HOMEs, routes and container logs |
| `evaluator/` | Evaluator HOME, native ledger and retained candidates |
| `turn-history.json` | Per-turn records |
| `review.json` | Ballot outcome |
| `result.json` | Final outcome, including `exploration_elapsed_seconds` and `exploration_reason` |
| `container-logs/` | Logs captured at teardown |

`exploration_elapsed_seconds` and `exploration_reason` exist for the worker
adapter's receipt gate, which compares the **exploration** span against the
budget minus the reserve. Without them the gate falls back to the wall clock,
which is the whole budget by construction and therefore clears unconditionally.

## Requirements

- Trusted local Docker host; Python >= 3.11 (`tomllib`).
- `docker`, `pydantic>=2`, `requests`, and Humanize (`hmz`) on the host.
- A local agent image containing Humanize and the desired provider CLIs.
- A local evaluator image plus a freshly prepared native evaluator HOME and its
  read-only data mounts.
- Non-root UID/GID with read/write access to the prepared evaluator seed and
  controller artifacts. No Docker socket and no privileged agent mount.

## Validation

**The test suite for this variant is not in this repository.** It lives in
KaggleBench at `tests/flowbench/flows/test_submit3_minimal.py` -- 59 tests,
currently passing -- and imports this flow by absolute path:

```bash
PYTHONPATH=/path/to/humanize2/src:/path/to/flowverse/flows \
  uv run --no-sync pytest tests/flowbench/flows/test_submit3_minimal.py -q
```

The flowverse test `tests/test_flame_chase_submit3_artifacts_only.py` covers the
**ancestor**, not this flow; it imports `flame_chase_submit3_artifacts_only` and
exercises the allowlist export that this variant does not run. The same applies
to `tests/submit3_docker_smoke.py`, whose cap is 3 and which checks allowlist
inheritance.

See `VALIDATION.md` for the live-probe record.

## Limitations

- **The cap is enforced, not concealed from inspection.** It is absent from the
  prompt and withheld from `/session/status`, but this source is public and the
  controller's own control plane reports it. The claim is that the *actor* is
  not told, not that the number is secret.
- **The file policy is gone, and with it its guarantees.** Turns share one tree,
  so anything an actor writes -- notes, logs, analysis -- is visible to its
  successor. The ancestor's allowlist is the thing that used to prevent that.
- **Format-valid submissions do not prove compliance** with the anti-lookup
  clause, and the clause itself bars only external lookup.
- A guarded evaluator restart restores its ledger and turn admission; reopening
  the same turn is idempotent and cannot reset its quota or deadline.
- Per-turn HOMEs are archived, not garbage-collected. Allow disk accordingly.
- Actor errors, evaluator loss and failed turns stop dispatch rather than
  silently starting a new session. Cleanup targets only containers carrying this
  launch's random ownership label.
- Provider failures fail closed rather than silently switching or retrying, and
  do not count as successful completed experiments.
