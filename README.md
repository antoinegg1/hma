# hma

The **submit3-minimal** Flame Chase controller, with the **humanize** agent runtime
(`hmz`) vendored alongside it, so that the protocol and the runtime it was measured
against stay pinned to one another.

```
flows/flame_chase_submit3_minimal/   the controller  (~120 KB, 10 files)
src/hmz/                             the agent runtime, vendored verbatim
```

## What the protocol is

Two actors alternate, each in a **fresh Docker container and a fresh HOME**, sharing
**one working tree** for the whole cell. It is ordinary Flame Chase transfer plus
exactly two additions:

1. An **undisclosed cap** on accepted submissions per session. The controller ends the
   session after the cap-th accepted submission; no sentence of the prompt says a cap
   exists.
2. One **closing ballot**, cast by the model that did *not* write the standing
   submission, which may re-point the cell's result at another accepted submission.
   The ballot is paid for **out of** the cell budget, not added to it.

`flows/flame_chase_submit3_minimal/README.md` is the authoritative description,
including what this variant deliberately does *not* do — the ancestor's artifacts-only
handoff machinery is still present in `handoff.py` but is dead code here.

> The `submit3` in the name is historical: the 2026-09-12 ancestor's cap was three.
> Here the cap is a config value, and the production arm used five.

## What this repository is not

**It is not a one-command reproduction.** Running a cell end to end additionally needs,
none of which lives here:

- pinned local **agent and evaluator images** (`agent_image`, `evaluator_image`)
- prepared **MLE-bench task data** for the agent and evaluator mounts
- **credentials** for whichever CLIs the actors are configured with
- a **gateway proxy module**, supplied through the `kimi_proxy_module` config key

`flows/flame_chase_submit3_minimal/example.json` shows the shape of a config with those
slots marked. Read it before assuming a missing file is a bug.

## Relationship to upstream

`src/hmz/` is a **verbatim vendored copy** of
[humanfia/humanize2](https://github.com/humanfia/humanize2) at commit
`413d02e44d0cc0514b9f5bd3fcefea156b047a49`, excluding only `__pycache__`. It is vendored
rather than depended on so that this repository is a fixed pair of controller and
runtime. **Upstream fixes do not reach this copy on their own** — re-vendoring is a
deliberate act. See `NOTICE`.

The controller itself is maintained in
[antoinegg1/flowverse](https://github.com/antoinegg1/flowverse) under
`flows/flame_chase_submit3_minimal/`.

## Licence

Apache-2.0. See `LICENSE` and `NOTICE`.
