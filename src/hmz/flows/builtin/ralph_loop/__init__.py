"""Ralph loop (flowbench: ralph_loop) -- a fresh session every turn, so nothing carries over.

hmz exec -f ralph_loop -a claude/MODEL:high "$(cat TASK.md)"

Add `-c budget.yaml` to hold it to something other than the budget it comes with, and
`hmz -f ralph_loop -c budget.yaml` opens the interface on the same setup.

Nothing carries over inside a run, and two things carry between runs: which round it is on,
kept as `rounds`, and what it has spent, kept as `output`. A loop like this is left going for
days and is stopped -- esc, a machine that goes down, a turn that takes the process with it --
so running it again goes on from the round it reached rather than back at one. What the agent
did is not kept: every round is a session of its own, written down by the backend that ran it,
and the next round starts from the task and the repository whether or not it is the first.

What ends it is the budget. A loop with nothing else to stop it runs until somebody stops it,
which is a bill nobody agreed to and a week of rounds nobody read; so it is held to `budget`
million output tokens, and 0 is the loop that goes on until it is stopped by hand. Output
rather than every kind, because output is what the model is asked to produce and the only
kind a loop of its own accord grows: what goes in is the task and the repository, and a round
that read more of them is not a round that did more.

Or a run of rounds that did nothing. A round whose turn failed answers with nothing and spends
nothing, so a loop whose account was refused or whose model it may not run sits under a budget
that never moves and goes round on the same failure for as long as it is left. Three such
rounds in a row end it. What it kept is left rather than cleared: a loop that stalled is one
to fix and start again from, not one that is over.

The spend is kept because the rounds are. A budget that started again at nothing every time
the loop was picked up would be no budget at all for the loop a week of restarts is, so what
is counted is every run of this flow in this workspace. A loop that has spent it is over, and
what is over is not picked up: it clears what it kept, so the next run here opens on a budget
of its own and at round one rather than stopping before it has taken a turn.
"""

import time
from typing import Any

from pydantic import BaseModel, Field

from hmz.flows import Agent, flow

MILLION = 1_000_000.0

STALLED = 3

class Config(BaseModel):
    """What this flow takes."""

    model_config = {"extra": "forbid"}

    budget: float = Field(
        default=10.0,
        ge=0,
        description="millions of output tokens the loop may spend before it stops, counted "
        "across every run of it in this workspace, or 0 to go on until it is stopped",
    )

@flow(resumable=True)
def run(
    agents: tuple[Agent],
    task: str,
    config: Config | None = None,
    state: dict[str, Any] | None = None,
) -> None:
    (agent,) = agents
    held = config or Config()
    kept = state if state is not None else {}

    before = kept.get("output", 0.0)
    stalled = 0
    while True:

        kept["rounds"] = kept.get("rounds", 0) + 1
        print(f"round {kept['rounds']}")

        answered = agent(task, suppress=True)
        kept["output"] = spent = before + agent.spent().output
        if held.budget and spent >= held.budget * MILLION:
            print(f"stopping: {spent / MILLION:.2f}M output tokens of {held.budget:g}M")

            kept.clear()
            return
        stalled = 0 if answered else stalled + 1
        if stalled >= STALLED:
            print(f"stopping: {stalled} rounds in a row answered with nothing")

            return
        time.sleep(5)
