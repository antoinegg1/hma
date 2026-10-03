"Exactly one fresh session; native goals are enabled only in the goal baseline."

import os
from typing import Annotated, Any

from hmz.flows import Agent, AgentDefaults, flow

TaskAgent = Annotated[Agent, AgentDefaults(goals=os.environ.get("HMA_WORKFLOW") == "goal")]


def close_session(session: Any) -> None:
    "Unwind the provider; the supervisor additionally stops the whole container."
    try:
        session.close()
    finally:
        owner = getattr(session, "_agent", None)
        down = getattr(owner, "_down", None)
        if callable(down):
            down()
        else:
            server = getattr(owner, "_server", None)
            if server is not None:
                server.stop()
                owner._server = None


@flow
def run(agents: tuple[TaskAgent], task: str) -> None:
    "Natural return is enough: never continue just to reach the submission cap."
    session = agents[0].new()
    try:
        if os.environ.get("HMA_WORKFLOW") == "goal":
            session.pursue(task, suppress=False)
        else:
            session(task, suppress=False)
    finally:
        close_session(session)
