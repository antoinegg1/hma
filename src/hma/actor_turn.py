'Exactly one fresh NoGoals session; the external supervisor owns alternation.'

from typing import Annotated, Any

from hmz.flows import Agent, AgentDefaults, flow

NoGoals = Annotated[Agent, AgentDefaults(goals=False)]

def close_session(session: Any) -> None:
    'Unwind the provider; the supervisor additionally stops the whole container.'
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
def run(agents: tuple[NoGoals], task: str) -> None:
    'Natural return is enough: never continue just to reach the submission cap.'
    session = agents[0].new()
    try:
        session(task, suppress=True)
    finally:
        close_session(session)
