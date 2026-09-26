"""Kimi Code: the app server it serves itself from, where a session is more than its prompts.

``kimi --prompt`` takes a prompt and nothing else. A ``/goal`` written into one is text the model
reads rather than a goal its runtime keeps, there is no flag for swarm mode, the effort an agent
is configured with has nowhere to go, and a turn already running has nowhere to be talked to.
``kimi web`` is the same binary serving the sessions its own browser client drives, and there
all four are things done to the session a turn is submitted to.
"""

from __future__ import annotations

import collections
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import weakref
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, cast

from .base import AgentBase, SessionBase
from .config import AgentConfig
from .event import Event, Failed, Question, Usage, say

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from pydantic import BaseModel

_LISTENING = re.compile(r"^Kimi server: (\S+)/#token=(\S+)$")

SWARM = "swarm"

@dataclass
class _Running:
    """The turn under way, if one is: which session it is in, and what it is running at.

    Written as the turn opens and cleared when it is over, read by whoever wants to put a word
    in. A prompt sent to a session that is working is queued rather than run, and steering it
    is what moves it into the turn already running instead of leaving it for the next one.
    """

    session: str | None = None
    config: dict[str, Any] = field(default_factory=dict[str, Any])

_POLL_SECONDS = 1.0
_CALL_SECONDS = 60.0
_STOP_SECONDS = 5.0

_KINDS = {
    "input": "input_tokens",
    "output": "output_tokens",
    "cache_read": "cache_read_tokens",
    "cache_write": "cache_creation_tokens",
}

_BLOCKS = {"text": "text", "thinking": "reasoning", "tool_use": "tool"}

_PERMITTED = {
    "read-only": {"permission_mode": "auto", "plan_mode": True},
    "workspace-write": {"permission_mode": "auto", "plan_mode": False},
    "auto": {"permission_mode": "auto", "plan_mode": False},
    "bypass": {"permission_mode": "yolo", "plan_mode": False},
}

class _AppServer:
    """A `kimi web` daemon of our own, and the calls one turn of a session is made of."""

    def __init__(self, argv: list[str], env: Mapping[str, str] | None = None) -> None:
        """Starts the daemon and waits for it to say where it is listening.

        Args:
          argv: The command that starts it, already wrapped for wherever its work is to land.
          env: The whole environment to start it in, which is this process's own less what the
            agent's provider hushes and plus what it sets, or None to inherit this one. The
            daemon is the agent's, so its account is the agent's too.

        Raises:
          subprocess.CalledProcessError: If it stops without ever saying, which would leave a
            flow waiting on a server that is not there. Reported as a failed turn, because it
            is the turn that starts it that has nowhere to run.
        """
        self._argv = argv
        self._stopping = threading.Lock()
        self._stopped = False
        self._proc = subprocess.Popen(
            argv,
            
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            env=dict(env) if env else None,
            start_new_session=os.name != "nt",
        )
        assert self._proc.stdout is not None  
        for line in self._proc.stdout:
            if (listening := _LISTENING.match(line.strip())) is not None:
                self._base = f"{listening[1]}/api/v1"
                self._token = listening[2]
                break
        else:
            raise Failed(
                self._proc.wait(), argv, "", f"{argv[0]} stopped without listening"
            )

        threading.Thread(
            target=collections.deque, args=(self._proc.stdout, 0), daemon=True
        ).start()

    def call(self, method: str, path: str, body: Any = None) -> Any:
        """Makes one call to the daemon.

        Args:
          method: The HTTP method to make it with.
          path: The path under the daemon's API root.
          body: What to send as JSON, or None to send nothing.

        Returns:
          What the daemon answered with, unwrapped from its envelope.

        Raises:
          subprocess.CalledProcessError: If it refuses the call or cannot be reached, which is
            a failed turn however it failed -- reported the way every other backend reports one,
            so that a flow catches turns rather than transports.
        """

        request = urllib.request.Request(  
            self._base + path,
            data=None if body is None else json.dumps(body).encode(),
            method=method,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=_CALL_SECONDS) as response:  
                said: dict[str, Any] = json.load(response)
        except urllib.error.HTTPError as refused:
            raise Failed(
                refused.code, self._argv, "", refused.read().decode(errors="replace")
            ) from refused
        except OSError as unreachable:  
            raise Failed(1, self._argv, "", str(unreachable)) from unreachable

        if said.get("code"):
            raise Failed(
                1, self._argv, "", f"{path}: {said.get('msg') or said['code']}"
            )
        return said.get("data")

    def stop(self) -> None:
        """Takes the daemon and its children down, leaving its sessions on disk."""
        with self._stopping:
            if self._stopped:
                return
            self._stopped = True
            if os.name == "nt":
                if self._proc.poll() is None:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=_STOP_SECONDS)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                self._proc.wait()
                return

            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._proc.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=_STOP_SECONDS)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._proc.pid, signal.SIGKILL)
            self._proc.wait()

@dataclass(frozen=True, kw_only=True)
class KimiCodeCLIAgentConfig(AgentConfig):
    """What Kimi Code is configured with: the common model, and an effort that says width too.

    Attributes:
      effort: How hard to think, in Kimi's own wording, optionally prefixed `swarm` to run
        every turn as a fleet of subagents rather than as one agent -- `max` and `swarmmax` are
        the same thinking at either width.
    """

class KimiCodeCLISession(SessionBase):
    """A Kimi Code conversation, held by the app server and named by it as it opens.

    The id is the server's, handed out before the turn rather than read back out of a resume
    hint, so a second agent working alongside cannot be resumed by mistake.
    """

    _agent: KimiCodeCLIAgent  

    def __init__(
        self, agent: AgentBase, cwd: str | os.PathLike[str] | None = None
    ) -> None:
        """Initializes a session running nothing yet.

        Args:
          agent: The agent whose config every turn of this session runs at.
          cwd: The directory this conversation works in, as for `SessionBase`.
        """
        super().__init__(agent, cwd)
        
        self._running = _Running()

        self._counted: Counter[str] = Counter()

    @property
    def named(self) -> str | None:
        """The session the daemon holds, which it names as the turn opens it."""
        return self._id or self._running.session

    def interject(self, text: str) -> None:
        """Puts a word into the turn already running, rather than into the next one.

        A prompt sent to a session that is working is queued, and would be answered as a turn
        of its own once this one ended. Steering moves it into the turn that is running, which
        is what makes it a word put in rather than a turn queued behind.

        Args:
          text: What to say to the agent.

        Raises:
          RuntimeError: If no turn is running, so there is none to steer it into.
          subprocess.CalledProcessError: If the daemon refuses either call.
        """
        running = self._running
        if running.session is None:
            raise RuntimeError("no turn is running to be talked to")

        self.steering(text, ticket=text)
        server = self._agent.server
        queued = server.call(
            "POST",
            f"/sessions/{running.session}/prompts",
            {"content": [{"type": "text", "text": text}], **running.config},
        )
        try:
            server.call(
                "POST",
                f"/sessions/{running.session}/prompts:steer",
                {"prompt_ids": [queued["prompt_id"]]},
            )
        except BaseException:
            self.unsteered(text)  
            raise

    def _stream(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,  
    ) -> Iterator[Event]:
        """Sends one turn, opening the session on the first call and resuming it after.

        The daemon answers a turn whole, but it writes the turn down as it goes: what the
        agent said is read back as it is written, which is what makes the turn watchable.

        A shape is not a setting of a turn here -- the daemon takes a prompt and nothing
        else about the answer -- so a turn asked for one has already been asked in the
        prompt, which is what :attr:`SessionBase.shapes` says of this backend.
        """
        yield from self._submit(prompt, goal=False)

    def _asked(self, session: str) -> None:
        """Answers whatever the turn has stopped to ask, if it has stopped to ask anything.

        The daemon holds a question until it is answered and the turn waits on it, so a poll
        that only read messages would read a session that never moves again. An answer is
        matched to one of the options where it names one, since a question that offered them
        need not take anything else; a question nobody is there to answer is skipped, which
        the tool reads as an answer and carries on from.

        Args:
          session: The session the turn is running in.
        """
        server = self._agent.server
        held = server.call("GET", f"/sessions/{session}/questions")

        waiting: list[Any] = (
            cast("dict[str, Any]", held).get("items") or []
            if isinstance(held, dict)
            else held or []
        )
        for raw in waiting:
            pending = cast("dict[str, Any]", raw)
            if not pending.get("question_id"):
                continue  
            answers: dict[str, dict[str, Any]] = {}
            for asked in cast("list[Any]", pending.get("questions") or []):
                question = cast("dict[str, Any]", asked)
                offers: list[Any] = question.get("options") or []
                options = {
                    str(cast("dict[str, Any]", option).get("label", "")).lower(): cast(
                        "dict[str, Any]", option
                    ).get("id")
                    for option in offers
                    if isinstance(option, dict)
                }
                said = self._agent.asked(
                    Question(
                        text=str(
                            question.get("question") or question.get("header") or ""
                        ),
                        options=tuple(
                            str(cast("dict[str, Any]", option)["label"])
                            for option in offers
                            if isinstance(option, dict)
                            and cast("dict[str, Any]", option).get("label")
                        ),
                    )
                )
                if said is None:
                    answers[str(question["id"])] = {"kind": "skipped"}
                elif (chosen := options.get(said.strip().lower())) is not None:
                    answers[str(question["id"])] = {
                        "kind": "single",
                        "option_id": chosen,
                    }
                else:
                    answers[str(question["id"])] = {"kind": "other", "text": said}
            server.call(
                "POST",
                f"/sessions/{session}/questions/{pending['question_id']}",
                {"answers": answers},
            )

    def _counting(self, server: _AppServer, session: str) -> Usage:
        """What this session has spent since the last time it was asked.

        The daemon counts the whole conversation, so what has been spent since is the rise
        across it. Told to the meters as it is read, which is what a rate taken while the turn
        is still running is made of.

        Args:
          server: The daemon holding the session.
          session: The session to ask about.

        Returns:
          The rise since the last reading, by kind, which is nothing at all where it has not
          moved.

        Raises:
          subprocess.CalledProcessError: If the daemon will not say, which is not a failed
            turn: what a run costs is worth nothing at the price of the run.
        """
        held = server.call("GET", f"/sessions/{session}")
        usage: dict[str, Any] = (
            cast("dict[str, Any]", held).get("usage") or {}
            if isinstance(held, dict)
            else {}
        )
        counted = Counter(
            {
                kind: int(usage.get(named) or 0)
                for kind, named in _KINDS.items()
                if usage.get(named)
            }
        )
        risen = Usage(
            {
                kind: tokens
                for kind in set(counted) | set(self._counted)
                if (tokens := counted[kind] - self._counted[kind]) > 0
            }
        )
        self._counted = counted
        self._spends(risen)
        return risen

    def _pursue(self, objective: str) -> str:
        """Runs the turn under a goal of Kimi's own, which its runtime steers until it is met.

        The objective is the prompt as well as the goal, which is what ``/goal`` does: the
        agent is told what to do, and the runtime is told what it is for.

        Returns:
          The agent's response once the goal is done with, stripped.
        """
        said = ""
        for event in self._submit(objective, goal=True):

            self._agent._heard(event)
            if event.kind == "result":
                said = event.text
        return said.strip()

    def _submit(self, prompt: str, *, goal: bool) -> Iterator[Event]:
        """Runs one turn, saying what the agent says as it says it.

        A goal-driven turn is many turns of the model, and it is over when the session falls
        idle rather than when the first of them ends. The session is asked whether it is still
        running before its messages are read, so that nothing said between the two is missed.

        Args:
          prompt: The input prompt for this turn, which is the objective as well when it is
            a goal the session is being set.
          goal: Whether to set the prompt as the session's goal before sending it.

        Yields:
          What the agent said, in the order it said it, and the answer it ended on.

        Raises:
          subprocess.CalledProcessError: If the daemon refuses any of the calls a turn is made
            of, leaving the session unopened so that the next call retries the turn.
        """
        effort = self.effort
        turn: dict[str, Any] = {
            "model": self._agent.config.model,
            "thinking": effort.removeprefix(SWARM),
            "swarm_mode": effort.startswith(SWARM),

            **_PERMITTED.get(self._agent.config.permission, _PERMITTED["bypass"]),
        }
        with self._lock:  

            try:
                server = self._agent.server

                if (session := self._id) is None:
                    session = server.call(
                        "POST", "/sessions", {"metadata": {"cwd": self._workspace()}}
                    )["id"]
                server.call(
                    "POST",
                    f"/sessions/{session}/profile",
                    {
                        "agent_config": turn
                        | ({"goal_objective": prompt} if goal else {})
                    },
                )

                self._running = _Running(session=session, config=turn)
                since = server.call(
                    "POST",
                    f"/sessions/{session}/prompts",
                    {"content": [{"type": "text", "text": prompt}], **turn},
                )["user_message_id"]
                answer = ""
                shown: dict[
                    str, int
                ] = {}  
                settled = False
                costing = Usage()  
                while True:

                    with contextlib.suppress(subprocess.CalledProcessError):
                        self._asked(session)
                    busy = server.call("GET", f"/sessions/{session}/status")["busy"]

                    with contextlib.suppress(subprocess.CalledProcessError):
                        costing = costing + self._counting(server, session)
                    if goal and not busy:

                        pursued = server.call("GET", f"/sessions/{session}/goal")
                        busy = pursued is not None and pursued["status"] == "active"
                    said = server.call(
                        "GET", f"/sessions/{session}/messages?after_id={since}"
                    )["items"]

                    for message in reversed(said):
                        if message["role"] != "assistant":

                            if message["id"] not in shown:
                                shown[message["id"]] = len(message["content"])
                                words = "".join(
                                    block.get("text") or ""
                                    for block in message["content"]
                                    if block.get("type") == "text"
                                )
                                if self.took(words) is not None:
                                    yield Event(kind="took", text=words)
                            continue
                        for block in message["content"][shown.get(message["id"], 0) :]:
                            kind = _BLOCKS.get(str(block.get("type")))

                            words = str(
                                (
                                    block.get("tool_name")
                                    if kind == "tool"
                                    else block.get(str(block.get("type")))
                                )
                                or ""
                            )
                            if kind is None or not words.strip():
                                continue
                            if not self._agent._watchers:

                                say(words, sys.stderr)
                            yield Event(kind=kind, text=words)
                        shown[message["id"]] = len(message["content"])

                    for (
                        message
                    ) in said:  
                        text = "".join(
                            block["text"]
                            for block in message["content"]
                            if block["type"] == "text"
                        )
                        if message["role"] == "assistant" and text:
                            answer = text
                            break
                    if settled:

                        self._adopt(session)
                        if not self._agent._watchers:

                            say(answer, sys.stdout)

                        with contextlib.suppress(subprocess.CalledProcessError):
                            costing = costing + self._counting(server, session)
                        spent = (
                            {self._agent.config.model: int(costing.total)}
                            if costing.total > 0
                            else {}
                        )
                        yield Event(
                            kind="result",
                            text=answer.strip(),
                            tokens=spent,
                            spent=costing,
                        )
                        return

                    settled = not busy
                    time.sleep(_POLL_SECONDS)

            finally:
                self._running = _Running()

class KimiCodeCLIAgent(AgentBase):
    """Kimi Code, driven through an app server of its own so a whole session is settable."""

    pursues: ClassVar[bool] = True

    def __init__(self, config: AgentConfig, *, name: str | None = None) -> None:
        """Initializes an agent whose server is not running yet.

        Args:
          config: The model and effort every session of this agent runs at.
          name: What to call this agent, defaulting to one nothing else answers to.
        """
        super().__init__(config, name=name)
        self._server: _AppServer | None = None

        self._server_as = ""
        self._serving = threading.Lock()

    @property
    def server(self) -> _AppServer:
        """The daemon this agent's turns are submitted to, started the first time it is asked for.

        One per agent rather than one per session, so that a flow dropping a session a turn
        does not start a server a turn; it is taken down when the agent is collected, or at
        exit for one held to the end. An anchored agent starts it through coganchor, which
        leaves the server here, holding the conversation, and its work on the target -- the
        same split the CLI ran under.
        """
        with (
            self._serving
        ):  
            if self._server is not None and self._server_as != self.node().name:

                self._server, self._server_as = None, ""
            if self._server is None:
                argv = [
                    "kimi",
                    "web",
                    "--no-open",
                    "--port",
                    "0",
                    "--log-level",
                    "error",
                ]

                account = self.node().name
                self._server = _AppServer(self.spawned(argv), self._environ())
                self._server_as = account

                weakref.finalize(self, self._server.stop)
            return self._server

    def stop(self) -> None:
        """Takes no further turn, and takes down the server the turn under way is waiting on."""
        super().stop()
        if self._server is not None:
            self._server.stop()
            self._server = None

    def new(self, cwd: str | os.PathLike[str] | None = None) -> KimiCodeCLISession:
        """Opens a new Kimi session, in the directory it is given or in this one."""
        return KimiCodeCLISession(self, cwd)
