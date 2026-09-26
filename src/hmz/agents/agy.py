"""agy: one run of Antigravity CLI per turn, reading the stream of JSON it answers in.

Its command line says everything an agent is configured with -- the model, how hard to think,
the conversation to carry on -- so a turn is one run of it rather than a conversation held open
on a server. What it writes on stdout with `--output-format stream-json` is a protocol rather
than the agent talking: one JSON object a line, tagged by `event`, opening on the `init` that
names the conversation and ending on the `result` that says what the turn came to.

What it may do is the one thing it cannot be told by halves: `--dangerously-skip-permissions`
is the whole of the lever, so the rungs below it are refused rather than quietly ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

from .base import AgentBase, CommandSessionBase
from .config import AgentConfig
from .event import Event, Failed, Usage

if TYPE_CHECKING:
    import os
    from collections.abc import Iterator

_COMMAND = "agy"

_TAKES = ("auto", "bypass")

_THINKING = "THINKING"

class AntigravityCLISession(CommandSessionBase):
    """An Antigravity conversation, resumed by the id its first turn reported.

    The id is minted by `agy` as the conversation opens and named on the line the stream opens
    with, so it is read back out of the turn that opened it and given to every turn after --
    it takes no id of its own choosing.
    """

    protocol: ClassVar[bool] = True

    shapes: ClassVar[bool] = True

    def __init__(
        self, agent: AgentBase, cwd: str | os.PathLike[str] | None = None
    ) -> None:
        """Initializes a session that has run no turn yet.

        Args:
          agent: The agent whose config every turn of this session runs at.
          cwd: The directory this conversation works in, as for `SessionBase`.
        """
        super().__init__(agent, cwd)
        
        self._said = ""
        self._failed: str | None = None
        
        self._costing = Usage()

    def _turn(self, prompt: str) -> tuple[list[str], str | None]:
        """Builds the `agy -p` one turn is.

        Args:
          prompt: The input prompt for this turn.

        Returns:
          The command, and None because the prompt is inside it: Antigravity CLI reads no
          prompt off stdin.
        """
        self._said, self._failed = "", None
        self._costing = Usage()
        argv = [
            _COMMAND,
            "--output-format",
            "stream-json",
            "--model",
            self._agent.config.model,
            
            "--dangerously-skip-permissions",
        ]

        if (schema := self._shaping) is not None:
            argv += ["--json-schema", json.dumps(schema.model_json_schema())]
        if self._id is not None:
            argv += ["--conversation", self._id]

        return [*argv, "--print", prompt], None

    def _reads(self, line: str, *, error: bool) -> Iterator[Event]:
        """Reads one line Antigravity CLI wrote, as the things it says the agent did.

        Args:
          line: The line, as written.
          error: Whether it came from stderr, which is its own log rather than the turn --
            kept for a failed turn's diagnostic and shown nowhere.

        Yields:
          What it said, which is nothing for a line saying nothing worth showing.
        """
        if error:
            return
        try:
            said: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            return  
        kind = str(said.get("event") or "")
        
        told = cast("dict[str, Any]", said.get(kind) or {})
        if kind == "step_update":
            yield from self._step(told)
        elif kind == "result":
            self._said = str(told.get("response") or "")
            self._costing = self._costing + self._cost(
                cast("dict[str, Any]", told.get("usage") or {})
            )

            status = str(told.get("status") or "")
            if failed := str(told.get("error") or ""):
                self._failed = failed
            elif status and status != "SUCCESS":
                self._failed = status

    def _step(self, told: dict[str, Any]) -> Iterator[Event]:
        """Reads one step of the turn, which is a piece of an answer or a tool going by.

        Args:
          told: The `step_update` payload, as read.

        Yields:
          What that step said, which is nothing for one that only moved a state along.
        """
        if words := str(told.get("text_delta") or ""):
            kind = (
                "reasoning" if str(told.get("step_type") or "") == _THINKING else "text"
            )
            yield Event(kind=kind, text=words)
            return
        
        named = str(told.get("tool_name") or "")
        if named and str(told.get("state") or "") != "DONE":
            about: dict[str, Any] = told.get("tool_info") or {}
            first = next(
                (
                    str(value)
                    for value in about.values()
                    if isinstance(value, str) and value.strip()
                ),
                "",
            )
            yield Event(kind="tool", text=f"{named} {first}".strip()[:120])

    def _cost(self, counted: dict[str, Any]) -> Usage:
        """What the turn cost, by the kind each token went on.

        Args:
          counted: The `usage`, as read.

        Returns:
          What it spent.
        """
        return Usage(
            {
                name: float(counted.get(name) or 0)
                for name in (
                    "input_tokens",
                    "output_tokens",
                    "thinking_tokens",
                    "cache_read_tokens",
                )
                if counted.get(name)
            }
        )

    def _result(self, transcript: str) -> Event:
        """The turn's answer, and what it cost, out of the lines it wrote.

        Args:
          transcript: The whole of stdout, already read line by line.

        Returns:
          The `result` the turn ends on.

        Raises:
          subprocess.CalledProcessError: If the turn failed, which it says in a word of its own
            as well as in its exit status.
        """
        if self._failed is not None:
            raise Failed(1, [_COMMAND], self._said, self._failed)
        if not transcript.strip():
            raise Failed(1, [_COMMAND], "", f"{_COMMAND} said nothing at all")
        spent = int(self._costing.total)
        return Event(
            kind="result",
            text=self._said.strip(),
            tokens={self._agent.config.model: spent} if spent > 0 else {},
            spent=self._costing,
        )

    def _read_session_id(self, transcript: str) -> str:
        """Reads back the conversation it opened, which the line it opens with names.

        Args:
          transcript: Everything the turn printed.

        Returns:
          The conversation's id.

        Raises:
          ValueError: If nothing the turn wrote names one, which is a turn that landed
            somewhere nobody can find again.
        """
        for line in transcript.splitlines():
            try:
                said: object = json.loads(line)
            except ValueError:
                continue
            if not isinstance(said, dict):
                continue
            held = cast("dict[str, Any]", said)
            
            named = held.get("conversation_id") or cast(
                "dict[str, Any]", held.get("result") or {}
            ).get("conversation_id")
            if named:
                return str(named)
        raise ValueError(f"{_COMMAND} named no conversation")

@dataclass(frozen=True, kw_only=True)
class AntigravityCLIAgentConfig(AgentConfig):
    """What Antigravity CLI is configured with: the common model and effort, and nothing else.

    The model is written as `agy models` lists it, which is a slug of its own -- and the
    effort is part of that slug, `gemini-3.7-flash-low` being that model at that effort. So
    the effort here is what a model was chosen at rather than something the CLI is told: it
    refuses the flag beside every model it lists.
    """

class AntigravityCLIAgent(AgentBase):
    """Antigravity CLI, driven through its own command line, one run per turn."""

    def __init__(self, config: AgentConfig, *, name: str | None = None) -> None:
        """Makes the agent, refusing a rung this backend has no way of running at.

        Args:
          config: What its turns run at.
          name: What the flow calls it, or None for one of its own.

        Raises:
          ValueError: If it was configured to be allowed less than everything. Antigravity CLI
            has one switch -- approve every tool, or stop and ask -- and nobody is at a prompt
            to be asked, so a rung it cannot express is said here rather than quietly run as
            the rung above it.
        """
        if config.permission not in _TAKES:
            raise ValueError(
                f"{_COMMAND} runs at {' or '.join(_TAKES)} only, "
                f"not {config.permission}: it has no way of being allowed less"
            )
        super().__init__(config, name=name)

    def new(self, cwd: str | os.PathLike[str] | None = None) -> AntigravityCLISession:
        """Opens a new Antigravity conversation, in the directory it is given or in this one."""
        return AntigravityCLISession(self, cwd)
