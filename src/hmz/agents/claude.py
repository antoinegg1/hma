"""Claude Code: one ``claude --print`` held open, spoken to in JSON a line at a time."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

from .base import AgentBase, StreamSessionBase
from .config import AgentConfig
from .event import Event, Question, Usage
from .hooks import EVERYWHERE, SUBAGENTS, Moment

if TYPE_CHECKING:
    import os
    from collections.abc import Iterator

_ASKS = "AskUserQuestion"

_CONTINUATION_TOOLS = (
    "Agent",
    "ScheduleWakeup",
    "CronCreate",
    "CronDelete",
    "CronList",
)

_WEB_TOOLS = ("WebSearch", "WebFetch")

_FLEET = ("Task", "Agent")

_ALLOWED_TOOLS_MAX = 32

_ALLOWED_TOOL_RULE_MAX_CHARS = 4096

_UNFINISHED = frozenset(
    {
        "max_tokens",
        "model_context_window_exceeded",
        "pause_turn",
        "tool_deferred",
        "tool_use",
    }
)

_PERMITTED = {
    "read-only": "plan",
    "workspace-write": "acceptEdits",
    "auto": "auto",
    "bypass": "manual",
}

_KINDS = {
    "input": "inputTokens",
    "output": "outputTokens",
    "cache_read": "cacheReadInputTokens",
    "cache_write": "cacheCreationInputTokens",
}
_AS_IT_GOES = {
    "input": "input_tokens",
    "output": "output_tokens",
    "cache_read": "cache_read_input_tokens",
    "cache_write": "cache_creation_input_tokens",
}

def _about(called: dict[str, Any]) -> str:
    """What a tool was called with, as the one line a row of a transcript has room for.

    Args:
      called: The tool's input, as Claude sent it.

    Returns:
      The first thing in it that is words -- the path, the command, the query -- or "".
    """
    return next(
        (
            str(value)
            for value in called.values()
            if isinstance(value, str) and value.strip()
        ),
        "",
    )

def _result_failure(said: dict[str, Any]) -> str | None:
    """Explains why a Claude result did not finish its turn, or says that it did.

    A turn held to a shape ends the one way that otherwise reads as unfinished: the last
    thing the model did was call `StructuredOutput`, so the result says `stop_reason:
    tool_use` -- and says the object beside it, which is the answer. So a result carrying
    one is a turn that finished, however it stopped.
    """
    reason: str | None = None
    shaped = said.get("structured_output") is not None
    if said.get("is_error"):
        reason = "the turn failed"
    elif (subtype := said.get("subtype")) not in (None, "success"):
        reason = f"Claude ended the turn with {subtype}"
    elif (terminal := said.get("terminal_reason")) not in (None, "completed"):
        reason = f"Claude ended the turn with {terminal}"
    elif not shaped and (stopped := said.get("stop_reason")) in _UNFINISHED:
        reason = f"Claude stopped with {stopped} before completing the turn"
    if reason is None:
        return None

    if result := said.get("result"):
        return str(result)
    errors = cast("list[Any]", said.get("errors") or [])
    if errors:
        return "; ".join(str(error) for error in errors)
    return reason

@dataclass(frozen=True, kw_only=True)
class ClaudeCodeAgentConfig(AgentConfig):
    """The common settings plus exact Claude-native tool allow rules."""

    allowed_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            len(self.allowed_tools) > _ALLOWED_TOOLS_MAX
            or self.allowed_tools != tuple(sorted(set(self.allowed_tools)))
            or any(
                not rule or len(rule) > _ALLOWED_TOOL_RULE_MAX_CHARS or "," in rule
                for rule in self.allowed_tools
            )
        ):
            raise ValueError("allowed_tools must be unique sorted Claude tool rules")

class ClaudeCodeSession(StreamSessionBase):
    """A Claude Code conversation, addressed by an id chosen up front.

    Pinning beats ``--continue``, which resumes whichever session in this directory is newest:
    a second agent working alongside would steal the resume.

    The process stands for the life of the session rather than the length of a turn, which is
    what streaming input buys: the turns of one conversation are lines written to a Claude that
    is already there, and so is anything said to it while a turn is running.
    """

    shapes: ClassVar[bool] = True

    takes_tools: ClassVar[bool] = True

    def __init__(
        self, agent: AgentBase, cwd: str | os.PathLike[str] | None = None
    ) -> None:
        """Initializes a session that has spent nothing yet.

        Args:
          agent: The agent whose config every turn of this session runs at.
          cwd: The directory this conversation works in, as for `SessionBase`.
        """
        super().__init__(agent, cwd)

        self._counted: dict[str, Counter[str]] = {}

        self._fed: Counter[str] = Counter()
        self._seen: dict[str, Counter[str]] = {}

        self._at: str | None = None
        
        self._named: str | None = None

        self._fleet: dict[str, str] = {}

        self._offering: tuple[str, ...] | None = None

        self._telling: tuple[str, ...] = ()

    @property
    def named(self) -> str | None:
        """What Claude called this session, which it says on the first line it writes."""
        return self._id or self._named

    def _command(self) -> list[str]:
        """Builds the ``claude --print`` that reads turns from stdin and says events on stdout.

        Opens the session while it is unopened and resumes it once it has an id, which is what
        an anchored session needs: its process ends with each turn, so the next one has a
        conversation to rejoin. An unanchored session opens once and stays open.
        """

        pinned = self._id or str(uuid.uuid4())
        argv = [
            "claude",
            "--print",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--resume" if self._id else "--session-id",
            pinned,
            "--permission-mode",
            _PERMITTED[self._agent.config.permission],
            *(

                ["--permission-prompt-tool", "stdio"]
                if self._agent.config.permission == "bypass"
                else []
            ),
            "--settings",
            json.dumps(
                {"fastMode": self._agent.config.service_tier == "fast"},
                separators=(",", ":"),
            ),
            "--model",
            self._agent.config.model,
            "--effort",
            self.effort,
        ]
        if self._shaping is not None:

            argv += ["--json-schema", json.dumps(self._shaping.model_json_schema())]

        denied: list[str] = []
        if not self._agent.goals_enabled:
            denied += _CONTINUATION_TOOLS
        if not self._agent.config.web_search:
            denied += _WEB_TOOLS
        if denied:
            argv += ["--disallowedTools", ",".join(denied)]
        allowed_tools = getattr(self._agent.config, "allowed_tools", ())
        if allowed_tools:
            argv += ["--allowedTools", ",".join(allowed_tools)]

        self._telling = self._offered()
        if self._telling:

            argv += [
                "--mcp-config",
                json.dumps(self._agent.toolbox.config(), separators=(",", ":")),
            ]
        return argv

    def _write(self, text: str, ticket: str = "") -> str:
        """Renders one thing to say as the user message Claude reads it as.

        A word put into a turn carries a `uuid`, which is what Claude names it by in the
        `command_lifecycle` lines it answers with -- so a turn told three things says which
        of them it has taken in, one at a time. Without one it says nothing at all, and a
        word put in would only ever be as good as the write that sent it.

        Args:
          text: What to say.
          ticket: The uuid to name it by, or "" for a turn's own prompt: the turn beginning
            is what says that one landed.

        Returns:
          The line, newline included.
        """
        said: dict[str, Any] = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            },
        }
        if ticket:
            said["uuid"] = ticket
        return json.dumps(said) + "\n"

    def _restarted(self) -> None:
        """Forgets what the last process had spent, which the new one has not counted."""
        self._counted, self._fed, self._seen = {}, Counter(), {}
        
        self._fleet = {}
        self._at = self.effort
        self._offering = self._telling

    def _offered(self) -> tuple[str, ...]:
        """What a Claude started now would be told the flow's own callbacks are.

        Returns:
          The name of every callback in front of the agent, sorted, so that a flow which
          builds its list afresh before each turn is offering the same thing each time.
          Names rather than everything a tool says: a name is what a tool is here -- two
          conversations offering one name are offering one tool -- and building every
          argument schema again before every turn to catch a reworded sentence would cost
          each turn more than the sentence is worth.
        """
        return tuple(sorted(one.name for one in self._agent.toolbox.offered()))

    def _stale(self) -> bool:
        """Whether the process up was started for something this turn is no longer.

        `--effort` is an argument of the process, so a flow that moves it mid-session is
        answered by ending this one and resuming the conversation in a process started at the
        new one -- exactly as asking for a shape is. So is `--mcp-config`: Claude reads what
        an MCP server has when it starts it and holds that list for the life of the process,
        so a flow that changes which callbacks it offers between two turns is answered the
        same way. What is compared is the list the process was actually told, not whether it
        was told anything: a tool swapped for another is one the model has never heard of and
        one it can still reach for and be told is not there.
        """
        if self._at is not None and self._at != self.effort:
            return True
        return self._offering is not None and self._offering != self._offered()

    def _spent(self, said: dict[str, Any]) -> tuple[dict[str, int], Usage]:
        """What the turn just ending cost, per model and by the kind it went on.

        Claude reports each model's usage as a running total for the session, so what this
        turn cost is the rise since the last one. Every kind of token counts: what a rate is
        measuring is the traffic, and a cache read crosses the wire like anything else.

        Args:
          said: The `result` event, as read.

        Returns:
          Tokens spent per model since the previous turn, models that did not move omitted,
          and the same spending by kind.
        """
        spent: dict[str, int] = {}
        risen: Counter[str] = Counter()
        used: dict[str, Any] = said.get("modelUsage") or {}
        for model, usage in used.items():
            counted = Counter(
                {
                    kind: int(usage.get(named) or 0)
                    for kind, named in _KINDS.items()
                    if usage.get(named)
                }
            )
            before = self._counted.get(model) or Counter()
            moved = Counter(
                {
                    kind: tokens
                    for kind in set(counted) | set(before)
                    if (tokens := counted[kind] - before[kind]) > 0
                }
            )
            if total := sum(moved.values()):
                spent[model] = total
            risen.update(moved)
            self._counted[model] = counted
        return spent, Usage(risen)

    def _live(self, said: dict[str, Any]) -> None:
        """Takes what one request to the model came to, as its answer arrives.

        Claude says what each of them cost on the message it produced, which is where a rate
        read while the turn is still running comes from -- the `result` at the end of the turn
        is minutes away, and a rate that only moved there would stand still for all of them.
        What the result then states is the whole of the turn, so only the shortfall is added.

        Args:
          said: The `assistant` event, as read.
        """
        message: dict[str, Any] = said.get("message") or {}
        usage: dict[str, Any] = message.get("usage") or {}

        named = str(message.get("id") or "")
        counted: Counter[str] = Counter(
            {
                kind: int(usage.get(spelled) or 0)
                for kind, spelled in _AS_IT_GOES.items()
                if usage.get(spelled)
            }
        )
        before = self._seen.get(named) or Counter()
        risen = Usage(
            {
                kind: tokens
                for kind in set(counted) | set(before)
                if (tokens := counted[kind] - before[kind]) > 0
            }
        )
        self._seen[named] = counted
        if risen.total:
            self._fed.update(risen)
            self._spends(risen)

    def _settle(self, risen: Usage) -> None:
        """Adds whatever the turn's own total says was spent beyond what was counted live.

        Args:
          risen: What the turn cost, by kind, as the `result` states it.
        """
        owed = Usage(
            {
                kind: tokens
                for kind in set(risen) | set(self._fed)
                if (tokens := risen.get(kind, 0.0) - self._fed[kind]) > 0
            }
        )
        self._fed, self._seen = Counter(), {}

        self._spends(owed, turn=False)

    def _read(self, line: str) -> Iterator[Event]:
        """Reads one event Claude wrote, as the things it says the agent did.

        A message carries a list of parts, and thinking, speaking and reaching for a tool can
        all be in the same one -- so every part is read, not the first that says anything.

        Args:
          line: The line, as written.

        Yields:
          What it said, which is nothing for a line saying nothing worth showing: a partial
          chunk, a tool's result coming back, or something a later Claude has added.
        """
        try:
            said: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            return  
        if said.get("type") == "control_request":
            
            self._answer(said)
        elif said.get("type") == "command_lifecycle":

            if said.get("state") == "started":
                words = self.took(str(said.get("command_uuid") or ""))
                if words is not None:
                    yield Event(kind="took", text=words)
        elif said.get("type") == "system" and said.get("session_id"):

            self._named = str(said["session_id"])
        elif said.get("type") == "result":
            if failure := _result_failure(said):

                tokens, risen = self._spent(said)
                self._settle(risen)
                yield Event(
                    kind="failed",
                    text=failure,
                    tokens=tokens,
                    spent=risen,
                )
                return
            if self._named is not None:
                self._adopt(self._named)  
            tokens, risen = self._spent(said)
            self._settle(risen)
            yield Event(
                kind="result",
                text=str(said.get("result") or ""),
                tokens=tokens,
                spent=risen,
            )
        elif said.get("type") == "assistant":
            self._live(said)
            for part in said.get("message", {}).get("content", []):
                if part.get("type") == "text" and part.get("text", "").strip():
                    yield Event(kind="text", text=part["text"])
                elif (
                    part.get("type") == "thinking" and part.get("thinking", "").strip()
                ):
                    yield Event(kind="reasoning", text=part["thinking"])
                elif part.get("type") == "tool_use":

                    called: dict[str, Any] = part.get("input") or {}
                    named = str(part.get("name") or "tool")
                    said_as = f"{named} {_about(called)}".strip()[:120]
                    if named in _FLEET:
                        marked = str(part.get("id") or "")
                        self._fleet[marked] = said_as
                        yield Event(kind="subagent", text=said_as, whose=marked)
                        continue
                    yield Event(kind="tool", text=said_as)
        elif said.get("type") == "user":

            for part in said.get("message", {}).get("content", []):
                if part.get("type") != "tool_result":
                    continue
                marked = str(part.get("tool_use_id") or "")
                if was := self._fleet.pop(marked, ""):
                    yield Event(kind="subagent-ends", text=was, whose=marked)

    def _answer(self, said: dict[str, Any]) -> None:
        """Answers something Claude asked of us over the same stream the turn is read from.

        Two kinds arrive here. The tool Claude uses to ask a person a question is put to the
        person. Everything else is a permission -- and under `bypass`, where Claude runs at
        `manual` and asks before every tool that would change something, every one of those
        asks lands here, which is humanize taking the deciding rather than skipping it. A flow
        watches its agent rather than gating it, so those are allowed with the input they came
        with, unless something hung on `PermissionRequest` says otherwise: that is the one
        moment a refusal actually stops the agent, because it is the one the backend waits on.
        What the account itself will not allow at all -- the hard `deny` list an organisation
        ships -- the CLI refuses before it ever asks, so a yes here is a yes to what the
        account leaves decidable and nothing more. A question nobody is there to answer is
        refused, which Claude reads as the tool having been declined and carries on from,
        rather than waiting on a reply that is not coming.

        An agent that may change nothing is the exception: a permission is a request to do
        something, and granting one under `read-only` would be handing back the rung the flow
        asked for. Claude in plan mode asks rather than acts, and the answer here is no.

        Args:
          said: The `control_request`, as read.
        """
        asked: dict[str, Any] = said.get("request") or {}
        called: dict[str, Any] = asked.get("input") or {}
        answers: dict[str, str] = {}
        tool = str(asked.get("tool_name") or "")
        if tool != _ASKS:
            asking = self._fire(
                Moment.PERMISSION_REQUEST,
                tool=tool,
                about=_about(called),
                called=called,
            )
            if self._agent.config.permission == "read-only":
                self._reply(
                    said,
                    {"behavior": "deny", "message": f"{tool} would change something"},
                )
                return
            if asking.refused:
                self._reply(
                    said,
                    {
                        "behavior": "deny",
                        "message": asking.because or f"{tool} was refused",
                    },
                )
                return
        else:
            for raw in cast("list[Any]", called.get("questions") or []):
                question = cast("dict[str, Any]", raw)
                wanted = str(question.get("question") or question.get("header") or "")
                offers: list[Any] = question.get("options") or []
                offered = tuple(
                    str(cast("dict[str, Any]", option)["label"])
                    for option in offers
                    if isinstance(option, dict)
                    and cast("dict[str, Any]", option).get("label")
                )
                answer = self._agent.asked(Question(text=wanted, options=offered))
                if answer is None:
                    self._reply(said, {"behavior": "deny", "message": "nobody to ask"})
                    return
                answers[wanted] = answer
        self._reply(
            said,
            {
                "behavior": "allow",
                "updatedInput": {**called, "answers": answers} if answers else called,
            },
        )

    def _reply(self, said: dict[str, Any], answer: dict[str, Any]) -> None:
        """Writes one answer back to Claude, against the request it answers.

        Args:
          said: The `control_request` being answered.
          answer: What to answer it with.
        """
        self._send(
            json.dumps(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": said.get("request_id"),
                        "response": answer,
                    },
                }
            )
            + "\n"
        )

    def _pursue(self, objective: str) -> str:
        """Runs the turn as Claude Code's own ``/goal``, which print mode expands like any other.

        Claude keeps the session going itself, by refusing to stop while the objective is
        unmet, so the turn is over only once it has been reached or given up on.
        """
        return self(f"/goal {objective}")

class ClaudeCodeAgent(AgentBase):
    """Claude Code, driven over its streaming JSON protocol so a turn can be talked to."""

    service_tiers = ("default", "fast")

    moments: ClassVar[frozenset[Moment]] = (
        EVERYWHERE | SUBAGENTS | {Moment.PERMISSION_REQUEST}
    )

    pursues: ClassVar[bool] = True

    def new(self, cwd: str | os.PathLike[str] | None = None) -> ClaudeCodeSession:
        """Opens a new Claude Code session, in the directory it is given or in this one."""
        return ClaudeCodeSession(self, cwd)
