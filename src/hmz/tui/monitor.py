"""What a flow is doing, kept as it happens: who is working, who handed to whom, and cost.

None of this is asked of the agents. A flow drives them and they answer; what is watched here
is the turns going past, which is the only place the order of a flow is ever visible -- the
flow itself is a Python file that could branch any way it likes.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["Monitor", "Shape", "Spend", "Under", "short", "thousands"]

_WINDOW = 300.0

_THOUSAND = 1000
_MILLION = 1_000_000

def thousands(count: int) -> str:
    """Renders a token count short enough for a status line.

    Args:
      count: How many tokens.

    Returns:
      The count, abbreviated once it stops fitting.
    """
    if count < _THOUSAND:
        return str(count)
    if count < _MILLION:
        return f"{count / _THOUSAND:.1f}k"
    return f"{count / _MILLION:.2f}M"

def short(agent: str) -> str:
    """An agent's name, cut down to what fits beside a transcript.

    Args:
      agent: The agent id, which is a Chrysos Heir's codename unless it was named.

    Returns:
      Something recognisable and narrow.
    """
    kind, _, tail = agent.partition("#")
    if not tail:  
        return agent[:16]
    backend = kind.removesuffix("Agent").removesuffix("CLI").removesuffix("Code")
    return f"{backend.lower()}#{tail[:4]}"

@dataclass(frozen=True, slots=True)
class Under:
    """One agent a flow's own agent started of its own, which is what a subagent is.

    Not an agent of the flow. Nobody chose what it runs, nothing can be said to it and it has
    no transcript of its own -- it is a thing the agent above it is doing, and it is drawn as
    one. What is known of it is what its backend said on the way past.

    Attributes:
      whose: The backend's own id for it, which is what pairs the one that started with the
        one that ended.
      about: What it was asked to do, as its backend said it.
      working: Whether it is still going.
    """

    whose: str
    about: str
    working: bool = True

@dataclass(frozen=True, slots=True)
class Shape:
    """The directed graph of a run so far, taken whole so a reader never sees it mid-change.

    Which is the shape of the flow, and the only place it is ever visible: a flow is a Python
    file that may branch any way it likes, so what it did is read off the turns going past
    rather than asked of it.

    Attributes:
      turns: How many turns each agent that has worked has taken.
      working: Which of them have a turn open right now.
      handovers: How often each agent handed on to each other agent, as the flow went from
        one to the next.
      under: The agents each of them has started of its own, oldest first -- the ones still
        going and the ones that have finished, since a fleet that vanished as it landed would
        be a turn nobody could see the shape of afterwards.
    """

    turns: Mapping[str, int]
    working: frozenset[str]
    handovers: Mapping[tuple[str, str], int]
    under: Mapping[str, tuple[Under, ...]] = field(
        default_factory=dict[str, tuple[Under, ...]]
    )

@dataclass(frozen=True, slots=True)
class Spend:
    """What one model has cost so far, and how fast it is costing it.

    Attributes:
      model: The model the tokens were spent on.
      tokens: Every token spent on it, in and out alike.
      rate: Tokens a second over the last five minutes, or over the whole run while the run
        is younger than that. Seconds on the clock, not seconds an agent was talking: a flow
        sleeps between rounds, commits, reads what the last turn wrote, and that time is time
        the tokens were spent over.
    """

    model: str
    tokens: int
    rate: float

@dataclass
class Monitor:
    """The running state of one flow, written from the turns and read by the interface."""

    working: Counter[str] = field(default_factory=Counter[str])
    
    turns: Counter[str] = field(default_factory=Counter[str])
    
    handovers: Counter[tuple[str, str]] = field(
        default_factory=Counter[tuple[str, str]]
    )
    
    models: dict[str, str] = field(default_factory=dict[str, str])

    fleets: dict[str, dict[str, Under]] = field(
        default_factory=dict[str, dict[str, Under]]
    )
    
    spent: Counter[str] = field(default_factory=Counter[str])

    totals: dict[tuple[str, str], int] = field(
        default_factory=dict[tuple[str, str], int]
    )

    recent: deque[tuple[float, str, int]] = field(
        default_factory=deque[tuple[float, str, int]]
    )

    rates: dict[str, float] = field(default_factory=dict[str, float])
    figured: int | None = None
    
    changed: int = 0
    
    began: float = field(default_factory=time.monotonic)

    until: float | None = None
    
    _last: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def begins(self, agent: str, model: str) -> None:
        """Notes that an agent has started a turn.

        Args:
          agent: Whose turn it is.
          model: The model that agent runs at.
        """
        with self._lock:
            self.models[agent] = model
            self.working[agent] += 1
            self.turns[agent] += 1
            if self._last is not None and self._last != agent:
                self.handovers[self._last, agent] += 1

    def ends(self, agent: str) -> None:
        """Notes that an agent's turn is over, and that it is the one to hand on from.

        Args:
          agent: Whose turn ended.
        """
        with self._lock:
            if self.working[agent] <= 1:
                del self.working[agent]
            else:
                self.working[agent] -= 1
            self._last = agent

    def started(self, agent: str, whose: str, about: str) -> None:
        """Notes that an agent has started an agent of its own.

        Args:
          agent: Whose fleet it is.
          whose: The backend's own id for the one that started.
          about: What it was asked to do.
        """
        with self._lock:
            self.fleets.setdefault(agent, {})[whose] = Under(whose, about)

    def finished(self, agent: str, whose: str, about: str = "") -> None:
        """Notes that one of those has come back.

        Kept rather than forgotten: a fleet that vanished as it landed would be a turn nobody
        could see the shape of afterwards. One that ended without ever being seen to start --
        a backend that says only the one half, a turn watched from part way through -- is
        written down as having ended, since it did.

        Args:
          agent: Whose fleet it is.
          whose: The backend's own id for the one that ended.
          about: What it was asked to do, for one nothing saw start.
        """
        with self._lock:
            held = self.fleets.setdefault(agent, {})
            was = held.get(whose)
            held[whose] = Under(
                whose, was.about if was is not None else about, working=False
            )

    def stops(self) -> None:
        """Notes that the run is over, which is what stops the clock the rate is read at."""
        with self._lock:
            self.until = time.monotonic()
            self.figured = None  

    def spend(
        self,
        agent: str,
        tokens: int,
        model: str | None = None,
        now: float | None = None,
    ) -> None:
        """Notes tokens an agent's backend has just reported spending.

        Args:
          agent: Who spent them.
          tokens: How many, in and out together.
          model: What they were spent on, if the backend said. What it says beats what the
            agent was configured with: a turn that reached for a sub-agent spent it there.
          now: When, defaulting to this moment. Given only so a test can say.
        """
        if tokens <= 0:
            return
        if model is not None:
            self.models[agent] = model
        model = self.models.get(agent, agent)

        self.counted("told", model, self.totals.get(("told", model), 0) + tokens, now)

    def counted(
        self, source: str, model: str, total: int, now: float | None = None
    ) -> None:
        """Notes what one source has now seen spent on one model, all told.

        A total rather than an addition, because a log is read again and again: what it says
        the second time is what it said the first time and more, and adding that would count
        the first of it twice.

        Args:
          source: Who says so, which is either the backends or their logs.
          model: What the tokens were spent on.
          total: Every token that source has seen spent on it.
          now: When, defaulting to this moment. Given only so a test can say.
        """
        with self._lock:
            if total <= self.totals.get((source, model), 0):
                return
            self.totals[(source, model)] = total

            seen = max(
                held for (_, named), held in self.totals.items() if named == model
            )
            if (risen := seen - self.spent[model]) <= 0:
                return
            self.spent[model] = seen
            self.recent.append((time.monotonic() if now is None else now, model, risen))
            self.changed += 1

    def spending(self, now: float | None = None) -> list[Spend]:
        """What each model has cost, and how fast, biggest spender first.

        Args:
          now: The moment to measure the rate at, defaulting to this one.

        Returns:
          One entry per model anything has been spent on. How fast is worked out again when
          something it is made of has moved -- tokens counted, or old ones falling out of the
          window -- and stands the rest of the time, rather than being worked out on a clock.
        """
        moment = time.monotonic() if now is None else now
        if self.until is not None:
            moment = min(
                moment, self.until
            )  
        with self._lock:
            aged = False
            while self.recent and self.recent[0][0] < moment - _WINDOW:
                self.recent.popleft()
                aged = True
            if aged or self.figured != self.changed:

                over = min(_WINDOW, moment - self.began)
                lately: Counter[str] = Counter()
                for _, model, tokens in self.recent:
                    lately[model] += tokens
                self.rates = {
                    model: lately[model] / over if over > 0 else 0.0
                    for model in self.spent
                }
                self.figured = self.changed
            return [
                Spend(model=model, tokens=tokens, rate=self.rates.get(model, 0.0))
                for model, tokens in self.spent.most_common()
            ]

    def now_working(self) -> list[str]:
        """Who has a turn open, taken whole so that a reader never sees it mid-change.

        Returns:
          The agents working, in a settled order.
        """
        with self._lock:
            return sorted(self.working)

    def shape(self) -> Shape:
        """The run as a graph: who has worked, who is working, who handed to whom.

        Taken under the lock and copied out of it, so that whatever draws it is drawing one
        moment of the run rather than three moments of three counters.

        Returns:
          The graph, which is what the diagram on `/status` is drawn from.
        """
        with self._lock:
            return Shape(
                turns=dict(self.turns),
                working=frozenset(self.working),
                handovers=dict(self.handovers),
                under={
                    agent: tuple(held.values())
                    for agent, held in self.fleets.items()
                    if held
                },
            )
