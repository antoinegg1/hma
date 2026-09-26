"""The sheets: which flow, how it is set up, what each of its agents runs, and how it goes.

Drawn as Claude Code draws its own `/model`, which is the same question one step along: a rule
of `▔` across the top, the question and a line about it indented three, the choices numbered
with `❯` against the one under the cursor and a `✔` against the one already in force, and
under them the one setting that is adjusted rather than chosen -- the effort -- on a line the
left and right arrows move along. The keys are said at the bottom and nowhere else.

One agent is three steps, in this order and one agent at a time: which coding agent takes its
turns and which account it runs as (:class:`RunsAs`), which model it runs and at what effort
(:class:`Models`), and -- only where the flow said that one may be pointed at a machine --
where its work lands (:class:`Anchors`). The order is the order of what depends on what: an
account belongs to a backend and a model belongs to the CLI that runs it, so neither can be
asked before the CLI has been. The backends are read one at a time, a tab apiece: the ones
installed here plus an optional one the sheet can teach somebody to install. Every model of
every CLI in one list is a list that grows each time any of them ships a model. The effort is
the line with the arrows on it, exactly as Claude Code's is, and beside it the things that
really are side questions about the same agent.

`/status` is the last of them, and is read rather than answered -- Claude Code's own, which is
a rule across, fields down the left and their values lined up beside them.
"""

from __future__ import annotations

import contextlib
import shlex
import sys
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    NamedTuple,
    cast,
    get_args,
    get_origin,
)

from rich.markup import escape
from textual import events, on, work
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option

from hmz import telemetry
from hmz.agents import ANYONE, FLOW, PERMISSIONS, SWARM, USER, anchored, driver
from hmz.agents.skills import Skill, skills
from hmz.backends import named
from hmz.kept import Kept, Runs
from hmz.telemetry import KEPT, SAYS, SENT

from .discover import installed, machines, ready_to_open
from .monitor import Shape, short, thousands
from .selecting import Choices

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence

    from pydantic import BaseModel
    from pydantic.fields import FieldInfo
    from textual.app import App, ComposeResult

    from hmz.agents import AgentBase, Board, Moment
    from hmz.backends import Model, Way
    from hmz.epic import Ran

    from hmz.fallbacks import Falls as Step
    from hmz.flows import Flowverse, Offer, Place
    from hmz.providers import Provider
    from hmz.sdk import Hmz

    from .monitor import Monitor, Under

__all__ = [
    "DETACHES",
    "EVERY",
    "STAYS",
    "STOPS",
    "Account",
    "Accounts",
    "Agent",
    "Alike",
    "Anchors",
    "Backends",
    "Catalogue",
    "Chosen",
    "Clis",
    "Configures",
    "Confirms",
    "Does",
    "Doing",
    "Drafts",
    "Drawn",
    "Entry",
    "Epics",
    "Failing",
    "Fallbacks",
    "Fitted",
    "Flows",
    "Flowverses",
    "Held",
    "Holds",
    "Imports",
    "Leaves",
    "Names",
    "Picks",
    "Popup",
    "Providers",
    "Saved",
    "Sheet",
    "Signing",
    "Signs",
    "Skills",
    "Speaks",
    "Status",
    "Ways",
    "called",
    "carries_on",
    "config_of",
    "model_of",
    "opens_on",
    "places_of",
    "pointed",
    "reads",
    "setting",
    "settled",
]

def called(places: tuple[Place, ...], at: int) -> str:
    """What to call the agent being configured, which every step of configuring it says.

    In one place because it is said in three, and an agent that read as two different things
    between one step and the next would be two.

    Args:
      places: One place per agent the flow drives, in the order it takes them.
      at: Which of them is being asked about, counting from zero.

    Returns:
      The name the flow calls it, or where it comes among them for a flow that named none.
    """
    return places[at].name or f"agent {at + 1} of {len(places)}"

def pointed(place: Place) -> bool:
    """Whether where one agent works is a question anybody is asked about it.

    Only for a place the flow declared `Remote`: a flow that says so is a flow that expects
    to be told where that agent works, and one that says nothing has said its agent works
    here. A container the flow named is not asked about either -- the flow settled it, and
    nobody else has any say in it.

    Args:
      place: What the flow declared.

    Returns:
      True if there is a machine to be chosen for it, which is a step of its own.
    """
    from hmz.agents import Remote

    return place.where is Remote or isinstance(place.where, Remote)

def _settled(place: Place) -> str:
    """The container a flow put one of its agents in, where it named one.

    Args:
      place: What the flow declared.

    Returns:
      The image, or "" for an agent that works here and one that is asked where it works --
      neither of which is something the flow settled.
    """
    from hmz.agents import Isolated

    return place.where.image if isinstance(place.where, Isolated) else ""

_RULE = "▔"
_INDENT = "   "

_DOT = " · "

_HERE = "❯"
_INFORCE = "✔"

_TICKED = "[✔]"
_EMPTY = "[ ]"

_LABEL = 26

_FIELD = 18

_LIVE = 0.5

class Held(NamedTuple):
    """What one agent of a running flow is holding, and whether it is the one being read.

    Attributes:
      many: How many conversations it has open, which is none for an agent that has opened
        none and for every agent of a flow that is not running.
      reading: Whether this agent's transcript is the one on the screen. All of its
        conversations are that one transcript, so this is a yes or a no rather than which of
        them: an agent is what is stepped onto, and a loop that opens one conversation a
        turn runs them all down the same screen.
      unread: Whether it has said something since it was last looked at.
      working: Whether any of its conversations has a turn open. Which is the first thing
        somebody looks for with several agents going at once -- who is thinking and who has
        stopped -- and the only one of these that changes by itself.
    """

    many: int = 0
    reading: bool = False
    unread: bool = False
    working: bool = False

_WORKING, _IDLE = "●", "○"

def _holds(held: Held) -> str:
    """What one agent's conversations say about themselves beside what it runs.

    Args:
      held: What it is holding.

    Returns:
      Whether it is working, how many conversations it has, `reading` for the one whose
      transcript is on the screen and `unread` for one that has said something since it was
      last looked at. Nothing at all for an agent holding none, which is every agent of a
      flow that is not running.
    """
    if not held.many:
        return ""
    said = f"{_WORKING if held.working else _IDLE} {held.many}"
    if held.reading:
        return f"{said}{_DOT}reading"
    return f"{said}{_DOT}unread" if held.unread else said

def reads(
    named: tuple[str, ...], runs: list[Runs], holding: Sequence[Held] = ()
) -> list[str]:
    """One line per agent a flow drives: what it runs, where, and what it is holding.

    In one place because it is read in two -- above the prompt while a flow runs, and on
    `/status` -- and an agent that read as two different things in them would be two. What it
    is holding is only asked for above the prompt, that being where a conversation is read
    and said to; `/status` asks for the same line without it, and it says nothing there.

    Args:
      named: What the flow calls each of them, "" apiece where it names none.
      runs: What each of them runs, and where its turns land.
      holding: The conversations each of them has open, in the same order, or nothing at all
        for a flow that is not running -- which holds none.

    Returns:
      One line apiece, in the order the flow takes them.
    """
    return [
        _DOT.join(
            escape(part)
            for part in (
                named[at] if at < len(named) else "",
                one.spec,
                one.anchor,

                one.permission,
                one.provider,
                
                "" if one.web_search else "no web search",
                _holds(holding[at]) if at < len(holding) else "",
            )
            if part
        )
        for at, one in enumerate(runs)
    ]

_SHEET = """
Anchors, Backends, Configures, Flows, Models, Providers, RunsAs, Signing, Skills, Status, Ways {
    align: center middle; background: $background; }
#sheet { width: 100%; height: auto; padding: 0; }
#rule { height: 1; color: $primary; }
#asked { padding: 0 0 0 3; text-style: bold; color: $primary; }
#about { padding: 0 3 1 3; color: $text-muted; width: 1fr; }
/* The tabs, for the one sheet that has any. A sheet with none says nothing here, and a
   label with nothing in it is a row nobody paid for. */
#tabs { padding: 0 0 1 3; width: 1fr; }
OptionList { border: none; background: $background; scrollbar-size: 0 0; padding: 0; }
/* The marker says where the cursor is, so the row is not filled as well. */
#choices > .option-list--option-highlighted {
    background: $background; color: $foreground; text-style: none; }
/* As wide as the sheet, so that what is said under the list and the keys under that wrap
   onto a second row rather than running off the side of a narrow terminal: a key nobody can
   see is a key nobody has. */
#tuning { padding: 1 0 1 3; width: 1fr; }
#keys { padding: 0 0 0 3; color: $text-muted; width: 1fr; }
/* The fields carry their own indent, as the numbered rows above them do. */
#said { padding: 0 0 1 0; }
"""

_POPUP = """
Confirms { align: center middle; background: transparent; }
#sheet { width: 66; max-width: 100%; height: auto; padding: 1 2; border: round $primary;
         background: $background; }
#rule { display: none; }
#tuning { display: none; }
#asked { padding: 0; text-style: bold; color: $primary; }
#about { padding: 0 0 1 0; color: $text-muted; width: 1fr; }
OptionList { border: none; background: $background; scrollbar-size: 0 0; padding: 0; }
#choices > .option-list--option-highlighted {
    background: $background; color: $foreground; text-style: none; }
#keys { padding: 1 0 0 0; color: $text-muted; width: 1fr; }
"""

_TURNS = "tab/shift+tab to switch"

_STEPS = "←/→ to switch"

_MOST = 14

_LEAST = 3

class Body(Vertical):
    """What a sheet is drawn down, which says when it has grown taller than the terminal.

    A sheet is a question with its keys under it, and the one part of it that can be any
    length is the list in the middle: every flow there is, every model a CLI runs. Drawn as
    tall as it likes, that list pushes the keys off the bottom of a short terminal -- so the
    column says when its height changes and the list is shortened to fit. Resize does not
    bubble, so nothing else would hear about it.
    """

    def on_resize(self) -> None:
        """Tells whoever is holding this column that it is a different height now."""
        sheet = self.screen
        if isinstance(sheet, Sheet):
            sheet.shortens()

class Sheet[T](ModalScreen[T | None]):
    """One question drawn the way Claude Code draws one, answered by picking a line.

    What answering it comes to is the sheet's own: a flow is a name, an agent is what it runs
    and where, and walking out without answering is None wherever it is asked.

    A sheet of several pages says so: the titles are across the top and tab and shift+tab turn
    between them, which is the one pair of keys a terminal has for exactly that. Nothing here
    is a chord -- a sheet asks one thing and its keys are its own, so a key that needed ctrl
    held down would be a key somebody had to already know.
    """

    CSS = _SHEET
    BINDINGS: ClassVar = [("escape", "back", "back")]

    TABS: ClassVar[tuple[str, ...]] = ()

    _drawn: int | None = None
    
    _counting = 1

    _typed: str = ""

    _searching = False
    
    _tab = 0

    _room: int | None = None
    
    _arming = ""

    LETTERS: ClassVar[frozenset[str]] = frozenset()

    def turnable(self) -> tuple[bool, ...]:
        """Which pages may be opened now, which is not always all of them.

        Returns:
          One per tab, in the order they go. All of them unless a sheet says otherwise -- a
          page that cannot be opened is one the tabs step over and one the titles say is
          shut, rather than one that is not there at all.
        """
        return tuple(True for _ in self.TABS)

    def action_next_tab(self) -> None:
        """Opens the next page there is to open."""
        self._turn_page(1)

    def action_prev_tab(self) -> None:
        """Opens the one before it."""
        self._turn_page(-1)

    def _turn_page(self, by: int) -> None:
        """Turns to the next page that may be opened, wrapping round at either end.

        Nothing is applied on the way: a menu is answered once, when it is left, so turning a
        page is reading rather than choosing.

        Args:
          by: One page forward or back.
        """
        able = self.turnable()
        if sum(able) < 2:  
            return
        at = self._tab
        for _ in range(len(self.TABS)):
            at = (at + by) % len(self.TABS)
            if able[at]:
                break
        if at == self._tab:
            return
        self._tab = at

        self._typed, self._searching = "", False
        self.query_one("#choices", OptionList).highlighted = 0
        self._drawn = 0
        self._turned()
        self._fill()

    def _turned(self) -> None:
        """What a sheet does as a page opens, which is nothing unless it says otherwise."""

    def _tab_line(self) -> str:
        """The titles, with the one being read marked and the shut ones struck through."""
        if not self.TABS:
            return ""
        able = self.turnable()
        said = _DOT.join(
            f"[b $primary]{escape(one)}[/]"
            if at == self._tab
            else f"[$text-muted]{escape(one)}[/]"
            if able[at]
            else f"[$text-muted][s]{escape(one)}[/s][/]"
            for at, one in enumerate(self.TABS)
        )
        if sum(able) > 1:
            said += f"   [$text-muted]{_TURNS}[/]"
        return said

    def action_search(self) -> None:
        """Starts narrowing the list by what is typed, until esc says to stop."""
        self._searching = True
        self.query_one("#choices", OptionList).highlighted = 0
        self._drawn = 0
        self._fill()

    def fits(self, *fields: str) -> bool:
        """Whether a row is one of the ones still worth showing.

        Args:
          fields: Everything the row says, which is all of it that is searched: what a thing
            is called, and where it came from.

        Returns:
          True if what has been typed is spread through one of them in order, so that a few
          letters anywhere in a name find it -- nobody types a model id out to narrow a list
          of them. One of them rather than all of them run together, or a search would run
          off the end of the name it was narrowing to and finish itself in the word beside
          it: `chat` would find `hma builtin`, which is a match nobody typed.
        """
        if not self._typed:
            return True
        wanted = self._typed.lower()
        for field in fields:
            looking, at = field.lower(), 0
            for letter in wanted:
                at = looking.find(letter, at) + 1
                if not at:
                    break
            else:
                return True
        return False

    def searching(self) -> str:
        """What to say about the search, which is nothing at all until it has been asked for.

        Returns:
          The line to put after the keys: which key starts a search where none is running,
          and what has been typed so far where one is -- with the block the next letter lands
          on, so that a search nothing has been typed into yet still looks like one.
        """
        if not self._searching:
            return f"{_DOT}s to search"
        return (
            f"{_DOT}search [$secondary]{escape(self._typed)}[/][reverse] [/reverse]"
            f"{_DOT}Esc to leave it"
        )

    def on_key(self, event: events.Key) -> None:
        """Takes a letter as narrowing the list, once a search has been asked for.

        Only then: every other key on these sheets is a letter of its own, and a list where
        typing always searched would be a list with no keys left. The arrows walk it and enter
        takes what is under the cursor, either way.

        Args:
          event: The key.
        """
        if not self._searching:
            return
        if event.key == "backspace":
            self._typed = self._typed[:-1]
        elif event.is_printable and event.character:
            self._typed += event.character
        else:
            return
        event.prevent_default()
        event.stop()
        self.query_one("#choices", OptionList).highlighted = 0
        self._drawn = 0
        self._fill()

    def compose(self) -> ComposeResult:
        """The rule, the question, the tabs, what there is to choose, what is tuned, the keys.

        Every sheet is made of the same parts whether or not it uses them. The tabs are the
        one part that is taken away again where a sheet has none -- see :meth:`tabbed` -- so
        that a sheet which is one list is drawn as one list and nothing moved down a row.
        """
        with Body(id="sheet"):
            yield Label(id="rule")
            yield Label(id="asked")
            yield Label(id="about")
            yield Label(id="tabs")
            yield Choices(id="choices")
            yield Label(id="tuning")
            yield Label(id="keys")

    def on_mount(self) -> None:
        """Rules the top of the sheet across, and asks."""
        self.query_one("#choices", OptionList).styles.max_height = _MOST
        self.query_one("#rule", Label).update(_RULE * self.size.width)

        self.tabbed(self._tab_line())
        self._ask()

    def tabbed(self, said: str) -> None:
        """Puts a row of tabs above the choices, or takes the row back where there are none.

        Args:
          said: The tabs, as markup, or "" for a sheet that is one list.
        """
        showing = self.query_one("#tabs", Label)
        showing.display = bool(said)
        showing.update(said)

    def on_resize(self) -> None:
        """Rules the new width across, and shortens the list to the room left under it."""
        if not self.query("#sheet"):
            return  
        self.query_one("#rule", Label).update(_RULE * self.size.width)
        self.shortens()

    def shortens(self) -> None:
        """Shortens the list until what is under it is inside the terminal.

        The list is what gives. Everything else on a sheet is a line or two -- what is being
        asked, what it comes to, the keys -- and the rows are what there are a hundred of, so
        a sheet that does not fit is a sheet whose list is too long for the terminal it is
        drawn in rather than a sheet with too much on it. The keys are the last row, so they
        are what falls off the bottom, and a key nobody can see is a key nobody has.

        Called each time the column changes height, which is each time the list is put up
        again, and each time the terminal changes size. It settles at once: how tall the rest
        of the sheet is does not depend on how many rows the list is showing.
        """
        listing = self.query_one("#choices", OptionList)
        column = self.query_one("#sheet", Body).outer_size.height
        rest = column - listing.outer_size.height
        room = max(_LEAST, min(_MOST, self.size.height - rest))
        if room == self._room:
            return
        self._room = room
        listing.styles.max_height = room

    def action_back(self) -> None:
        """Comes out of the search, or leaves once there is no search to come out of.

        A search is the one place esc has something to step back to: leaving from there would
        throw away the walk in as well as the wrong letters.
        """
        if self._searching:
            self._searching, self._typed = False, ""
            self._drawn = 0
            self._fill()
            return
        self.leaving()

    def leaving(self) -> None:
        """What esc comes to once there is no search to leave, which is walking out.

        A sheet holding changes that have not been applied says something else here -- see
        :class:`Drafts` -- because walking out of one of those is a decision rather than a
        step back.
        """
        self.dismiss(None)

    def _row(
        self,
        at: int,
        label: str,
        about: str,
        *,
        here: bool,
        inforce: bool,
        box: str = "",
    ) -> str:
        """One numbered choice, laid out as Claude Code lays one out.

        Args:
          at: Which one it is, counting from zero.
          label: What it is called.
          about: The line about it, which is said quietly.
          here: Whether the cursor is on it.
          inforce: Whether it is the one already in force.
          box: The switch in front of the name, for a list whose rows are switched on and off
            rather than picked between, or "" for a list that is picked from.

        Returns:
          The row, as markup.
        """
        mark = f"{_INDENT}[$primary]{_HERE}[/] " if here else f"{_INDENT}  "
        
        number = f"{at + 1:>{self._counting}}."
        
        switch = f"[$success]{escape(box)}[/] " if box else ""
        named = escape(label) + (f" [$success]{_INFORCE}[/]" if inforce else "")
        
        pad = " " * max(
            1,
            _LABEL - len(label) - (2 if inforce else 0) - (len(box) + 1 if box else 0),
        )
        return (
            f"{mark}[$text-muted]{number}[/] {switch}{named}{pad}"
            f"[$text-muted]{escape(about)}[/]"
        )

    @on(OptionList.OptionHighlighted)
    def _moved(self, event: OptionList.OptionHighlighted) -> None:
        """Redraws, so the marker sits beside the row the cursor moved to.

        Only when it has moved somewhere the marker is not already: putting the rows up sets
        the cursor, which posts one of these, and redrawing on that would be one keypress and
        renders without end -- which is what a list that lags is.

        Args:
          event: Where the cursor is now.
        """
        if event.option_index == self._drawn:
            return
        self._drawn = event.option_index

        self._arming = ""
        self._fill()

    def check_action(
        self,
        action: str,
        parameters: tuple[object, ...],  
    ) -> bool | None:
        """Whether one of this sheet's own keys is live, which a search turns most of them off.

        A key that is a letter is the sheet's only while nothing is being typed: the whole
        point of asking for a search is that the letters go into it. Everything else -- esc,
        the arrows, enter, the tabs -- means what it means either way.

        Args:
          action: What the key would do.
          parameters: What it would do it with.

        Returns:
          Whether to run it. A binding refused here is one the key falls through, so the
          letter reaches the search rather than being swallowed.
        """
        return not (self._searching and action in self.LETTERS)

    def under(self) -> str:
        """What the cursor is on, by the id the row was put up under.

        Returns:
          The id, less the `=` a row whose answer may be the empty string carries, or "" for
          a list with nothing in it and for a cursor sitting on a heading.
        """
        listing = self.query_one("#choices", OptionList)
        at = listing.highlighted
        if at is None or not 0 <= at < listing.option_count:
            return ""
        return str(listing.get_option_at_index(at).id or "").removeprefix("=")

    def _armed(self, what: str) -> bool:
        """Whether a key that has to be pressed twice has been pressed once already.

        Taking something away is the one thing on these sheets that cannot be undone, so it is
        asked for twice: the first press arms the row under the cursor and says so, and the
        second takes it away. Moving the cursor puts it down again -- see :meth:`_moved` --
        which is what makes a stray keypress harmless.

        Args:
          what: The row, by its id.

        Returns:
          True if this is the second press and the thing is to go.
        """
        if self._arming == what:
            self._arming = ""
            return True
        self._arming = what
        return False

    def _fill(self) -> None:
        """Puts the choices up, which each sheet says for itself."""
        raise NotImplementedError

    def _ask(self) -> None:
        """Draws whatever is being asked for now, which each sheet says for itself."""
        raise NotImplementedError

_KEEP, _DROP = "keep", "drop"

class Drafts[T](Sheet[T]):
    """A sheet that holds everything changed in it until it is asked to apply the lot.

    Which is what makes several pages one menu: turning a page applies nothing, so what is
    read on the second page is what the first page is holding rather than what is written
    down. Nothing lands until the menu is left and saving is confirmed -- and esc on a menu
    holding changes asks, because walking out of one is a decision rather than a step back.
    """

    _changed = False

    def changed(self) -> None:
        """Says that something has been changed, so that esc asks before throwing it away."""
        self._changed = True

    def applied(self) -> None:
        """Answers with everything held, which each menu says for itself."""
        raise NotImplementedError

    def leaving(self) -> None:
        """Asks whether to save what is held, and does whichever was asked for.

        Nothing at all where nothing was changed: a walk in to look and out again is not a
        question anybody wants asked of them.
        """
        if not self._changed:
            self.dismiss(None)
            return
        self.asks_to_save()

    @work
    async def asks_to_save(self) -> None:
        """Puts the question up, and does what it is answered with."""
        showing = cast(
            "App[None]",
            self.app,  
        )
        said = await showing.push_screen_wait(Confirms())
        if said == _KEEP:
            self.applied()
        elif said == _DROP:

            telemetry.snag("changes-dropped", sheet=type(self).__name__)
            self.dismiss(None)

class Chosen(NamedTuple):
    """What the flow menu was answered with: what to run, on what, and set up how.

    One answer rather than three, because the menu is one thing answered once: what is held
    on each of its pages lands together when it is saved, or none of it does.

    Attributes:
      flow: The flow to run, by the name it was offered under.
      agents: What each of its agents is, in the order the flow takes them.
      config: What the flow itself is set up with, or None for a flow that takes no setting
        up and one that was left as it comes.
    """

    flow: str
    agents: tuple[Runs, ...]
    config: BaseModel | None = None

def opens_on(
    agents: Mapping[str, tuple[Model, ...]], *, goals: bool = True
) -> list[Runs]:
    """The one agent to fall back on where nothing has been remembered for a place.

    The first backend installed here that has said what it runs and can be opened without
    further setup, at the first model it named -- which is that CLI's own idea of what it runs
    by default, and the only idea of it worth having. Nothing is written down here: a model
    named in this file would be a model this file was right about on the day it was written.

    Args:
      agents: The backends there are, and what each of them says it runs.
      goals: Whether backend goals start available to it.

    Returns:
      The one agent, or nothing at all where no backend here has both said what it runs and
      can be opened without further setup.
    """
    where = Path.cwd()
    for backend, found in agents.items():
        if found and ready_to_open(backend, where):

            one = found[0]

            effort = "high" if "high" in one.efforts else ""
            if not effort and one.efforts:
                effort = one.efforts[-1]
            return [Runs(f"{backend}/{one.name}:{effort}", goals=goals)]
    return []

def places_of(flow: str) -> tuple[Place, ...] | None:
    """The agents a flow drives, or None for a flow that will not load.

    Args:
      flow: The flow, by the name it was offered under -- not by the file that name resolves
        to, since a file may hold several and which of them was asked for is the half after
        the colon.

    Returns:
      One place per agent it drives, and None where reading the flow raised at all -- which
      is a flow to report rather than a reason for a menu not to draw.
    """
    try:
        return _hmz().flows.places(flow)
    except Exception:  
        return None

def model_of(flow: str) -> type[BaseModel] | None:
    """What a flow says it can be set up with, if it says anything.

    Args:
      flow: The flow, by name or as a path.

    Returns:
      The model to ask with, or None for a flow that takes no setting up -- and for one that
      will not load, which is a flow to report where it is run rather than here.
    """
    try:
        return _hmz().flows.configures(flow)
    except Exception:  
        return None

def config_of(flow: str, kept: dict[str, Any]) -> BaseModel | None:
    """How a flow was last set up, read back through the flow's own model rather than trusted.

    Args:
      flow: The flow.
      kept: What was written down for it, field by field.

    Returns:
      What it was set up with, or None for a flow that takes no setting up, has not been set
      up here, or has since changed enough that what was kept no longer reads -- a settings
      file is a convenience, and one that no longer fits is one to start over from.
    """
    model = model_of(flow)
    if model is None or not kept:
        return None
    try:
        return model.model_validate(kept)
    except Exception:  
        return None

def settled(
    runs: Sequence[Runs],
    places: Sequence[Place],
    agents: Mapping[str, tuple[Model, ...]] | None = None,
) -> list[Runs]:
    """One agent per place a flow drives, out of however many were remembered for it.

    A flow that has grown an agent since it was last run here is a flow with a place nothing
    was remembered for, and one that has lost one is a flow with an agent nobody will drive.
    Neither is a reason to start over: what is there is kept, and what is missing falls back
    on the agent the interface opens talking to.

    Args:
      runs: What was remembered, in the order the flow took them then.
      places: What the flow drives now.
      agents: The backends there are, for the place nothing was remembered for, or None
        where there is nothing to fall back on -- which leaves such a place unanswered.

    Returns:
      One apiece, with goals forced on for a place the flow declared it needs them at -- that
      one is the flow's own requirement rather than anybody's choice.
    """
    spare = opens_on(agents) if agents is not None else []
    held: list[Runs] = []
    for at, place in enumerate(places):
        if at < len(runs):
            one = runs[at]
        elif spare:

            one = spare[0]._replace(goals=place.goals_default)
        else:

            break
        held.append(one._replace(goals=True) if place.goal else one)
    return held

def _complete(runs: Runs) -> bool:
    """Whether one agent has been answered at all, which is a CLI and a model of that CLI.

    Args:
      runs: The agent.

    Returns:
      True if there is something to run it on.
    """
    cli, _, rest = runs.spec.partition("/")
    model, _, _ = rest.rpartition(":")
    return bool(cli and model)

_HALVES = "\x1f"

_FLOW_PAGE, _AGENT_PAGE = 0, 1
_SAVE = "save"

class Flows(Drafts[Chosen]):
    """Which flow runs and what each of its agents is: one menu, a page apiece.

    Two pages because they are two questions about one thing, and because they are not open
    at the same moments. A flow is chosen in order to be started, so choosing one while one is
    running is not a thing to offer at all -- that page is shut while a flow runs, and says so
    rather than going away. What its agents are is the other way round: an agent that is
    thinking too little, on the wrong account, or allowed too much is found out halfway
    through a run, so that page is never shut.

    The flows are read a place at a time -- every flowverse there is, fetched or not, and then
    this project's flows and yours -- with the left and right arrows stepping between the
    places and the list holding only the one being read. All of them run together under
    headings was one list nobody could see the end of, and one where walking to a flow meant
    walking past every flow that came before it. Stepping between the places is about which
    list of flows is being read; what can happen to a flowverse is `/flowverses`, which is a
    question about the places rather than about which flow to run.

    Choosing a flow asks what that flow itself takes, where it takes anything, and then turns
    to what will drive it. A key that set the flow up was a key nobody pressed: a flow with
    settings is chosen in order to be run with settings, and the moment it is chosen is the
    one moment somebody is thinking about that flow rather than about its agents.

    Nothing is applied by turning a page. What the menu holds is a draft of the whole of it,
    and it lands together from the save row or when saving is confirmed on the way out.
    """

    TABS: ClassVar = ("Flow", "Agents")
    LETTERS: ClassVar = frozenset({"search", "fork"})

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),

        Binding("tab", "next_tab", "next page", priority=True),
        Binding("shift+tab", "prev_tab", "previous page", priority=True),

        Binding("left", "before", "the place before", priority=True),
        Binding("right", "after", "the place after", priority=True),

        Binding("s", "search", "search", priority=True),
        Binding("f", "fork", "copy it here to change", priority=True),
    ]

    def __init__(
        self,
        flow: str,
        runs: Sequence[Runs],
        config: BaseModel | None,
        agents: dict[str, tuple[Model, ...]],
        kept: dict[str, Any],
        *,
        unavailable: frozenset[str] = frozenset(),
        running: bool = False,
        opening: int = 0,
    ) -> None:
        """Initializes the menu on what is set up now.

        Args:
          flow: The flow running now, or the one this workspace is set up to run.
          runs: What each of its agents is, in the order the flow takes them.
          config: What the flow itself is set up with, for one that takes setting up.
          agents: The backends offered here, and what each of them says it runs.
          kept: What each flow was last set up with here, by flow -- read when the draft flow
            changes, so that turning to a flow this workspace has run finds it as it was left.
          unavailable: The optional backends among them that still need installing.
          running: Whether a flow is running, which is what shuts the first page.
          opening: Which page to open on, for whoever opened the menu to reach one of them
            directly. A page that is shut is not opened on whatever is asked for.
        """
        super().__init__()
        self._agents = dict(agents)
        self._unavailable = unavailable
        self._underway = running
        self._kept = kept

        self._flow: str = flow
        self._places: tuple[Place, ...] = places_of(flow) or ()
        if runs:
            self._runs = (
                self._fitted(settled(runs, self._places, self._agents))
                if self._places
                else list(runs)
            )
            self._config = config
        else:

            self._runs = self._fitted(
                settled(self._remembered(flow), self._places, self._agents)
            )
            self._config = config_of(flow, self._held(flow).get("config") or {})

        self._offers: list[Offer] | None = None

        self._was = ""

        self._where = ""
        
        self._said = ""

        self._fetching = ""
        
        self._tab = _AGENT_PAGE if running else opening % len(self.TABS)

    def turnable(self) -> tuple[bool, ...]:
        """Which pages may be opened: the agents always, and the flows while none runs."""
        return (not self._underway, True)

    def _follows(self, listing: OptionList) -> None:
        """Takes which row the cursor is on off the list, rather than off a row number.

        Read here rather than kept as the cursor moves, so that the two cannot disagree: the
        list is a different list under each place, and the row under the cursor is the only
        thing that says which flow is meant. Kept as the whole id -- where it came from and
        which flow it is -- so that a row remembered under one place cannot be taken for a
        row of the next.

        Args:
          listing: The list.
        """
        at = listing.highlighted
        if at is None or not 0 <= at < listing.option_count:
            return
        named = str(listing.get_option_at_index(at).id or "")
        if named:
            self._was = named

    def _fitted(self, runs: Sequence[Runs]) -> list[Runs]:
        """One row per agent the flow drives, whatever there was to fill it with.

        A place nothing was remembered for and nothing falls back on still has a row here:
        this is the page it is set up on, and a place with no row is a place nobody can
        answer. What such a row says is that it has not been answered yet.

        Args:
          runs: What there is, in the order the flow takes them.

        Returns:
          One apiece, padded with an agent that names nothing.
        """
        return [
            runs[at] if at < len(runs) else Runs("") for at in range(len(self._places))
        ]

    def _held(self, name: str) -> dict[str, Any]:
        """What one flow was last set up with here, which is nothing for one never run."""
        held = self._kept.get(name)
        return cast("dict[str, Any]", held) if isinstance(held, dict) else {}

    def _remembered(self, name: str) -> list[Runs]:
        """What one flow's agents were last set up as here, in the order it takes them.

        Args:
          name: The flow.

        Returns:
          One apiece, and nothing at all for a flow this workspace has never run -- which is
          a flow whose agents fall back on the one the interface opens talking to.
        """
        from hmz.kept import read_back

        agents: dict[str, Any] = self._held(name).get("agents") or {}
        return [
            runs
            for runs in (
                read_back(cast("dict[str, Any]", one))
                for one in agents.values()
                if isinstance(one, dict)
            )
            if runs is not None
        ]

    def _ask(self) -> None:
        """Says what the menu is, puts up the page it opened on, and catches up on fetches."""
        self.query_one("#asked", Label).update("Flow")
        self._fill()
        self.query_one("#choices", OptionList).focus()
        self._catches_up()

    @work
    async def _catches_up(self) -> None:
        """Fetches whatever has never been fetched, as the menu opens.

        A flowverse that is here and has never been fetched is a list with nothing in it and
        a key to press about it, which is a step nobody would choose to take: it is here
        because its flows are wanted. humanize's own repository of the rest is the one this
        is ever true of -- one that was added was cloned as it was added -- and it is the one
        every flow that is not in the package is in.

        Off the loop and out of the way: the menu is drawn first and stays drawn, what is
        being read is left where it is, and a fetch that fails says so under the list. Once
        per opening, however it goes, so that a machine with no network says so once rather
        than hammering a server on every keystroke.
        """
        verses = _hmz().verses

        for one in verses.all():
            if not one.url or one.fetched:
                continue
            name = one.name

            def fetching(named: str = name) -> str:
                verses.fetch(named)
                return named

            await self._fetches(name, fetching)

    def _turned(self) -> None:
        """Puts the cursor back on the flow being read when the flows page opens again."""
        self._said = ""

    def _fill(self) -> None:
        """Puts up whichever page is open, and the titles above it."""
        self.query_one("#about", Label).update(
            "Which flow the agents are driven through. The first thing you say once it is "
            "chosen is what it is to do. A flow anywhere else is a path you type."
            if self._tab == _FLOW_PAGE
            else f"What each agent {escape(self._flow)} drives is: the CLI that takes its "
            "turns, the account they run as, the model at an effort, and what it may do. "
            "Enter opens one, and save applies the complete flow setup."
        )
        if self._tab != _FLOW_PAGE:
            self.tabbed(self._tab_line())
            self._agents_page()
            return

        wheres = self._stepping()
        if self._where not in wheres:
            self._where = self._opens(wheres)
        self.tabbed(f"{self._tab_line()}\n{self._where_line(wheres)}")
        self._flows_page()

    def _all(self) -> list[Offer]:
        """Every flow there is, read once."""
        if self._offers is None:
            self._offers = _hmz().flows.all()
        return self._offers

    def _wheres(self) -> list[str]:
        """The places flows come from, in the order the arrows step through them.

        Returns:
          Every flowverse there is, fetched or not, except an empty one of your own. A
          flowverse is one of them whether or not it has been downloaded -- fetching it is
          what having it here is for -- but your own directories are not places to fetch
          anything into, so an empty one is nothing to step to.
        """
        from hmz.flows import MINE

        return [
            one.name
            for one in _hmz().verses.all()
            if one.name not in MINE
            or any(offer.whose == one.name for offer in self._all())
        ]

    def _stepping(self) -> list[str]:
        """The places there are to step between, which a search narrows to the ones it found.

        Returns:
          Every place while nothing is typed. While something is, only the places holding a
          flow that matches it -- a search is for finding a flow whose flowverse is the thing
          nobody remembers, so it MUST NOT leave somebody stepping through empty lists to
          reach the one row it found. All of them again where it found nothing anywhere,
          there being no narrower list to offer than the one that is already empty.
        """
        wheres = self._wheres()
        if not self._typed:
            return wheres
        found = [
            whose
            for whose in wheres
            if any(one.whose == whose and self.fits(one.name) for one in self._all())
        ]
        return found or wheres

    def _opens(self, wheres: list[str]) -> str:
        """Which place is read when the page is drawn without one already being read.

        Args:
          wheres: The places there are to step between.

        Returns:
          The one the flow in force came from, that being the flow this page is about, and
          otherwise the first there is.
        """
        return next(
            (
                one.whose
                for one in self._all()
                if one.name == self._flow and one.whose in wheres
            ),
            wheres[0] if wheres else "",
        )

    def _where_line(self, wheres: list[str]) -> str:
        """The places flows come from, with the one being read marked and the keys said.

        Args:
          wheres: The places, in the order the arrows step through them.

        Returns:
          The strip, as markup. Every place, so that the one being read is read as one of
          however many there are: a flowverse nobody can see is a flowverse nobody steps to.
        """
        said = _DOT.join(
            f"[b $primary]{escape(one)}[/]"
            if one == self._where
            else f"[$text-muted]{escape(one)}[/]"
            for one in wheres
        )
        if len(wheres) > 1:
            said += f"   [$text-muted]{_STEPS}[/]"
        return said

    def _verse(self, named: str) -> Flowverse | None:
        """The flowverse of that name, or None for a name none of them answers to."""
        return _hmz().verses.find(named)

    def action_before(self) -> None:
        """Reads the place before this one."""
        self._steps(-1)

    def action_after(self) -> None:
        """Reads the one after it."""
        self._steps(1)

    def _steps(self, by: int) -> None:
        """Turns to another of the places flows come from, wrapping round at either end.

        Args:
          by: One place on or back.
        """
        if self._tab != _FLOW_PAGE:
            return  
        wheres = self._stepping()
        if len(wheres) < 2:  
            return
        at = wheres.index(self._where) if self._where in wheres else 0
        self._where = wheres[(at + by) % len(wheres)]

        self._was, self._arming, self._said = "", "", ""
        self._fill()

    def _flows_page(self) -> None:
        """Puts up the flows of the place being read, and nothing from any other place."""
        listing = self.query_one("#choices", OptionList)
        self._follows(listing)
        mine = [
            one
            for one in self._all()
            if one.whose == self._where and self.fits(one.name)
        ]
        self._counting = len(str(max(len(mine), 1)))
        held = [f"{self._where}{_HALVES}{one.name}" for one in mine]
        if not held and not self._typed:

            held = [f"{self._where}{_HALVES}"]
        if self._was not in held:

            self._was = next(
                (one for one in held if one.partition(_HALVES)[2] == self._flow),
                held[0] if held else "",
            )
        rows = [
            Option(
                self._row(
                    at,
                    one.name,
                    _briefly(one.about, self.size.width),
                    here=held[at] == self._was,
                    inforce=one.name == self._flow,
                ),
                id=held[at],
            )
            for at, one in enumerate(mine)
        ]
        if not rows and held:
            rows = [
                Option(
                    f"{_INDENT}  [$text-muted]{self._empty(self._where)}[/]", id=held[0]
                )
            ]
        listing.set_options(rows)
        listing.highlighted = held.index(self._was) if self._was in held else None
        self._drawn = listing.highlighted
        said = self._nothing()
        self.query_one("#tuning", Label).update(
            f"[$text-muted]{said}[/]" if said else ""
        )
        self.query_one("#keys", Label).update(
            f"Enter to choose · f copies it here · Esc to close{self.searching()}"
        )

    def _empty(self, whose: str) -> str:
        """What a place with no flows in it says on the row where its flows would be."""
        verse = self._verse(whose)
        if verse is not None and not verse.fetched:
            return "not fetched yet; /flowverses fetches it"
        return "nothing in it yet"

    def _nothing(self) -> str:
        """What to say under the flows: how a fetch went, or that a search found nothing."""
        if self._fetching:
            return f"fetching {escape(self._fetching)}…"
        if self._said:
            return self._said
        if self._typed and not any(self.fits(one.name) for one in self._all()):
            return "no flow of that name"
        return ""

    def _agents_page(self) -> None:
        """Puts up each agent the flow drives, followed by saving the complete setup."""
        listing = self.query_one("#choices", OptionList)
        named = tuple(place.name for place in self._places)
        lines = reads(named, self._runs)
        total = len(self._places) + 1
        self._counting = len(str(total))
        at = min(listing.highlighted or 0, total - 1)
        rows = [
            Option(
                self._row(
                    seen,
                    called(self._places, seen),
                    lines[seen].split(_DOT, 1)[-1]
                    if self._runs[seen].spec
                    else "not chosen yet",
                    here=seen == at,
                    inforce=False,
                ),
                id=f"={seen}",
            )
            for seen in range(len(self._places))
        ]
        rows.append(
            Option(
                self._row(
                    len(self._places),
                    _SAVE,
                    "save the flow and all of its agents",
                    here=at == len(self._places),
                    inforce=False,
                ),
                id=f"={_SAVE}",
            )
        )
        listing.set_options(rows)
        listing.highlighted = at
        self._drawn = listing.highlighted
        said = self._said or ("" if self._places else self._noagents())
        self.query_one("#tuning", Label).update(
            f"[$text-muted]{said}[/]" if said else ""
        )
        self.query_one("#keys", Label).update(
            "Enter to save · Esc to close"
            if at == len(self._places)
            else "Enter to set one up · Esc to close"
        )

    def _noagents(self) -> str:
        """Why there is no agent to set up, which is not always the same reason."""
        if places_of(self._flow) is None:
            return f"{escape(self._flow)} will not load; nothing here can be set up"
        return f"{escape(self._flow)} drives no agents; it talks only to you"

    @work
    async def _configures(self) -> None:
        """Asks what the flow itself takes, and turns to what will drive it.

        Which is the moment to ask it: a flow that takes settings has just been chosen, and
        what it is set up with is a thing about the flow rather than about its agents. A flow
        that takes none is not asked -- a sheet with nothing on it is not a question -- and
        the walk is the same either way, so nobody has to know which kind they picked.
        """
        model = model_of(self._flow)
        if model is not None:
            showing = cast(
                "App[None]",
                self.app,  
            )
            held = await showing.push_screen_wait(
                Configures(
                    self._flow,
                    model,
                    self._config if isinstance(self._config, model) else None,
                )
            )
            if held is not None:
                self._config = held
                self.changed()

        self._said = ""
        self._turn_page(1)

    def action_fork(self) -> None:
        """Copies the flow under the cursor into this project's own, to be changed.

        A flow is a directory, so a copy of one is a flow of yours: the entry point, what it
        imports and the skills it brings all come across, under the name it already had --
        and your own flows are looked in first, so from then on that name means your copy.

        Which is the way to change a flow at all. A flowverse is somebody else's repository,
        fetched again over whatever was written into it, so an edit made there is an edit
        that goes away; a copy here is yours, and is what `f` is for.
        """
        from hmz.flows import LOCAL

        if self._tab != _FLOW_PAGE:
            return
        named = self._was.partition(_HALVES)[2]
        if not named:
            self._said = "no flow under the cursor to copy"
            self._fill()
            return
        try:
            at = _hmz().flows.fork(named)
        except (OSError, ValueError) as why:
            self._said = escape(str(why))
            self._fill()
            return

        self._offers, self._was = None, ""
        self._where = LOCAL
        mine = escape(named.rpartition("/")[2])
        self._said = (
            f"copied to {escape(at)} -- yours to change, and {mine} now means it"
        )
        self._fill()

    async def _fetches(self, named: str, doing: Callable[[], str]) -> None:
        """Runs one git fetch off the event loop, and shows the list it left behind.

        Off the loop because a clone is seconds of network: a menu that stopped redrawing
        while it ran would be one that looked as though it had gone away. What is being read
        is left where it is: this is the flowverse nobody has fetched being fetched because
        its flows are wanted, rather than somebody asking to be taken to it.

        Args:
          named: What is being fetched, said under the list while it runs.
          doing: What to do, answering with the flowverse it left behind.
        """
        import asyncio

        if self._fetching:
            return
        self._fetching, self._said = named or "it", ""
        self._fill()
        try:
            await asyncio.to_thread(doing)
        except (OSError, ValueError) as why:

            self._said = escape(str(why))
            self._fetching = ""
            self._fill()
            return
        self._fetching, self._offers = "", None
        self._fill()

    @on(OptionList.OptionSelected)
    def _took(self, event: OptionList.OptionSelected) -> None:
        """Chooses the flow under the cursor, or opens the agent under it.

        Args:
          event: What was chosen.
        """
        if self._tab == _FLOW_PAGE:
            _, _, name = str(event.option.id or "").partition(_HALVES)
            if name:
                self._chose(name)
            return
        held = str(event.option.id or "").removeprefix("=")
        if held == _SAVE:
            self.applied()
            return
        try:
            at = int(held)
        except ValueError:
            return
        self._configuring(at)

    def _chose(self, name: str) -> None:
        """Takes a flow as the one to run, and reads back what it was last set up with.

        Nothing is written down: what the menu holds is a draft, and a flow chosen and then
        walked away from must leave the interface exactly as ready as it was.

        Args:
          name: The flow, by the name it was offered under.
        """
        if name != self._flow:
            places = places_of(name)
            if places is None:
                self._said = f"{escape(name)} will not load"
                self._fill()
                return
            self._flow, self._places = name, places
            self._runs = self._fitted(
                settled(self._remembered(name), places, self._agents)
            )
            self._config = config_of(name, self._held(name).get("config") or {})
            self.changed()

        self._configures()

    @work
    async def _configuring(self, at: int) -> None:
        """Opens one agent of the flow, and holds whatever comes back as a draft.

        Args:
          at: Which of them, counting from zero.
        """
        if not 0 <= at < len(self._places):
            return
        showing = cast(
            "App[None]",
            self.app,  
        )
        chosen = await showing.push_screen_wait(
            Agent(
                called(self._places, at),
                self._runs[at],
                self._agents,
                place=self._places[at],
                unavailable=self._unavailable,
            )
        )
        if chosen is None:
            return  
        self._runs[at] = chosen.runs
        self.changed()
        self._fill()

    def applied(self) -> None:
        """Answers with the flow, its agents and how it is set up, all of it at once.

        Unless one of them has not been answered: a flow driven by an agent that names no
        model is a flow that stops on its first turn, and the page it would be answered on is
        the page to be looking at when that is said.
        """
        missing = [
            called(self._places, at)
            for at, one in enumerate(self._runs)
            if not _complete(one)
        ]
        if missing:
            self._tab = _AGENT_PAGE
            telemetry.snag("save-refused", missing=len(missing))
            self._said = f"{escape(', '.join(missing))} has no model yet"
            self._fill()
            return
        self.dismiss(Chosen(self._flow, tuple(self._runs), self._config))

def _added(url: str, name: str) -> str:
    """Fetches a flowverse and answers with what it is called here."""
    return _hmz().verses.add(url, name).name

def _came_from(one: Flowverse) -> str:
    """Where a flowverse came from, as a row may show it.

    Asked of which flowverse it is rather than of whether its URL is empty: an empty URL
    means both `the package's own` and `a directory whose origin could not be read`, and
    answering the second with the first would put humanize's name on somebody else's flows.

    Args:
      one: The flowverse.

    Returns:
      The URL with whatever was signed into it taken out -- a private one is added as
      `https://x-access-token:$TOKEN@...`, and this is drawn where somebody can read it --
      or, for the ones fetched from nowhere, what they are instead: the package's own flows,
      and the directory each of yours is read from.
    """
    return _hmz().verses.whence(one, "not a clone of anything")

class Holds(Sheet[None]):
    """What one flowverse holds, which is read rather than chosen from.

    A reading and not a menu: which flow to run is asked on `/flow`, where the flows of every
    place are walked. This is the other question -- what is in this one -- and it is the one
    question about a flowverse that costs something to answer, since what a file holds is not
    a fact its name carries: reading a flow means running it.
    """

    LETTERS: ClassVar = frozenset({"search"})

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),
        Binding("s", "search", "search", priority=True),
    ]

    def __init__(self, one: Flowverse) -> None:
        """Reads one flowverse's flows.

        Args:
          one: The flowverse.
        """
        super().__init__()
        self._verse = one
        self._offers: list[Offer] | None = None

    def _ask(self) -> None:
        """Says which flowverse this is, and puts its flows up."""
        self.query_one("#asked", Label).update(self._verse.name)
        self.query_one("#about", Label).update(
            f"What this flowverse holds, read from {escape(_came_from(self._verse))}. "
            "Which of them to run is asked on /flow, where every place's flows are."
        )
        self._fill()
        self.query_one("#choices", OptionList).focus()

    def _flows(self) -> list[Offer]:
        """The flows it holds, read once: reading one means running its entry point."""
        if self._offers is None:
            try:
                self._offers = _hmz().verses.holds(self._verse)
            except OSError:
                self._offers = []
        return self._offers

    def _fill(self) -> None:
        """Puts the flows up, each with the line it says about itself."""
        listing = self.query_one("#choices", OptionList)
        shown = [one for one in self._flows() if self.fits(one.name, one.about)]
        self._counting = len(str(max(len(shown), 1)))
        at = min(listing.highlighted or 0, max(len(shown) - 1, 0))
        listing.set_options(
            Option(
                self._row(
                    seen,
                    one.name,
                    _briefly(one.about, self.size.width),
                    here=seen == at,
                    inforce=False,
                ),
                id=f"={one.name}",
            )
            for seen, one in enumerate(shown)
        )
        listing.highlighted = at if shown else None
        self._drawn = listing.highlighted
        said = "" if shown else self._nothing()
        self.query_one("#tuning", Label).update(
            f"[$text-muted]{said}[/]" if said else ""
        )
        self.query_one("#keys", Label).update(f"Esc to close{self.searching()}")

    def _nothing(self) -> str:
        """Why there is nothing in it, which is not always the same reason."""
        if not self._verse.fetched:
            return "not fetched yet; r fetches it"
        if self._typed:
            return "no flow of that name in it"
        return "nothing in it: a flowverse keeps its flows in flows/"

class Flowverses(Sheet[list[str]]):
    """The places flows come from: what there is, what one holds, and what can happen to one.

    Its own menu rather than keys on the one a flow is chosen at. Adding a repository,
    fetching one again and taking one away are things done to the list of places rather than
    to the flow under the cursor, and a sheet that asks `which flow` with three keys on it
    about something else is a sheet asking two questions. `/flow` still steps between the
    places with the arrows, that being about which list of flows is being read.

    What happens here happens as it is asked for rather than being held until the menu is
    saved: each of these runs git, and something that has already been cloned is not a draft.
    """

    LETTERS: ClassVar = frozenset({"search", "adding", "refresh", "drop"})

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),
        Binding("s", "search", "search", priority=True),
        Binding("a", "adding", "add one", priority=True),
        Binding("r", "refresh", "fetch it again", priority=True),
        Binding("d", "drop", "take it away", priority=True),
    ]

    def __init__(self) -> None:
        """Reads every flowverse there is."""
        super().__init__()
        self._found: list[Flowverse] = []

        self._was = ""
        
        self._said = ""
        
        self._fetching = ""
        
        self._told: list[str] = []

    def _ask(self) -> None:
        """Says what these are, and puts them up."""
        self.query_one("#asked", Label).update("Flowverses")
        self.query_one("#about", Label).update(
            "Where flows come from: a git repository with a flows/ directory apiece, cloned "
            "under humanize's home, and the flows of your own read where they lie. Each is "
            "offered under its name here. Enter says what one holds."
        )
        self._read()
        self._fill()
        self.query_one("#choices", OptionList).focus()

    def _read(self) -> None:
        """Reads the flowverses off the disk, which is what the rows are drawn from."""
        self._found = _hmz().verses.all()

    def _about(self, one: Flowverse) -> str:
        """What a row says about one flowverse: where it came from, and whether it is here."""
        said = _came_from(one)
        if not one.fetched:
            return f"{said}{_DOT}not fetched yet"
        return said

    def _fill(self) -> None:
        """Puts the flowverses up, marked where the cursor is."""
        listing = self.query_one("#choices", OptionList)
        self._follows(listing)
        shown = [one for one in self._found if self.fits(one.name, one.url)]
        self._counting = len(str(max(len(shown), 1)))
        if all(one.name != self._was for one in shown):
            self._was = shown[0].name if shown else ""
        listing.set_options(
            Option(
                self._row(
                    seen,
                    one.name,
                    self._about(one),
                    here=one.name == self._was,
                    inforce=False,
                ),
                id=f"={one.name}",
            )
            for seen, one in enumerate(shown)
        )
        listing.highlighted = (
            next((at for at, one in enumerate(shown) if one.name == self._was), 0)
            if shown
            else None
        )
        self._drawn = listing.highlighted
        said = f"fetching {escape(self._fetching)}…" if self._fetching else self._said
        self.query_one("#tuning", Label).update(
            f"[$text-muted]{said}[/]" if said else ""
        )
        self.query_one("#keys", Label).update(
            "Enter says what one holds · a adds one · r fetches one again · "
            f"d twice takes one away · Esc to close{self.searching()}"
        )

    def _follows(self, listing: OptionList) -> None:
        """Takes which flowverse the cursor is on off the list, by name."""
        at = listing.highlighted
        if at is not None and 0 <= at < listing.option_count:
            named = str(listing.get_option_at_index(at).id or "").removeprefix("=")
            if named:
                self._was = named

    def _under(self) -> Flowverse | None:
        """The flowverse the cursor is on, or None where the list has nothing in it."""
        return next((one for one in self._found if one.name == self._was), None)

    @on(OptionList.OptionSelected)
    def _took(self, event: OptionList.OptionSelected) -> None:
        """Opens what the flowverse under the cursor holds.

        Args:
          event: What was chosen.
        """
        named = str(event.option.id or "").removeprefix("=")
        one = next((each for each in self._found if each.name == named), None)
        if one is not None:
            self._holds(one)

    @work
    async def _holds(self, one: Flowverse) -> None:
        """Reads what one flowverse holds, which means running each flow in it."""
        showing = cast(
            "App[None]",
            self.app,  
        )
        await showing.push_screen_wait(Holds(one))
        self._fill()

    @work
    async def action_adding(self) -> None:
        """Asks where a flowverse is and what to call it here, and clones it."""
        showing = cast(
            "App[None]",
            self.app,  
        )
        said = await showing.push_screen_wait(Fetches())
        if said is None:
            return
        url, name = said
        await self._fetches(name or url, lambda: _added(url, name))

    @work
    async def action_refresh(self) -> None:
        """Fetches the flowverse under the cursor again, or for the first time."""
        one = self._under()
        if one is None:
            return
        if not one.url:
            from hmz.flows.verses import MINE

            said = (
                f"is read from {MINE[one.name]}"
                if one.name in MINE
                else "came with humanize"
            )
            self._said = f"{escape(one.name)} {said}; there is nothing to fetch"
            self._fill()
            return
        name = one.name

        def fetching() -> str:
            _hmz().verses.fetch(name)
            return name

        await self._fetches(name, fetching)

    def action_drop(self) -> None:
        """Takes the flowverse under the cursor away, flows and all, once d is twice."""
        one = self._under()
        if one is None:
            return
        if not self._armed(one.name):
            self._said = f"press d again to take {escape(one.name)} away, flows and all"
            self._fill()
            return
        try:
            _hmz().verses.remove(one.name)
        except (OSError, ValueError) as why:
            self._said = escape(str(why))
            self._fill()
            return
        self._said = f"{escape(one.name)} is no longer here"
        self._told.append(f"[dim]{escape(one.name)} is no longer here[/dim]")
        self._was = ""
        self._read()
        self._fill()

    async def _fetches(self, named: str, doing: Callable[[], str]) -> None:
        """Runs one git fetch off the event loop, and shows the list it left behind.

        Off the loop because a clone is seconds of network: an interface that stopped
        redrawing while it ran would be one that looked as though it had gone away.

        Args:
          named: What is being fetched, said under the list while it runs.
          doing: What to do, answering with the flowverse it left behind.
        """
        import asyncio

        if self._fetching:
            return
        self._fetching, self._said = named or "it", ""
        self._fill()
        try:
            name = await asyncio.to_thread(doing)
        except (OSError, ValueError) as why:

            self._said, self._fetching = escape(str(why)), ""
            self._fill()
            return
        self._fetching = ""
        self._said = f"{escape(name)} is fetched"
        self._told.append(f"[dim]{escape(name)} is fetched[/dim]")
        self._read()
        self._was = name
        self._fill()

    def leaving(self) -> None:
        """Leaves, saying in the transcript whatever happened while this was open."""
        self.dismiss(self._told or None)

def _written(
    at: int, counting: int, named: str, about: str, shown: str, *, here: bool
) -> str:
    """One row that is written into rather than picked between.

    Args:
      at: Which one it is, counting from zero.
      counting: How wide the numbering is, so every row starts in the same column.
      named: What the answer is kept under.
      about: What is being asked, said quietly beside it.
      shown: What has been typed, as it is to be shown.
      here: Whether the cursor is on it.

    Returns:
      The row, as markup.
    """
    mark = f"{_INDENT}[$primary]{_HERE}[/] " if here else f"{_INDENT}  "
    number = f"{at + 1:>{counting}}."

    caret = "[reverse] [/reverse]" if here else ""
    
    label = escape(named) + " " * max(1, _SETTING - len(named))
    room = _VALUE - len(shown) - 1
    return (
        f"{mark}[$text-muted]{number}[/] {label}"
        f"[$secondary]{escape(shown)}[/]{caret}{' ' * max(1, room)}"
        f"[$text-muted]{escape(about)}[/]"
    )

def _briefly(said: str, width: int) -> str:
    """One flow's line about itself, clipped to the room the row has for it.

    Args:
      said: The line, which is the first line of what the flow says about itself and so is
        as long as that sentence is.
      width: How wide the sheet is.

    Returns:
      As much of it as fits beside the name, ending in an ellipsis where it was cut.
    """
    room = max(width - len(_INDENT) - _LABEL - 8, 20)
    return said if len(said) <= room else f"{said[: room - 1].rstrip()}…"

class Fetches(Sheet[tuple[str, str]]):
    """Where a flowverse is, and what it is to be called here.

    A form rather than a list, as signing in to an account is: there is nothing to pick, both
    rows being written where they stand.
    """

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),
        Binding("enter", "done", "done", priority=True),
    ]

    _ASKS = (
        ("repository", "a URL, or owner/repo for one on GitHub"),
        ("name", "what to call it here, blank for the repository's own name"),
    )

    def __init__(self) -> None:
        """Initializes the asking."""
        super().__init__()
        self._counting = len(str(len(self._ASKS)))
        self._typed_in: dict[str, str] = {}
        
        self._wrong = ""

    def _ask(self) -> None:
        """Says what a flowverse is, and what the keys do while it is being named."""
        self.query_one("#asked", Label).update("Add a flowverse")
        self.query_one("#about", Label).update(
            "A git repository with a `flows/` directory in it: one `.py` file per flow, and "
            "whatever they import beside them. It is cloned into ~/.humanize/flowverses, and "
            "every flow in it is then offered under the name it is kept under."
        )
        self._fill()
        self.query_one("#choices", OptionList).focus()

    def _fill(self) -> None:
        """Puts the two rows up, with the caret in the one under the cursor."""
        listing = self.query_one("#choices", OptionList)
        at = self._at
        listing.set_options(
            Option(
                _written(
                    seen,
                    self._counting,
                    held,
                    about,
                    self._typed_in.get(held, ""),
                    here=seen == at,
                ),
                id=f"={held}",
            )
            for seen, (held, about) in enumerate(self._ASKS)
        )
        listing.highlighted = at
        self._drawn = at
        self.query_one("#tuning", Label).update(
            f"[$error]{escape(self._wrong)}[/]" if self._wrong else ""
        )
        self.query_one("#keys", Label).update(
            "Type to answer · Backspace to rub out · Enter to fetch it · Esc to go back"
        )

    @property
    def _at(self) -> int:
        """Which row the cursor is on, counting from zero."""
        listing = self.query_one("#choices", OptionList)
        return min(listing.highlighted or 0, len(self._ASKS) - 1)

    def on_key(self, event: events.Key) -> None:
        """Takes a letter as answering the row under the cursor.

        Args:
          event: The key.
        """
        held = self._ASKS[self._at][0]
        if event.key == "backspace":
            self._typed_in[held] = self._typed_in.get(held, "")[:-1]
        elif event.is_printable and event.character:
            self._typed_in[held] = self._typed_in.get(held, "") + event.character
        else:
            return
        event.prevent_default()
        event.stop()
        self._wrong = ""
        self._fill()

    def action_done(self) -> None:
        """Answers with where it is and what to call it, once there is somewhere to fetch."""
        url = self._typed_in.get("repository", "").strip()
        name = self._typed_in.get("name", "").strip()
        if not url:
            self._wrong = "a flowverse is a repository, and none was named"
            self._fill()
            return
        if name:
            try:
                _hmz().verses.where(name)
            except ValueError as why:
                self._wrong = str(why)
                self._fill()
                return
        self.dismiss((url, name))

class Speaks(Sheet[tuple[str, str]]):
    """A CLI of your own that speaks the Agent Client Protocol, and what starts it.

    A form rather than a list, as adding a flowverse is: there is nothing to pick, both rows
    being written where they stand. Two questions because the protocol answers neither -- it
    has no discovery and no flag every agent agrees on -- so the command is asked for, and the
    name it is to be known by here is asked for beside it.
    """

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),
        Binding("enter", "done", "done", priority=True),
    ]

    _ASKS = (
        ("command", "what starts it, as you would type it: my-agent --acp"),
        ("name", "what to call it here, blank for the command's own name"),
    )

    def __init__(self) -> None:
        """Initializes the asking."""
        super().__init__()
        self._counting = len(str(len(self._ASKS)))
        self._typed_in: dict[str, str] = {}
        
        self._wrong = ""

    def _ask(self) -> None:
        """Says what one of these is, and what the keys do while it is being named."""
        self.query_one("#asked", Label).update("Add a CLI that speaks ACP")
        self.query_one("#about", Label).update(
            "Any coding agent that speaks the Agent Client Protocol can be driven from here. "
            "humanize spawns the command you give and talks to it over its own stdin and "
            "stdout. The protocol says nothing about which models it runs or how hard it can "
            "be asked to think, so it runs as whoever installed it configured it."
        )
        self._fill()
        self.query_one("#choices", OptionList).focus()

    def _fill(self) -> None:
        """Puts the two rows up, with the caret in the one under the cursor."""
        listing = self.query_one("#choices", OptionList)
        at = self._at
        listing.set_options(
            Option(
                _written(
                    seen,
                    self._counting,
                    held,
                    about,
                    self._typed_in.get(held, ""),
                    here=seen == at,
                ),
                id=f"={held}",
            )
            for seen, (held, about) in enumerate(self._ASKS)
        )
        listing.highlighted = at
        self._drawn = at
        self.query_one("#tuning", Label).update(
            f"[$error]{escape(self._wrong)}[/]" if self._wrong else ""
        )
        self.query_one("#keys", Label).update(
            "Type to answer · Backspace to rub out · Enter to add it · Esc to go back"
        )

    @property
    def _at(self) -> int:
        """Which row the cursor is on, counting from zero."""
        listing = self.query_one("#choices", OptionList)
        return min(listing.highlighted or 0, len(self._ASKS) - 1)

    def on_key(self, event: events.Key) -> None:
        """Takes a letter as answering the row under the cursor.

        Args:
          event: The key.
        """
        held = self._ASKS[self._at][0]
        if event.key == "backspace":
            self._typed_in[held] = self._typed_in.get(held, "")[:-1]
        elif event.is_printable and event.character:
            self._typed_in[held] = self._typed_in.get(held, "") + event.character
        else:
            return
        event.prevent_default()
        event.stop()
        self._wrong = ""
        self._fill()

    def action_done(self) -> None:
        """Answers with the command and the name, once there is something to start."""
        said = self._typed_in.get("command", "").strip()
        name = self._typed_in.get("name", "").strip()
        if not said:
            self._wrong = "nothing was given to start it with"
            self._fill()
            return
        try:
            argv = shlex.split(said)
        except ValueError as why:  
            self._wrong = str(why)
            self._fill()
            return
        if not argv:
            self._wrong = "nothing was given to start it with"
            self._fill()
            return
        self.dismiss((said, name or Path(argv[0]).name))

class Skills(Sheet[None]):
    """What one CLI would load here, shown and not touched.

    A skill installed on this machine is that CLI's own: installed the way it installs one,
    switched off the way it switches one off, and the same for every agent of every flow.
    humanize used to switch them per agent and no longer does -- what a person has installed
    is not something a flow is entitled to rewrite, and a list that could be adjusted here
    while the CLI's own list said otherwise was two answers to one question.

    So this is a reading: what the agent will be carrying, where each of them came from, and
    the line saying where to go to change it. What humanize does add is the flow's own
    skills, which are mounted onto the sessions it opens rather than installed here.
    """

    LETTERS: ClassVar = frozenset({"search"})

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),
        Binding("s", "search", "search", priority=True),

        Binding("enter", "back", "done", priority=True),
    ]

    def __init__(self, backend: str) -> None:
        """Initializes the reading.

        Args:
          backend: The CLI whose skills these are.
        """
        super().__init__()
        self._backend = backend
        self._found: list[Skill] | None = None

    def _ask(self) -> None:
        """Says whose skills these are, and who is to be asked to change them."""
        self.query_one("#asked", Label).update(f"What {self._backend} loads here")
        self.query_one("#about", Label).update(
            "The skills this CLI finds, which every agent of it carries. They are its own: "
            f"install one, or switch one off, the way {escape(self._backend)} itself does "
            "it. A flow's own skills are mounted onto the sessions it opens and are not "
            "installed here."
        )
        self._fill()
        self.query_one("#choices", OptionList).focus()

    def _skills(self) -> list[Skill]:
        """The skills there are to show, read once: this is redrawn per keystroke."""
        if self._found is None:
            self._found = skills(self._backend)
            self._counting = len(str(len(self._found)))
        return self._found

    def _fill(self) -> None:
        """Puts the skills up, each with where it came from."""
        listing = self.query_one("#choices", OptionList)
        shown = [
            skill
            for skill in self._skills()
            if self.fits(skill.name, skill.about, skill.whose)
        ]
        at = min(listing.highlighted or 0, max(len(shown) - 1, 0))
        listing.set_options(
            Option(
                self._row(
                    seen,
                    skill.name,
                    f"{skill.about}  ({skill.whose})" if skill.about else skill.whose,
                    here=seen == at,
                    inforce=False,
                ),
                id=skill.name,
            )
            for seen, skill in enumerate(shown)
        )
        listing.highlighted = at if shown else None
        self._drawn = at
        self.query_one("#tuning", Label).update(
            f"[$text-muted]{self._said()}[/]" if self._said() else ""
        )
        self.query_one("#keys", Label).update(f"Esc to go back{self.searching()}")

    def _said(self) -> str:
        """The line under the list: where to go to change any of this, or why there is none.

        Returns:
          That these are the CLI's own and are managed there, for a CLI that keeps skills;
          that a CLI which keeps none anywhere has none to show; and, where it keeps them
          and none is installed, that there are none here yet.
        """
        profile = named(self._backend)
        if profile is None or not (
            profile.skills or profile.shared or profile.config or profile.works
        ):
            return f"{escape(self._backend)} keeps no skills of its own here"
        if not self._skills():
            return (
                f"{escape(self._backend)} has none installed here; install one the way "
                f"{escape(self._backend)} installs one"
            )
        return (
            f"These are {escape(self._backend)}'s own: add one, or switch one off, where "
            f"{escape(self._backend)} keeps them"
        )

class Anchors(Sheet[str]):
    """Where one agent's turns land: this machine, or one an anchor reaches.

    A row of the sheet one agent is set up on, and only for a place the flow declared
    `Remote`: a flow that says so is one that expects to be told where that agent works, and
    one that said nothing has said its agent works here.

    The agent itself runs here whatever is chosen -- its credentials, its state directory and
    its link to its model provider stay put. What moves is the project it reads and the
    commands it runs, which is why this is a question about the agent rather than about the
    flow: two agents of one flow may work on two machines.

    Listed rather than typed where the machine is one this one can see -- a container that is
    running, a host with an entry in the ssh config -- and typed where it is not: a target is
    a string, and the row for what has been typed appears among them, as soon as it reads as
    one, while a search is running.
    """

    LETTERS: ClassVar = frozenset({"search"})

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),
        Binding("s", "search", "search", priority=True),
    ]

    def __init__(self, named: str, current: str = "") -> None:
        """Initializes the moving.

        Args:
          named: What the flow calls the agent this is about, which every step of configuring
            it says.
          current: The target this agent is on now, or "" for this machine.
        """
        super().__init__()
        self._named = named
        self._current = current
        self._found: list[tuple[str, str]] | None = None

    def _ask(self) -> None:
        """Lists the machines there are to work on, and says what choosing one does."""
        self.query_one("#asked", Label).update(f"Select where {self._named} works")
        self.query_one("#about", Label).update(
            "The machine its work lands on. The agent still runs here; what moves is the "
            "project it reads and the commands it runs. Anywhere else is a target you type "
            "-- ssh://HOST, docker://CONTAINER, tcp://HOST:PORT."
        )
        self.query_one("#tuning", Label).update("")
        self._fill()

    def _fill(self) -> None:
        """Puts the machines up, with whatever has been typed among them if it reads as one."""
        listing = self.query_one("#choices", OptionList)
        if self._found is None:
            
            self._found = machines()
        rows: list[tuple[str, str, str]] = [("", "this machine", "nothing moves")]
        rows.extend((target, target, whose) for target, whose in self._found)
        shown = [row for row in rows if self.fits(row[1], row[2])]
        if self._typed and not any(row[0] == self._typed for row in shown):

            try:
                anchored(self._typed)
            except ValueError:
                pass
            else:
                shown.append((self._typed, self._typed, "as typed"))
        self._counting = len(str(len(shown)))
        at = min(listing.highlighted or 0, max(len(shown) - 1, 0))
        listing.set_options(
            Option(
                self._row(
                    seen, label, whose, here=seen == at, inforce=target == self._current
                ),

                id=f"={target}",
            )
            for seen, (target, label, whose) in enumerate(shown)
        )
        listing.highlighted = at if shown else None
        self._drawn = at
        self.query_one("#keys", Label).update(
            f"Enter to choose · s then a target names one of your own{self.searching()}"
        )

    @on(OptionList.OptionSelected)
    def _took(self, event: OptionList.OptionSelected) -> None:
        """Answers with the target that was picked.

        Args:
          event: What was chosen.
        """
        self.dismiss(str(event.option.id).removeprefix("="))

class Falls(Sheet[str]):
    """Which account a turn under this one carries on under when it fails.

    A name rather than a mark: each account names the next, so what a turn walks is a chain
    -- a subscription that runs out falls to a key, and a key that is refused falls to a
    gateway -- rather than there being one place every failure of that CLI goes.

    Only that CLI's own accounts are offered: an account is credentials for one backend, and
    a turn cannot be carried on under credentials for another.
    """

    LETTERS: ClassVar = frozenset({"search"})

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),
        Binding("s", "search", "search", priority=True),
    ]

    def __init__(self, cli: str, name: str, current: str = "") -> None:
        """Initializes the choosing.

        Args:
          cli: The backend these accounts are of.
          name: The account this is about, which is not among the ones offered.
          current: What it falls back to now, or "" for the end of the line.
        """
        super().__init__()
        self._cli = cli
        self._name = name
        self._current = current
        self._found: list[Provider] | None = None

    def _ask(self) -> None:
        """Says whose accounts these are, and what carrying on under one means."""
        self.query_one("#asked", Label).update(
            f"Where {self._cli}/{self._name} falls back to"
            if self._name
            else f"Where {self._cli}, as this machine is signed in, falls back to"
        )
        self.query_one("#about", Label).update(
            "The account a turn under this one carries on under, once the tries the place "
            "was given are spent. It happens inside the conversation that was running, and "
            "the account it moves to has a fallback of its own -- so what a turn walks is a "
            "chain, to the end of it."
        )
        self.query_one("#tuning", Label).update("")
        self._fill()

    def _accounts(self) -> list[Provider]:
        """That CLI's own accounts, read once: this is redrawn per keystroke."""
        if self._found is None:
            self._found = [
                one for one in _hmz().accounts.all(self._cli) if one.name != self._name
            ]
        return self._found

    def _fill(self) -> None:
        """Puts the accounts up, with the end of the line first."""
        listing = self.query_one("#choices", OptionList)
        rows: list[tuple[str, str, str]] = [
            ("", "nowhere", "the end of the line: a failed turn is a failed turn")
        ]
        rows.extend((one.name, one.name, _sets(one)) for one in self._accounts())
        shown = [row for row in rows if self.fits(row[1], row[2])]
        self._counting = len(str(max(len(shown), 1)))
        at = min(listing.highlighted or 0, max(len(shown) - 1, 0))
        listing.set_options(
            Option(
                self._row(
                    seen, label, about, here=seen == at, inforce=name == self._current
                ),
                id=f"={name}",
            )
            for seen, (name, label, about) in enumerate(shown)
        )
        listing.highlighted = at if shown else None
        self._drawn = at
        self.query_one("#tuning", Label).update(
            ""
            if self._accounts()
            else f"[$text-muted]{escape(self._cli)} has no other account to fall back "
            "to; a on the menu behind this makes one[/]"
        )
        self.query_one("#keys", Label).update(
            f"Enter to choose · Esc to go back{self.searching()}"
        )

    @on(OptionList.OptionSelected)
    def _took(self, event: OptionList.OptionSelected) -> None:
        """Answers with the account that was picked, or "" for the end of the line."""
        self.dismiss(str(event.option.id).removeprefix("="))

_TRIES = (0, 1, 2, 3, 5, 8, 13, 21)
_FOR = (0.0, 30.0, 60.0, 300.0, 900.0, 3600.0)

_HOW_MANY = "tries"
_POLICY = "policy"
_HOW_LONG = "for"

class Retries(Sheet[tuple[int, str, float]]):
    """How a turn at one place is tried again before it falls back to another.

    A turn fails for two kinds of reason and only one of them is worth another try: a prompt
    the model refused is the same refusal every time, and a gateway that answered 503 is the
    same call away from working. So a place says how many tries it gets, how long to wait
    between them, and how long the whole of it may go on for.

    Three rungs rather than three things to type: each is stepped where it stands, which is
    how every other setting in an order is answered here.
    """

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),
        Binding("left", "easier", "back one", priority=True),
        Binding("right", "harder", "on one", priority=True),
        Binding("enter", "done", "done", priority=True),
    ]

    def __init__(self, named: str, tries: int, policy: str, timeout: float) -> None:
        """Initializes the sheet on what the place says now.

        Args:
          named: The place this is about, which the question at the top says.
          tries: How many tries beyond the first it gets now.
          policy: How long it waits between them now.
          timeout: The longest the retrying may go on for now, or 0.0 for no limit.
        """
        super().__init__()
        self._named = named
        self._retries = tries
        self._policy = policy
        self._timeout = timeout

    def _ask(self) -> None:
        """Says which place this is about, and what trying again means."""
        self.query_one("#asked", Label).update(f"How {self._named} is tried again")
        self.query_one("#about", Label).update(
            "What happens when a turn at this place fails. The arrows step the row under "
            "the cursor. Once the tries are spent, the turn carries on wherever this place "
            "falls back to."
        )
        self._fill()
        self.query_one("#choices", OptionList).focus()

    def _rows(self) -> list[tuple[str, str, str]]:
        """Every row this is made of: its id, what it is now, and what it means."""
        said = _hmz().fallbacks.named(self._policy)
        return [
            (
                _HOW_MANY,
                "none" if not self._retries else str(self._retries),
                "how many times over a failed turn is tried again",
            ),
            (
                _POLICY,
                self._policy,
                said.about if said is not None else "how long to wait between tries",
            ),
            (
                _HOW_LONG,
                _lasting(self._timeout),
                "the longest the trying again may go on for",
            ),
        ]

    def _fill(self) -> None:
        """Puts the three rows up, with the cursor where it was."""
        listing = self.query_one("#choices", OptionList)
        rows = self._rows()
        self._counting = len(str(len(rows)))
        at = min(listing.highlighted or 0, len(rows) - 1)
        listing.set_options(
            Option(
                self._row(
                    seen, name, f"{value}   {about}", here=seen == at, inforce=False
                ),
                id=f"={name}",
            )
            for seen, (name, value, about) in enumerate(rows)
        )
        listing.highlighted = at
        self._drawn = at
        self.query_one("#tuning", Label).update(
            "[$text-muted]none, and a failed turn is a failed turn[/]"
            if not self._retries
            else ""
        )
        self.query_one("#keys", Label).update(
            "Left and right to step this one · Enter to accept · Esc to go back"
        )

    def action_easier(self) -> None:
        """Steps the row under the cursor back one."""
        self._step(-1)

    def action_harder(self) -> None:
        """Steps it on one."""
        self._step(1)

    def _step(self, by: int) -> None:
        """Moves whichever row the cursor is on, wrapping round at either end.

        Args:
          by: One rung on or back.
        """
        listing = self.query_one("#choices", OptionList)
        at = listing.highlighted or 0
        held = self._rows()[at][0] if 0 <= at < len(self._rows()) else ""
        if held == _HOW_MANY:
            self._retries = _stepped(_TRIES, self._retries, by)
        elif held == _POLICY:
            names = [one.name for one in _hmz().fallbacks.policies()]
            self._policy = _stepped(names, self._policy, by)
        elif held == _HOW_LONG:
            self._timeout = _stepped(_FOR, self._timeout, by)
        else:
            return
        self._fill()

    def action_done(self) -> None:
        """Answers with what the place is to say from here on."""
        self.dismiss((self._retries, self._policy, self._timeout))

def _stepped[T](among: Sequence[T], held: T, by: int) -> T:
    """One rung on or back through a list, wrapping round and starting from the nearest.

    Args:
      among: The rungs, in order.
      held: What it is now, which need not be one of them -- a setting written by hand is
        stepped from the first rung rather than refused.
      by: One on or back.

    Returns:
      The rung to move to.
    """
    at = among.index(held) if held in among else 0
    return among[(at + by) % len(among)]

def _lasting(seconds: float) -> str:
    """How long something may go on for, as a row of a sheet says it."""
    if not seconds:
        return "as long as it takes"
    if seconds < 60:  
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.0f}m"

_SETTING = 34
_VALUE = 13

_ON = "on"
_OFF = "off"

def _shown(value: object) -> str:
    """One setting's value, as a line about it says it.

    Args:
      value: What it is set to.

    Returns:
      A switch as `on` or `off`, anything else as it is written, and something unset as the
      empty string rather than as `None` -- a setting nobody has given a value is blank.
    """
    if isinstance(value, bool):
        return _ON if value else _OFF
    return "" if value is None else str(value)

def _grouped(field: FieldInfo) -> str:
    """Which part of the sheet a setting belongs under, if the flow said.

    A flow groups its settings by writing `json_schema_extra={"section": "..."}` where it
    declares them: twenty settings in one list is a list nobody reads, and the flow is the
    only thing that knows which of them belong together.

    Args:
      field: The field, as the model declared it.

    Returns:
      The heading to draw above it, or "" for a flow that grouped nothing.
    """
    extra = field.json_schema_extra
    if not isinstance(extra, dict):
        return ""
    said = cast("dict[str, Any]", extra).get("section")
    return str(said) if said else ""

def _flowing(started: str) -> list[str]:
    """Which flow is running, and inside which, for the row that names one.

    A flow may reach for another by name and run it, so the flow a run is in is not always
    the flow that was started -- and a sheet that named only the one somebody chose would be
    a sheet that stopped being true the moment a flow called another.

    Args:
      started: The flow that was chosen, which is what this says with nothing running.

    Returns:
      One line apiece, the one that was started first and whatever it called under it, each
      with how long it has been going; and just the one that is set up to run where nothing
      is running.
    """
    now = _hmz().flows.running()
    if not now:
        return [escape(started)]
    return [
        f"{'  ' * at}{'▸ ' if at else ''}{escape(one.flow)}"
        f"   [$text-muted]{time.monotonic() - one.since:.0f}s[/]"
        for at, one in enumerate(now)
    ]

def setting(config: BaseModel | None) -> list[str]:
    """What a flow was set up with, one line per setting that is not at its default.

    Read in two places -- `/status` and the box a run opens with -- and only the settings
    that were changed: a flow with forty of them says nothing by listing the thirty-nine
    nobody touched, and the one that was touched is the thing worth reading.

    Args:
      config: What the flow was set up with, or None for a flow that takes no setting up or
        was left as it comes.

    Returns:
      One `name value` apiece, in the order the model declares them, and nothing at all for
      a flow left entirely at its defaults.
    """
    if config is None:
        return []
    return [
        f"{name:<{_SETTING}}{_shown(getattr(config, name))}"
        for name, field in type(config).model_fields.items()
        if getattr(config, name) != field.get_default(call_default_factory=True)
    ]

class Configures(Sheet["BaseModel"]):
    """How the flow is set up, asked once between choosing it and choosing its agents.

    A flow says what it can be set up with by declaring a model, and this is that model with
    a cursor on it: one row per field, the name, what it is set to, and the line the field
    was declared with. Nothing here knows what any of the settings mean -- the types say how
    a value is moved, and the model itself says which combinations it will not take, so a
    flow that refuses `gen_idea` without `gen_plan` refuses it here rather than an hour in.

    Every value is held as it is typed and handed to the model to read back, so a field is
    only ever wrong in one place: pydantic coerces `on`, `42` and `discussion` into the bool,
    the int and the literal the flow declared, and says what is wrong with anything else.
    """

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),
        ("left", "prev", "previous value"),
        ("right", "next", "next value"),

        Binding("enter", "done", "done", priority=True),
    ]

    def __init__(
        self, flow: str, model: type[BaseModel], now: BaseModel | None
    ) -> None:
        """Initializes the setting up.

        Args:
          flow: The flow these settings are for.
          model: What it says it can be set up with.
          now: How it is set up already, or None to start from the model's own defaults.
        """
        super().__init__()
        self._flow = flow
        self._model = model
        self._fields = list(model.model_fields.items())
        self._counting = len(str(len(self._fields)))

        self._typed_in: dict[str, str] = {
            name: _shown(
                getattr(now, name)
                if now is not None
                else field.get_default(call_default_factory=True)
            )
            for name, field in self._fields
        }
        
        self._wrong = ""

        self._was = 0

    def _ask(self) -> None:
        """Says what is being set up, and what the keys do while it is."""
        self.query_one("#asked", Label).update(f"Set up {self._flow}")
        self.query_one("#about", Label).update(
            "How this flow runs, which it says for itself. Left and right move a setting "
            "along, typing writes one, and enter takes the lot. What is refused here is "
            "refused by the flow rather than by this list."
        )
        self._fill()
        self.query_one("#choices", OptionList).focus()

    def _fill(self) -> None:
        """Puts the settings up, grouped, with the marker beside the one under the cursor."""
        listing = self.query_one("#choices", OptionList)
        at = self._at(listing.highlighted)
        rows: list[Option] = []
        group = ""
        for seen, (name, field) in enumerate(self._fields):
            under = _grouped(field)
            if under != group:
                group = under

                if group:
                    if rows:
                        rows.append(Option("", disabled=True))
                    rows.append(
                        Option(f"{_INDENT}[$primary]{escape(group)}[/]", disabled=True)
                    )
            rows.append(Option(self._line(seen, name, here=seen == at), id=name))
        listing.set_options(rows)
        listing.highlighted = self._row_of(at) if self._fields else None
        self._drawn = listing.highlighted
        self.query_one("#tuning", Label).update(
            f"[$error]{escape(self._wrong)}[/]" if self._wrong else ""
        )

        written = bool(self._fields) and not self._steps(self._fields[at][0])
        self.query_one("#keys", Label).update(
            "Type to set · Backspace to rub out · Enter to accept · Esc to go back"
            if written
            else "←/→ to change · Enter to accept · Esc to go back"
        )

    def _row_of(self, at: int) -> int:
        """Which row of the list one setting is on, once the headings are counted.

        Args:
          at: Which setting it is, counting from zero.

        Returns:
          The row.
        """
        rows = 0
        group = ""
        for seen, (_, field) in enumerate(self._fields):
            under = _grouped(field)
            if under != group:
                group = under
                if group:
                    rows += 2 if rows else 1
            if seen == at:
                return rows
            rows += 1
        return rows

    def _at(self, row: int | None) -> int:
        """Which setting a row of the list is, which is what the cursor is really on.

        Args:
          row: Where the cursor is, or None for a list nothing is highlighted in.

        Returns:
          The setting, counting from zero, and the nearest one where the cursor is on a
          heading -- which is where it lands when the list is first put up.
        """
        listing = self.query_one("#choices", OptionList)
        if row is not None and 0 <= row < listing.option_count:
            named = listing.get_option_at_index(row).id
            if named is not None:
                return next(
                    (
                        seen
                        for seen, (one, _) in enumerate(self._fields)
                        if one == named
                    ),
                    0,
                )
        return self._was

    def _line(self, at: int, name: str, *, here: bool) -> str:
        """One setting: what it is called, what it is set to, and what it is for.

        A setting that is written carries a caret under the cursor, where the next letter
        would land. Without it a blank one reads as a setting nothing can be typed into --
        which is the one thing about this list that has to be visible, since a switch and a
        word look the same until you try to type at one.

        Args:
          at: Which one it is, counting from zero.
          name: The field.
          here: Whether the cursor is on it.

        Returns:
          The row, as markup.
        """
        mark = f"{_INDENT}[$primary]{_HERE}[/] " if here else f"{_INDENT}  "
        number = f"{at + 1:>{self._counting}}."
        value = self._typed_in[name]
        about = dict(self._fields)[name].description or ""

        caret = "[reverse] [/reverse]" if here and not self._steps(name) else ""

        named = escape(name) + " " * max(1, _SETTING - len(name))
        room = _VALUE - len(value) - (1 if caret else 0)
        return (
            f"{mark}[$text-muted]{number}[/] {named}"
            f"[$secondary]{escape(value)}[/]{caret}{' ' * max(1, room)}"
            f"[$text-muted]{escape(about)}[/]"
        )

    @property
    def _under(self) -> str:
        """The setting the cursor is on, or "" for a model that declares none."""
        if not self._fields:
            return ""
        listing = self.query_one("#choices", OptionList)
        self._was = self._at(listing.highlighted)
        return self._fields[self._was][0]

    def _steps(self, name: str) -> tuple[str, ...]:
        """What a setting steps through, where it is one of a fixed few.

        Args:
          name: The field.

        Returns:
          Every value it takes, in the order the flow wrote them -- the two words of a
          switch, or the words of a literal -- and nothing at all for one that is written
          rather than stepped.
        """
        kind = dict(self._fields)[name].annotation

        for said in (kind, *get_args(kind)):
            if get_origin(said) is Literal:
                return tuple(str(one) for one in get_args(said))
        if kind is bool:
            return (_OFF, _ON)
        return ()

    def _move(self, by: int) -> None:
        """Moves the setting under the cursor along, however that setting moves.

        Args:
          by: One step forward or back.
        """
        name = self._under
        if not name:
            return
        if steps := self._steps(name):
            at = (
                steps.index(self._typed_in[name])
                if self._typed_in[name] in steps
                else 0
            )
            self._typed_in[name] = steps[(at + by) % len(steps)]
        elif dict(self._fields)[name].annotation in (int, float):
            try:
                now = float(self._typed_in[name] or 0)
            except ValueError:
                now = 0
            moved = now + by
            self._typed_in[name] = str(
                int(moved) if dict(self._fields)[name].annotation is int else moved
            )
        else:
            return  
        self._wrong = ""
        self._fill()

    def action_next(self) -> None:
        """Moves the setting under the cursor one value on."""
        self._move(1)

    def action_prev(self) -> None:
        """Moves the setting under the cursor one value back."""
        self._move(-1)

    def action_back(self) -> None:
        """Leaves without setting anything, which leaves the flow as it was."""
        self.dismiss(None)

    def action_done(self) -> None:
        """Reads every setting back into the model, and answers with it if it takes them.

        What the model refuses is shown where it was typed rather than raised at the flow:
        a combination the flow will not run is a combination to correct before it starts,
        and this is the moment it is being said.
        """
        from pydantic import ValidationError

        try:
            self.dismiss(self._model.model_validate(self._typed_in))
        except ValidationError as refused:
            first = refused.errors()[0]
            where = ".".join(str(part) for part in first.get("loc") or ())
            self._wrong = f"{where}: {first['msg']}" if where else str(first["msg"])
            self._fill()

    def on_key(self, event: events.Key) -> None:
        """Takes a letter as writing the setting under the cursor.

        There is nothing to search here -- every setting is on screen at once -- so the keys
        that narrow a list elsewhere are the ones that write a value.

        Args:
          event: The key.
        """
        name = self._under
        if not name or self._steps(name):
            return  
        if event.key == "backspace":
            self._typed_in[name] = self._typed_in[name][:-1]
        elif event.is_printable and event.character:
            self._typed_in[name] += event.character
        else:
            return
        event.prevent_default()
        event.stop()
        self._wrong = ""
        self._fill()

class Picks(Sheet[str]):
    """A question that is only a list of named things, answered by picking one of them.

    Two of the sheets here are that and nothing else -- which CLI a new account is for, and
    how to sign into it -- and two lists drawn two ways would read as two different kinds of
    question. So the drawing is here, and each of them says only what it asks and what there
    is to choose between.
    """

    asked = ""
    about = ""

    keys = ""

    LETTERS: ClassVar = frozenset({"search"})

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),

        Binding("s", "search", "search", priority=True),
    ]

    def __init__(self, current: str = "") -> None:
        """Initializes the choosing.

        Args:
          current: What is in force already, which is the row the tick goes against.
        """
        super().__init__()
        self._current = current
        self._rows: list[tuple[str, str, str]] | None = None

    def rows(self) -> list[tuple[str, str, str]]:
        """What there is to choose between, which each sheet says for itself.

        Returns:
          One `(what picking it answers with, what it is called, the line about it)` apiece,
          in the order to show them.
        """
        raise NotImplementedError

    def nothing(self) -> str:
        """What to say under the list where the list alone does not say it.

        Returns:
          The line, already escaped, or "" for a list that speaks for itself.
        """
        return ""

    def _ask(self) -> None:
        """Says what is being chosen, and puts the choices up."""
        self.query_one("#asked", Label).update(self.asked)
        self.query_one("#about", Label).update(self.about)
        self._fill()
        self.query_one("#choices", OptionList).focus()

    def _fill(self) -> None:
        """Puts the rows up, with the marker beside the one the cursor is on."""
        listing = self.query_one("#choices", OptionList)
        if self._rows is None:
            
            self._rows = self.rows()
        shown = [row for row in self._rows if self.fits(row[1], row[2])]
        self._counting = len(str(len(shown)))
        at = min(listing.highlighted or 0, max(len(shown) - 1, 0))
        listing.set_options(
            Option(
                self._row(
                    seen, label, about, here=seen == at, inforce=answer == self._current
                ),

                id=f"={answer}",
            )
            for seen, (answer, label, about) in enumerate(shown)
        )
        listing.highlighted = at if shown else None
        self._drawn = at
        said = self.nothing()
        self.query_one("#tuning", Label).update(
            f"[$text-muted]{said}[/]" if said else ""
        )
        self.query_one("#keys", Label).update(
            f"{self.keys}Enter to choose · Esc to cancel{self.searching()}"
        )

    @on(OptionList.OptionSelected)
    def _took(self, event: OptionList.OptionSelected) -> None:
        """Answers with what was picked.

        Args:
          event: What was chosen.
        """
        self.dismiss(str(event.option.id).removeprefix("="))

def _hmz() -> Hmz:
    """humanize, as the one object every sheet reaches a store through.

    Made where it is wanted rather than held: it costs nothing until something is asked of
    it, and a sheet that reads a store twice reads the same store both times.
    """
    from hmz.sdk import Hmz

    return Hmz()

def _sets(provider: Provider) -> str:
    """What one account says about itself on a row: the way it was made by, and what it sets.

    Args:
      provider: The account.

    Returns:
      The line, with the variables named and never a value in it -- this is drawn where
      somebody can read it, and a key on a screen is a key in a photograph.
    """
    variables = ", ".join(sorted(provider.env))
    return f"{provider.way}{_DOT}{variables}" if variables else provider.way

def _drives(backend: str) -> type[AgentBase] | None:
    """What drives one backend, or None for a name nothing here drives.

    A CLI somebody added themselves is driven too -- by the one class that speaks the Agent
    Client Protocol -- so this asks what would build it rather than reading one table.

    Args:
      backend: The backend, by name.

    Returns:
      The agent class, or None.
    """
    try:
        return driver(backend)[0]
    except KeyError:
        return None

def _installing(backend: str) -> str:
    """The command that adds an optional backend to this Python environment."""
    if backend != "dsh":
        return f"install {backend}, then reopen humanize"
    executable = str(Path(sys.executable).absolute())
    command = (
        f"uv pip install --python {shlex.quote(executable)} "
        "'deepseek-harness-sdk>=0.1.0rc6,<0.2'"
    )
    return f"DeepSeek Harness is not installed; run: {command}; then reopen hmz"

class Alike(Sheet[tuple[str, ...]]):
    """Which other CLIs to write one account down for as well.

    A vendor's credential is the vendor's rather than the CLI's: an Anthropic key is an
    Anthropic key whether Claude Code, pi, opencode or mimocode is holding it. So an account
    just made is often an account several other backends could be run as, and this is the
    moment to say so -- making the same key four times by hand is four places to correct when
    it is rotated.

    A form of switches rather than a list to pick from: it asks about all of them at once.
    The ones installed here start on, since those are the ones an agent could be run on
    tomorrow; the rest are listed and off, an account being worth writing down before the CLI
    that will use it is on this machine.
    """

    LETTERS: ClassVar = frozenset()

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),

        Binding("enter", "done", "done", priority=True),
        Binding("space", "flip", "turn one on or off", priority=True),
        Binding("left", "off", "off", priority=True),
        Binding("right", "on", "on", priority=True),
    ]

    def __init__(self, one: Provider, among: Sequence[str]) -> None:
        """Asks about one account.

        Args:
          one: The account that has just been made or corrected.
          among: The other backends it could be run as, in the order to show them.
        """
        super().__init__()
        self._one = one
        self._among = list(among)
        here = installed()
        self._on = {cli for cli in self._among if cli in here}

    def _ask(self) -> None:
        """Says what this is, and puts the backends up."""
        self.query_one("#asked", Label).update(
            f"{self._one.cli}/{self._one.name} runs more than {self._one.cli}"
        )
        self.query_one("#about", Label).update(
            "What this account holds is the vendor's rather than the CLI's, so these "
            "backends could each be run as it. Copying it writes the same account down for "
            "them under the same name, over one already there -- which is how a key rotated "
            "is a key rotated everywhere at once."
        )
        self._fill()
        self.query_one("#choices", OptionList).focus()

    def _fill(self) -> None:
        """Puts the backends up, each with its switch."""
        listing = self.query_one("#choices", OptionList)
        self._counting = len(str(max(len(self._among), 1)))
        at = min(listing.highlighted or 0, max(len(self._among) - 1, 0))
        here = installed()
        listing.set_options(
            Option(
                self._row(
                    seen,
                    cli,
                    "installed here" if cli in here else "not installed here yet",
                    here=seen == at,
                    inforce=False,
                    box="[x]" if cli in self._on else "[ ]",
                ),
                id=f"={cli}",
            )
            for seen, cli in enumerate(self._among)
        )
        listing.highlighted = at if self._among else None
        self._drawn = at
        self.query_one("#tuning", Label).update("")
        self.query_one("#keys", Label).update(
            "Space or the arrows turn one on and off · Enter copies it to the ones on · "
            "Esc copies it to none"
        )

    def action_flip(self) -> None:
        """Turns the one under the cursor round."""
        self._steps()

    def action_on(self) -> None:
        """Turns it on."""
        self._steps(onto=True)

    def action_off(self) -> None:
        """Turns it off."""
        self._steps(onto=False)

    def _steps(self, *, onto: bool | None = None) -> None:
        """Sets the switch under the cursor.

        Args:
          onto: What to set it to, or None to turn it round.
        """
        listing = self.query_one("#choices", OptionList)
        at = listing.highlighted
        if at is None or not 0 <= at < len(self._among):
            return
        cli = self._among[at]
        wanted = (cli not in self._on) if onto is None else onto
        if wanted:
            self._on.add(cli)
        else:
            self._on.discard(cli)
        self._fill()

    @on(OptionList.OptionSelected)
    def _took(self, event: OptionList.OptionSelected) -> None:
        """Answers with everything switched on, enter being the whole form.

        Args:
          event: What was chosen, which is unread: the row is not what this asks about.
        """
        del event
        self.action_done()

    def action_done(self) -> None:
        """Answers with the backends to copy this account to, in the order they were shown."""
        self.dismiss(tuple(cli for cli in self._among if cli in self._on))

async def also(host: App[None], one: Provider) -> tuple[str, ...]:
    """Asks which other backends to write one account down for, and writes it down for them.

    Args:
      host: The interface, which is what the sheet is pushed onto.
      one: The account.

    Returns:
      What it was copied to, and nothing at all where it could run nothing else, where the
      question was walked out of, or where every copy failed.
    """
    accounts = _hmz().accounts
    among = accounts.serves(one)
    if not among:
        return ()
    said = await host.push_screen_wait(Alike(one, among))
    copied: list[str] = []
    for cli in said or ():
        try:
            accounts.copies(one, cli)
        except (OSError, ValueError):
            continue  
        copied.append(cli)
    return tuple(copied)

class Made(NamedTuple):
    """What making an account came to.

    Attributes:
      provider: The account written down, or None where the walk was left without making one.
      status: What the way's own command exited with, or 0 for a way that runs nothing and
        for one nobody got as far as running.
      why: What went wrong before anything was written down, or "" where nothing did.
      way_runs: Whether the way had a command of its own, which is what tells an account that
        was signed in from one that was only written down.
      runs: How many models its CLI said it runs as this account, asked as soon as the account
        landed. Zero for one that was never got as far as asking, and for one whose CLI would
        not say -- which is not an account that cannot be used, only one whose models have to
        be asked for again before there are any to choose from.
      copied: The other backends this account was written down for as well, which is nothing
        for one that could run nothing else and for one nobody asked to copy.
    """

    provider: Provider | None = None
    status: int = 0
    why: str = ""
    way_runs: bool = False
    runs: int = 0
    copied: tuple[str, ...] = ()

async def made(host: App[None], cli: str, *, whose: str = "") -> Made:
    """Walks one backend's way in, and writes down the account it makes.

    Here rather than beside whatever asked for it, because both places that ask are here:
    `/providers`, which asks which backend first, and the sheet an agent's own account is
    chosen on, which knows the backend already and would otherwise have to send somebody out
    of the question they are answering to answer it.

    Args:
      host: The interface, which is what the sheets are pushed onto and what hands the
        terminal over while a login owns it.
      cli: The backend the account is for.
      whose: What to call it, for one already named, or "" to ask.

    Returns:
      What came of it: the account, whether its way in exited badly, and what stopped it
      before anything was written down. All three empty for a walk that was left.
    """
    accounts = _hmz().accounts
    way: Way | None = None
    while True:
        if way is None:
            named_way = await host.push_screen_wait(Ways(cli))
            if named_way is None:
                return Made()  
            way = accounts.way(cli, named_way)
            if way is None:
                return (
                    Made()
                )  
        signs = await host.push_screen_wait(Signing(cli, way, name=whose))
        if signs is None:
            way = None  
            continue
        break
    try:
        provider = accounts.make(cli, signs.name or whose, way, signs.answers)
    except (ValueError, OSError) as why:  
        return Made(why=str(why))
    if not way.argv:
        return Made(
            provider=provider,
            runs=await asks(cli, provider.name),

            copied=await also(host, provider),
        )

    with handed_over(host):
        status = accounts.sign_in(provider, way, signs.answers)
    return Made(
        provider=provider,
        status=status,
        way_runs=True,

        runs=0 if status else await asks(cli, provider.name),
        copied=() if status else await also(host, provider),
    )

async def asks(cli: str, name: str) -> int:
    """Asks a new account's CLI what it runs, so that there is a list when one is asked for.

    Here rather than where an account is written down: what a backend runs is that account's
    and is found by starting that backend, which is a thing to do once an account exists and
    not a thing the store of them should be doing at all.

    Off the event loop, because it is a coding agent starting up.

    Args:
      cli: The backend the account is for.
      name: What the account is called.

    Returns:
      How many models it said it runs, and zero where it would not say -- which is a list to
      ask for again rather than an account that will not work.
    """
    import asyncio

    try:
        return len(await asyncio.to_thread(_hmz().accounts.ask, cli, name))
    except Exception:  
        return 0

@contextlib.contextmanager
def handed_over(host: App[None]) -> Generator[None]:
    """Gives the terminal away for as long as something else needs to own it.

    Where there is one to give: a driver that cannot be suspended is one nobody is watching --
    a test, a web terminal -- and what was going to run still has to run.

    Args:
      host: The interface holding the terminal.
    """
    from textual.app import SuspendNotSupported

    try:
        with host.suspend():
            yield
    except SuspendNotSupported:
        yield

_SPEAKS = "\x00speaks"

class Backends(Picks):
    """Which coding agent a new account is for.

    Every backend humanize drives rather than the ones installed here: an account is
    credentials, and credentials are worth writing down before the CLI that will use them is
    on this machine. And, last, the one row here that is not an account at all: a CLI of your
    own that speaks ACP, which is what somebody who has got this far and cannot find their
    agent in the list came to say.
    """

    asked = "Select which coding agent this account is for"
    about = (
        "The CLI whose credentials these are. An account is one backend's -- what signs in "
        "to Claude Code is not what signs in to codex -- and the ways in are its own."
    )

    def rows(self) -> list[tuple[str, str, str]]:
        """Every backend there is, saying how each of them can be signed into."""
        held = _hmz()
        accounts = held.accounts
        return [
            (
                profile.name,
                profile.name,
                ", ".join(way.name for way in accounts.ways(profile.name)),
            )
            for profile in held.backends()
        ] + [
            (
                _SPEAKS,
                "a CLI of your own",
                "one that speaks ACP, written down as a backend from here on",
            )
        ]

class Ways(Picks):
    """How to sign into one backend: its subscription, a key, a gateway, somebody's cloud.

    What a backend offers rather than what could be written: each of these lands somewhere
    different -- a login writes the CLI's own store, a key is a variable -- and an account is
    one of them, answered.
    """

    asked = "Select how to sign in"
    about = (
        "What this account is. A way with a command of its own is handed the terminal once "
        "the questions are answered, so its own browser or device code owns the screen; one "
        "that is only answers is written down as they are given."
    )

    def __init__(self, backend: str) -> None:
        """Initializes the choosing.

        Args:
          backend: The CLI these are the ways into.
        """
        super().__init__()
        self._backend = backend

    def rows(self) -> list[tuple[str, str, str]]:
        """Every way that backend offers, and the one every backend has."""
        return [
            (way.name, way.name, way.about)
            for way in _hmz().accounts.ways(self._backend)
        ]

    def nothing(self) -> str:
        """Says so for a name no backend answers to, which is the only way this is empty."""
        if self._rows:
            return ""
        return f"{escape(self._backend)} is not a coding agent humanize drives"

class Signs(NamedTuple):
    """What an account is to be made out of: what to call it, and what its way was told.

    Attributes:
      name: What the account is called, which is what an agent is configured with.
      answers: What each question was answered with, by the variable that answer becomes.
    """

    name: str
    answers: dict[str, str]

_CALLED = ""

_TYPED = " "
_TYPED_ABOUT = (
    "the variables, as NAME=VALUE, one per line -- shift+enter breaks the line"
)

_BREAKS = ("shift+enter", "ctrl+j")

class Signing(Sheet[Signs]):
    """What a way in has to be told before an account can be made out of it.

    A form rather than a list, so it is drawn as the settings of a flow are: one row per
    question, the variable the answer becomes, what has been typed into it, and the question
    said quietly beside it. What the backend called a secret is drawn as bullets and never
    shown back --
    it is on its way into a credential store, and a screen is somewhere it can be read off.
    """

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),

        Binding("enter", "done", "done", priority=True),
    ]

    def __init__(
        self,
        cli: str,
        way: Way,
        name: str = "",
        held: Mapping[str, str] | None = None,
    ) -> None:
        """Initializes the answering.

        Args:
          cli: The backend this account is for.
          way: The way in it is being made by, whose questions these are.
          name: What it is called already, for one being signed in again -- a name it has is
            not a name to ask for twice -- or "" to ask for one.
          held: What that account holds now, for one being corrected rather than made. A
            secret among them is not read back on to the screen: it is on its way into a
            credential store, and a screen is somewhere it can be read off. So a corrected
            account is one whose secrets are typed again, which is what correcting one is.
        """
        super().__init__()
        self._cli = cli
        self._way = way
        self._name = name

        self._fields: list[tuple[str, str, bool]] = [
            *([] if name else [(_CALLED, "what to call this account", False)]),
            *((one.env, one.about, one.secret) for one in way.asks),

            *([] if way.asks else [(_TYPED, _TYPED_ABOUT, True)]),
        ]
        self._counting = len(str(len(self._fields)))

        asked = {one.env: one for one in way.asks}
        self._typed_in: dict[str, str] = (
            {_CALLED: name}
            | {one.env: one.fixed for one in way.asks}
            | {
                where: value
                for where, value in (held or {}).items()

                if where in asked and not asked[where].secret
            }
        )
        
        self._wrong = ""

    def _ask(self) -> None:
        """Says what is being signed into, and what the keys do while it is."""
        self.query_one("#asked", Label).update(
            f"Sign in to {escape(self._cli)} by {escape(self._way.name)}"
        )
        self.query_one("#about", Label).update(
            "What this way in has to be told. Typing answers the row under the cursor and "
            "enter takes the lot. A secret is drawn as bullets and never shown back."
        )
        self._fill()
        self.query_one("#choices", OptionList).focus()

    def _fill(self) -> None:
        """Puts the questions up, with the caret in the one under the cursor."""
        listing = self.query_one("#choices", OptionList)
        at = self._at
        listing.set_options(
            Option(
                self._line(seen, held, about, secret=secret, here=seen == at),
                id=f"={held}",
            )
            for seen, (held, about, secret) in enumerate(self._fields)
        )
        listing.highlighted = at if self._fields else None
        self._drawn = listing.highlighted
        self.query_one("#tuning", Label).update(
            f"[$error]{escape(self._wrong)}[/]" if self._wrong else ""
        )
        self.query_one("#keys", Label).update(
            "Type to answer · Backspace to rub out · Enter to accept · Esc to go back"
        )

    @property
    def _at(self) -> int:
        """Which question the cursor is on, counting from zero."""
        listing = self.query_one("#choices", OptionList)
        return min(listing.highlighted or 0, max(len(self._fields) - 1, 0))

    def _line(self, at: int, held: str, about: str, *, secret: bool, here: bool) -> str:
        """One question: what the answer becomes, what has been typed, and what is being asked.

        Args:
          at: Which one it is, counting from zero.
          held: The variable the answer is kept under, or "" for the name.
          about: The question, as the backend puts it.
          secret: Whether what is typed is a secret.
          here: Whether the cursor is on it.

        Returns:
          The row, as markup.
        """

        value = self._typed_in.get(held, "")
        shown = "•" * len(value) if secret else value
        return _written(at, self._counting, held or "name", about, shown, here=here)

    def on_key(self, event: events.Key) -> None:
        """Takes a letter as answering the question under the cursor.

        There is nothing to search here -- every question is on screen at once -- so the keys
        that narrow a list elsewhere are the ones that answer.

        Args:
          event: The key.
        """
        if not self._fields:
            return
        held = self._fields[self._at][0]
        if event.key == "backspace":
            self._typed_in[held] = self._typed_in.get(held, "")[:-1]
        elif event.key in _BREAKS and held == _TYPED:

            self._typed_in[held] = self._typed_in.get(held, "") + "\n"
        elif event.is_printable and event.character:
            self._typed_in[held] = self._typed_in.get(held, "") + event.character
        else:
            return
        event.prevent_default()
        event.stop()
        self._wrong = ""
        self._fill()

    def on_paste(self, event: events.Paste) -> None:
        """Pastes an answer into the question under the cursor."""
        if not self._fields or not event.text:
            event.stop()
            return
        held = self._fields[self._at][0]
        pasted = event.text.replace("\r\n", "\n").replace("\r", "\n")
        if held != _TYPED:

            pasted = pasted.split("\n", 1)[0]
        self._typed_in[held] = self._typed_in.get(held, "") + pasted
        event.stop()
        self._wrong = ""
        self._fill()

    def action_done(self) -> None:
        """Answers with what it is to be called and what its way was told, once that is all.

        What is missing is said where it was typed rather than raised at whoever opened the
        sheet: a question left blank is a question to answer, and this is where answering it
        happens.
        """
        accounts = _hmz().accounts
        name = (self._name or self._typed_in.get(_CALLED, "")).strip()
        answers = {
            held: value
            for held, value in self._typed_in.items()
            if held.strip() and value
        }
        try:
            accounts.where(self._cli, name)
            if said := self._typed_in.get(_TYPED, "").strip():

                answers |= accounts.env(said.replace("\r", "\n"))
        except ValueError as why:
            self._wrong = str(why)
            self._fill()
            return
        if still := accounts.asks(self._way, answers):
            self._wrong = f"{still[0]} is still to be answered"
            self._fill()
            return
        if not answers and not self._way.argv:
            self._wrong = "an account that says nothing signs nothing in"
            self._fill()
            return
        self.dismiss(Signs(name, answers))

class Popup(Picks):
    """A question that arrived rather than one somebody walked to.

    Drawn as a box in the middle of the screen rather than as a sheet: a sheet is walked to
    and fills the width it is drawn in, and this arrives over whatever was there, says one
    thing and is answered in a keypress. Each of these says for itself what it asks and what
    box it is drawn in; what is here is the one thing they all do, which is to be read rather
    than searched.
    """

    def check_action(
        self,
        action: str,
        parameters: tuple[object, ...],
    ) -> bool | None:
        """Whether one of the keys is live, which a question of two answers narrows.

        Args:
          action: What the key would do.
          parameters: What it would do it with.

        Returns:
          Whether to run it. Never the search: two rows are read rather than narrowed, and a
          box in the middle of the screen has no room to say what was typed into one.
        """
        return action != "search" and super().check_action(action, parameters)

class Confirms(Popup):
    """Whether to keep what a menu is holding, asked as it is walked out of.

    A menu applies nothing until it is left, so leaving one is the moment the changes in it
    either land or do not. Asked rather than assumed either way: what was changed took typing
    to change, and throwing it away silently is worse than one more question.

    Drawn as a box in the middle of the screen rather than as a sheet, because it is not one:
    a sheet is a question somebody walked to, and this is one that arrived over the menu they
    were walking out of. Two answers, since the third -- going back to the menu -- is what esc
    already is everywhere else, and an answer that is also a key is a row that says the key is
    not there.
    """

    CSS = _POPUP

    asked = "Save what was changed?"
    about = "Nothing in this menu has been applied yet."

    def rows(self) -> list[tuple[str, str, str]]:
        """The two things there are to do about a menu holding changes."""
        return [
            (_KEEP, "save and close", "write it down and apply it"),
            (_DROP, "discard and close", "leave everything as it was"),
        ]

    def _fill(self) -> None:
        """Puts the two answers up, and says what esc is here.

        Esc is the third answer -- back to the menu, changing nothing -- so it says so. Every
        other sheet leaves on it, and one that said `cancel` over a menu holding changes would
        read as the one thing it is not.
        """
        super()._fill()
        self.query_one("#keys", Label).update(
            "Enter to choose · Esc to go back to the menu"
        )

STOPS, DETACHES, STAYS = "stops", "detaches", "stays"

class Leaves(Popup):
    """What is to become of the flow that is running, asked as the interface is closed.

    Closing the interface is not on its own a thing to do to a run. A flow is a loop and a
    turn thinks for minutes, so a day's work is behind the same three letters that close a
    window -- and where the run is being held somewhere a terminal closing cannot reach, the
    two are genuinely different things and only the person at the prompt knows which is meant.

    Drawn as a box in the middle of the screen rather than as a sheet, for the reason the
    question about a menu holding changes is: a sheet is a question somebody walked to, and
    this is one that arrived.
    """

    CSS = f"Leaves {{ align: center middle; background: transparent; }}\n{_POPUP}"

    asked = "A flow is running here."
    about = "Closing the interface is not, on its own, a thing to do to a run."

    def __init__(self, *, held: bool) -> None:
        """Initializes the question.

        Args:
          held: Whether this run is being held somewhere that outlives this terminal, which
            is what makes leaving it running an answer there is.
        """
        super().__init__()
        self._held = held

    def rows(self) -> list[tuple[str, str, str]]:
        """The two answers, the second of which is whichever one is true here."""
        return [
            (
                STOPS,
                "stop the flow, then leave",
                "every agent takes no further turn, and the loop ends",
            ),
            (
                DETACHES,
                "leave it running, and let go of this terminal",
                "the run carries on where nothing is reading it; `hmz` opens it again",
            )
            if self._held
            else (
                STAYS,
                "stay here",
                "nothing is stopped and nothing is closed",
            ),
        ]

    def _fill(self) -> None:
        """Puts the two answers up, and says what esc is here."""
        super()._fill()
        self.query_one("#keys", Label).update("Enter to choose · Esc to stay here")

_REPORTS, _QUIET = "on", "off"

_SENTRY = "reports"
_SENT = "sent"
_WORKSPACE = "workspace"
_RUNS = "flow"
_PROFILES = "profile"
_FORGET = "forget"

class Adjusted(NamedTuple):
    """What the settings menu answers with: what to change, and what to forget.

    Attributes:
      enable_sentry: Whether humanize reports its own failures from now on, or None where
        that was not touched.
      profile: Whether a run in this directory is profiled as well as traced.
      forget: Whether to forget what this workspace was set up to run.
    """

    enable_sentry: bool | None = None
    profile: bool = False
    forget: bool = False

class Adjusts(Drafts[Adjusted]):
    """What humanize remembers: the settings that are everywhere, and this directory's.

    Two pages because they are two kinds of thing rather than two halves of one. What is on
    the first is true of this machine however many projects are driven from it -- whether
    humanize reports its own failures is the whole of it today -- and what is on the second is
    one directory's: the flow it opens on, and the agents that flow was last set up with.

    A menu rather than a file to edit, for the reason every other menu here is one: what is
    written down is written down in humanize's own words, and a person should not have to know
    the shape of a YAML file to turn a thing off. Nothing lands until it is left and saving is
    confirmed, exactly as everywhere else.
    """

    TABS: ClassVar = ("Everywhere", "This directory")

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),
        Binding("tab", "next_tab", "next page", priority=True),
        Binding("shift+tab", "prev_tab", "previous page", priority=True),
        Binding("left", "easier", "back one", priority=True),
        Binding("right", "harder", "on one", priority=True),
    ]

    def __init__(
        self,
        *,
        enable_sentry: bool | None,
        workspace: str,
        flow: str,
        agents: int,
        flows: int,
        overridden: bool = False,
        profile: bool = False,
    ) -> None:
        """Initializes the menu on what is remembered now.

        Args:
          enable_sentry: Whether humanize reports its own failures, or None while nobody has
            been asked.
          workspace: The directory this is the second page of.
          flow: The flow it was last set up to run, or "" for one it never was.
          agents: How many agents that flow was set up with here.
          flows: How many flows this directory has been set up to run.
          overridden: Whether the environment is answering the reporting question for this
            run, so that a row saying one thing while humanize does another says so.
          profile: Whether a run here is profiled as well as traced.
        """
        super().__init__()
        self._sentry = enable_sentry
        self._overridden = overridden
        self._workspace = workspace
        self._flow = flow
        self._agents = agents
        self._flows = flows
        self._profile = profile
        self._forget = False
        self._said = (
            f"{SAYS} is set, so this run does the opposite of what this says"
            if overridden
            else ""
        )

    def _ask(self) -> None:
        """Says what the menu is, and puts up the page it opened on."""
        self.query_one("#asked", Label).update("Settings")
        self._fill()
        self.query_one("#choices", OptionList).focus()

    def _rows(self) -> list[tuple[str, str, str]]:
        """The rows of whichever page is open: its id, what it says, and what it means."""
        if self._tab:
            return [
                (
                    _WORKSPACE,
                    _shortly(self._workspace),
                    "the directory these are remembered for",
                ),
                (
                    _RUNS,
                    self._flow or "nothing yet",
                    f"the flow it opens on, set up with {_many(self._agents, 'agent')}",
                ),
                (
                    _PROFILES,
                    _YES if self._profile else _NO,
                    "profile the programs a run here starts",
                ),
                (
                    _FORGET,
                    _YES if self._forget else _NO,
                    f"forget what is remembered here, across {_many(self._flows, 'flow')}",
                ),
            ]
        return [
            (
                _SENTRY,
                {True: _YES, False: _NO, None: "not answered yet"}[self._sentry],
                "report what goes wrong to humanize",
            ),
            (_SENT, "", "what a report carries, and what it never does"),
        ]

    def _fill(self) -> None:
        """Puts up whichever page is open, and the titles above it."""
        self.query_one("#about", Label).update(
            "What humanize remembers about this machine. The arrows step the row under the "
            "cursor. Nothing lands until this menu is left and saving is confirmed."
            if not self._tab
            else "What humanize remembers about this directory: the flow it opens on, and "
            "what that flow was last set up to run."
        )
        listing = self.query_one("#choices", OptionList)
        rows = self._rows()
        self._counting = len(str(len(rows)))
        at = min(listing.highlighted or 0, len(rows) - 1)
        listing.set_options(
            Option(
                self._row(
                    seen,
                    name,
                    f"{value}   {about}" if value else about,
                    here=seen == at,
                    inforce=False,
                ),
                id=f"={name}",
            )
            for seen, (name, value, about) in enumerate(rows)
        )
        listing.highlighted = at
        self._drawn = at
        self.tabbed(self._tab_line())
        self.query_one("#tuning", Label).update(
            f"[$text-muted]{self._said}[/]" if self._said else ""
        )
        self.query_one("#keys", Label).update(
            "Left and right to step this one · Enter opens what it opens · Esc to close"
        )

    def action_easier(self) -> None:
        """Steps the row under the cursor back one, which for a switch is the same as on."""
        self._step()

    def action_harder(self) -> None:
        """Steps it on one."""
        self._step()

    def _step(self) -> None:
        """Turns round whichever switch the cursor is on."""
        listing = self.query_one("#choices", OptionList)
        at = listing.highlighted or 0
        rows = self._rows()
        held = rows[at][0] if 0 <= at < len(rows) else ""
        if held == _SENTRY:
            self._sentry = not self._sentry
        elif held == _PROFILES:
            self._profile = not self._profile
        elif held == _FORGET:
            self._forget = not self._forget
        else:
            return
        self._said = ""
        self.changed()
        self._fill()

    @on(OptionList.OptionSelected)
    def _took(self, event: OptionList.OptionSelected) -> None:
        """Says what a report carries, for the one row that is a thing to read."""
        held = str(event.option.id or "").removeprefix("=")
        if held == _SENT:
            sent, kept = "; ".join(SENT), "; ".join(KEPT)
            self._said = f"Sent: {sent}. Never: {kept}."
            self._fill()
            return
        self._step()

    def applied(self) -> None:
        """Answers with what was changed, which is nothing where nothing was."""
        self.dismiss(
            Adjusted(
                enable_sentry=self._sentry,
                profile=self._profile,
                forget=self._forget,
            )
        )

_ENOUGH = 2

def _shortly(said: str) -> str:
    """One path, as much of it as a row has room for: the last parts of it."""
    parts = said.rstrip("/").split("/")
    return "/".join(parts[-_ENOUGH:]) if len(parts) > _ENOUGH else said

class Reports(Popup):
    """Whether humanize reports its own failures, asked once, on a first start.

    Asked rather than assumed either way. Assumed on, it would be a tool that started sending
    things about somebody's machine before they had heard of it; assumed off, it would be a
    tool whose crashes nobody ever sees, which on something this young is how a bug survives
    a year. So it is a question, put once, with what it means written out beside it -- what
    goes, and what does not -- and answered for every project from then on.

    Drawn as a box in the middle of the screen, for the reason the save question is: it is
    not a sheet somebody walked to. Esc leaves it unanswered, and unanswered is asked again
    next time rather than taken as a no.
    """

    CSS = f"Reports {{ align: center middle; background: transparent; }}\n{_POPUP}"

    asked = "Report what goes wrong to humanize?"

    def __init__(self) -> None:
        """Initializes the question on its default answer, which is yes."""
        super().__init__()
        sent, kept = "; ".join(SENT), "; ".join(KEPT)
        self.about = (
            f"humanize is early, and a crash nobody sees is a bug nobody fixes. "
            f"Sent: {sent}. Never sent: {kept}."
        )

    def rows(self) -> list[tuple[str, str, str]]:
        """The two answers, the one that helps first."""
        return [
            (
                _REPORTS,
                "yes, report them",
                "what broke, and what was running when it did",
            ),
            (_QUIET, "no, send nothing", "nothing about this machine leaves it"),
        ]

    def _fill(self) -> None:
        """Puts the two answers up, and says what esc is here."""
        super()._fill()
        self.query_one("#keys", Label).update(
            "Enter to choose · Esc to be asked again next time · /settings changes it later"
        )

class Fitted(NamedTuple):
    """One agent as a sheet answered with it: what it is, and what it is called.

    Attributes:
      runs: The agent itself.
      name: What it is saved under, for one being edited in the agents menu, and "" for one
        of a flow's -- an agent of a flow is called what the flow calls it, which is not
        something anybody here may rename.
    """

    runs: Runs
    name: str = ""

_ASPECT = 12
_HOW = 34

_YES, _NO = "on", "off"

_LOCAL = "as local"

_IMPORT = "import"
_NAME = "name"
_CLI = "cli"
_ACCOUNT = "provider"
_MODEL = "model"
_EFFORT = "effort"
_SWARM = "swarm"
_SKILLS = "skills"
_PERMIT = "permission"
_GOALS = "goals"
_SEARCHES = "web search"
_WHERE = "where"
_SAVE_AS = "save as"

_STEPPED = (_EFFORT, _SWARM, _PERMIT, _GOALS, _SEARCHES)

class Agent(Drafts[Fitted]):
    """Everything one agent is, on one sheet, each row opened or stepped where it stands.

    Which is the walk of three sheets that used to ask it, folded into the thing it was asking
    about. An agent is not three questions -- it is one thing with a CLI, an account, a model
    at an effort, a set of skills, a rung of what it may do and a machine its work lands on --
    and asking it as a walk meant that changing the effort of an agent already set up was four
    keypresses through two sheets that had nothing to say.

    The order the rows go in is still the order of what depends on what: the CLI settles which
    accounts there are to choose from and which models that CLI will name, and the account
    settles which of them it may name. Changing the CLI therefore lets go of the model, which
    belonged to the CLI before it.

    A saved agent can be copied in at the top and saved as a reusable copy at the bottom. What
    is imported is a copy: an agent tuned inside a flow is that flow's, and writing the changes
    back into the thing it was copied from would change every other flow that had imported it.
    """

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),

        Binding("left", "easier", "back one", priority=True),
        Binding("right", "harder", "on one", priority=True),
    ]

    def __init__(
        self,
        named: str,
        runs: Runs,
        agents: dict[str, tuple[Model, ...]],
        *,
        place: Place | None = None,
        unavailable: frozenset[str] = frozenset(),
        name: str = "",
        naming: bool = False,
    ) -> None:
        """Initializes the sheet on what the agent is now.

        Args:
          named: What to call the agent being set up, which the question at the top says.
          runs: What it is now, which every row reads back.
          agents: The backends offered here, and what each of them says it runs.
          place: What the flow declared about this one, or None for a saved agent -- which
            belongs to no flow and so is asked every question there is.
          unavailable: The optional backends that still need installing.
          name: What it is saved under, for one being edited in the agents menu.
          naming: Whether it has a name of its own to be typed, which a flow's agent has not.
        """
        super().__init__()
        self._named = named
        self._agents = dict(agents)
        self._unavailable = unavailable
        self._place = place
        self._called = name
        self._is_named = naming
        cli, _, rest = runs.spec.partition("/")
        model, _, effort = rest.rpartition(":")

        self._cli: str = cli
        self._model: str = model

        self._swarm: bool = effort.startswith(SWARM)
        self._effort: str = effort.removeprefix(SWARM)
        self._permission = (
            PERMISSIONS.index(runs.permission)
            if runs.permission in PERMISSIONS
            else len(PERMISSIONS) - 1
        )
        self._provider: str = runs.provider
        self._goals = True if place is not None and place.goal else runs.goals
        self._searches = runs.web_search
        self._anchor = runs.anchor

        self._catalogue: tuple[Model, ...] | None = None
        self._read_for: tuple[str, str] = ("", "")

        self._said = ""

    def _ask(self) -> None:
        """Says whose agent this is, and what setting it up settles."""
        self.query_one("#asked", Label).update(f"Set up {escape(self._named)}")
        saving = (
            "Save accepts this setup; save as keeps a reusable copy."
            if self._place is not None
            else "Save accepts this setup."
        )
        self.query_one("#about", Label).update(
            "What this one agent is. Enter opens the row under the cursor, and the arrows "
            f"step the ones that are a rung rather than a list. {saving}"
        )
        self._fill()
        self.query_one("#choices", OptionList).focus()

    def _rows(self) -> list[tuple[str, str, str]]:
        """Every row this agent is made of: its id, what it is now, and what it means.

        Returns:
          One `(id, what it is set to, the line about it)` apiece, in the order they are
          asked. A row nobody is being asked about is not among them: a flow that settled
          where its agent works has not left that question open.
        """
        rows: list[tuple[str, str, str]] = []
        if self._is_named:
            rows.append((_NAME, self._called, "what this agent is saved under"))
        if self._place is not None:
            rows.append((_IMPORT, "", "copy a saved agent into this one"))
        rows.extend(
            [
                (_CLI, self._cli or "—", "which coding agent takes its turns"),
                (_ACCOUNT, self._provider or _LOCAL, "the account those turns run as"),
                (_MODEL, self._model or "—", "which of that CLI's models it runs"),
                (_EFFORT, self._effort or "—", "how hard it thinks"),
            ]
        )
        if self._swarms():
            rows.append(
                (_SWARM, _YES if self._swarm else _NO, "one turn run as a fleet")
            )
        rows.extend(
            [
                (
                    _SKILLS,
                    "as its CLI finds them",
                    "what it will be carrying, which its CLI keeps",
                ),
                (
                    _PERMIT,
                    PERMISSIONS[self._permission],
                    "what it may do without being asked",
                ),
                (
                    _GOALS,
                    _YES if self._goals else _NO,
                    "required by the flow"
                    if self._place is not None and self._place.goal
                    else "whether the backend's own goals are available",
                ),
            ]
        )

        if self._tellable():
            rows.append(
                (
                    _SEARCHES,
                    _YES if self._searches else _NO,
                    "whether it may search the web",
                )
            )
        if self._place is None or pointed(self._place):
            rows.append(
                (
                    _WHERE,
                    self._anchor or "this machine",
                    "the machine its work lands on",
                )
            )
        elif image := _settled(self._place):
            rows.append((_WHERE, f"in a container of {image}", "the flow settled this"))
        rows.append((_SAVE, "", "accept this agent setup"))
        if self._place is not None:
            rows.append((_SAVE_AS, "", "save a reusable agent you can import"))
        return rows

    def _fill(self) -> None:
        """Puts the rows up, with the marker beside the one the cursor is on."""
        listing = self.query_one("#choices", OptionList)
        rows = self._rows()
        self._counting = len(str(max(len(rows), 1)))
        at = min(listing.highlighted or 0, max(len(rows) - 1, 0))
        listing.set_options(
            Option(
                self._line(seen, held, value, about, here=seen == at),
                id=f"={held}",
            )
            for seen, (held, value, about) in enumerate(rows)
        )
        listing.highlighted = at if rows else None
        self._drawn = at
        self.query_one("#tuning", Label).update(
            f"[$text-muted]{self._said}[/]" if self._said else ""
        )
        held = rows[at][0] if rows else ""
        self.query_one("#keys", Label).update(
            "←/→ to change · Esc to close"
            if held in _STEPPED
            else "Type to name it · Esc to close"
            if held == _NAME
            else "Enter to save · Esc to close"
            if held == _SAVE
            else "Enter to save a copy · Esc to close"
            if held == _SAVE_AS
            else "Enter to open · Esc to close"
        )

    def _line(self, at: int, held: str, value: str, about: str, *, here: bool) -> str:
        """One row: what is being said, what it is set to, and what it means.

        Args:
          at: Which one it is, counting from zero.
          held: What the row is called.
          value: What it is set to.
          about: The line about it, said quietly.
          here: Whether the cursor is on it.

        Returns:
          The row, as markup.
        """
        mark = f"{_INDENT}[$primary]{_HERE}[/] " if here else f"{_INDENT}  "
        number = f"{at + 1:>{self._counting}}."

        caret = "[reverse] [/reverse]" if here and held == _NAME else ""
        
        opens = "" if held in _STEPPED or held in (_NAME, _SAVE) else " ▸"
        
        named = escape(held) + " " * max(1, _ASPECT - len(held))
        room = _HOW - len(value) - len(opens) - (1 if caret else 0)
        return (
            f"{mark}[$text-muted]{number}[/] {named}"
            f"[$secondary]{escape(value)}[/]{caret}[$text-muted]{opens}[/]"
            f"{' ' * max(1, room)}[$text-muted]{escape(about)}[/]"
        )

    def _models(self) -> tuple[Model, ...]:
        """What the chosen CLI says it runs as the chosen account, read once per pair."""
        if self._catalogue is None or self._read_for != (self._cli, self._provider):
            self._read_for = (self._cli, self._provider)
            self._catalogue = (
                _hmz().accounts.models(self._cli, self._provider)
                if self._provider
                else self._agents.get(self._cli, ())
            )
        return self._catalogue

    def _under_model(self) -> Model | None:
        """The model this agent runs, as the CLI described it, or None where it named none."""
        return next(
            (one for one in self._models() if one.name == self._model),
            None,
        )

    def _efforts(self) -> tuple[str, ...]:
        """What the chosen model takes, hardest first.

        Returns:
          The efforts, or the one this agent is already at for a model the CLI has not
          described -- an agent read back off a file names a model whose catalogue may not
          have been fetched yet, and its effort is still the effort it runs at. A model whose
          own name carries its effort -- Antigravity lists `gemini-3.7-flash-low` -- says so
          by offering that one and no other.
        """
        model = self._under_model()
        if model is not None and model.efforts:
            return model.efforts
        return (self._effort,) if self._effort else ()

    def _swarms(self) -> bool:
        """Whether the chosen model runs a turn as a fleet as well as as an agent."""
        model = self._under_model()
        return model is not None and model.swarms

    def _tellable(self) -> bool:
        """Whether the chosen CLI can be told whether its agents may search the web."""
        from hmz.backends import named

        profile = named(self._cli) if self._cli else None
        return profile is not None and profile.searches

    def _made(self) -> Runs:
        """This agent as it now stands, which is what the sheet answers with."""

        wide = SWARM if self._swarm and self._swarms() else ""
        return Runs(
            spec=f"{self._cli}/{self._model}:{wide}{self._effort}",
            anchor=self._anchor,

            permission=(
                PERMISSIONS[self._permission]
                if self._permission < len(PERMISSIONS) - 1
                else ""
            ),
            provider=self._provider,
            goals=self._goals,

            web_search=self._searches or not self._tellable(),
        )

    def applied(self) -> None:
        """Answers with the agent as it now stands, and what it is called."""
        self.dismiss(Fitted(self._made(), self._called))

    @property
    def _held(self) -> str:
        """Which row the cursor is on, by id."""
        return self.under()

    def on_key(self, event: events.Key) -> None:
        """Takes a letter as writing the name, which is the one row that is written.

        There is nothing to search here -- every row is on the screen at once -- so the keys
        that narrow a list elsewhere are the ones that name this agent.

        Args:
          event: The key.
        """
        if self._held != _NAME:
            return
        if event.key == "backspace":
            self._called = self._called[:-1]
        elif event.is_printable and event.character:
            self._called += event.character
        else:
            return
        event.prevent_default()
        event.stop()
        self.changed()
        self._fill()

    def action_harder(self) -> None:
        """Steps the row under the cursor one on, where it is one that is stepped."""
        self._step(-1)

    def action_easier(self) -> None:
        """Steps it one back."""
        self._step(1)

    def _step(self, by: int) -> None:
        """Moves whichever rung the cursor is on, however that one moves.

        Args:
          by: One step along the efforts towards the one that thinks least, which is the same
            direction as one step back through everything else.
        """
        held = self._held
        if held == _EFFORT:
            efforts = self._efforts()
            if not efforts:
                return
            at = efforts.index(self._effort) if self._effort in efforts else 0
            self._effort = efforts[min(max(at + by, 0), len(efforts) - 1)]
        elif held == _SWARM:
            self._swarm = not self._swarm
        elif held == _PERMIT:

            self._permission = (self._permission - by) % len(PERMISSIONS)
        elif held == _GOALS:
            if self._place is not None and self._place.goal:
                return  
            self._goals = not self._goals
        elif held == _SEARCHES:
            self._searches = not self._searches
        else:
            return
        self.changed()
        self._said = ""
        self._fill()

    @on(OptionList.OptionSelected)
    def _took(self, event: OptionList.OptionSelected) -> None:
        """Opens the row under the cursor, or steps it where it is one that is stepped.

        Args:
          event: What was chosen.
        """
        held = str(event.option.id or "").removeprefix("=")
        if held in _STEPPED:
            self._step(-1)
            return
        if held == _SAVE:
            self.applied()
            return
        if held in (_CLI, _ACCOUNT, _MODEL, _SKILLS, _WHERE, _IMPORT, _SAVE_AS):
            self._opens(held)

    @work
    async def _opens(self, held: str) -> None:
        """Asks whatever that row is a way of asking, and holds the answer.

        Args:
          held: The row, by id.
        """
        showing = cast(
            "App[None]",
            self.app,  
        )
        if held == _CLI:
            await self._chose_cli(showing)
        elif held == _ACCOUNT:
            await self._chose_account(showing)
        elif held == _MODEL:
            await self._chose_model(showing)
        elif held == _SKILLS:
            await self._chose_skills(showing)
        elif held == _WHERE:
            await self._chose_where(showing)
        elif held == _IMPORT:
            await self._imports(showing)
        elif held == _SAVE_AS:
            await self._saves_as(showing)
        self._fill()

    async def _chose_cli(self, showing: App[None]) -> None:
        """Asks which coding agent takes this one's turns, and lets go of what was its."""
        chosen = await showing.push_screen_wait(
            Clis(
                self._agents,
                self._cli,
                place=self._place,
                unavailable=self._unavailable,
            )
        )
        if chosen is None or chosen == self._cli:
            return

        self._cli, self._provider, self._model, self._effort = chosen, "", "", ""
        self._swarm = False
        self._said = ""
        self.changed()

    async def _chose_account(self, showing: App[None]) -> None:
        """Asks which account its turns run as, out of that CLI's own."""
        if not self._cli:
            self._said = "choose the coding agent first; the accounts are its own"
            return
        chosen = await showing.push_screen_wait(Accounts(self._cli, self._provider))
        if chosen is None or chosen == self._provider:
            return
        
        self._provider, self._said = chosen, ""
        self.changed()

    async def _chose_model(self, showing: App[None]) -> None:
        """Asks which of that CLI's models it runs, and starts it at the hardest effort."""
        if not self._cli:
            self._said = "choose the coding agent first; a model belongs to the CLI"
            return
        chosen = await showing.push_screen_wait(
            Catalogue(self._cli, self._provider, self._models(), self._model)
        )
        if chosen is None:
            return
        self._model, self._said = chosen, ""
        self._catalogue, self._read_for = None, ("", "")
        efforts = self._efforts()
        if self._effort not in efforts:

            self._effort = efforts[0] if efforts else ""
        self.changed()

    async def _chose_skills(self, showing: App[None]) -> None:
        """Shows what its CLI would load, which is the CLI's own and is not changed here."""
        if not self._cli:
            self._said = "choose the coding agent first; the skills are its own"
            return
        await showing.push_screen_wait(Skills(self._cli))

    async def _chose_where(self, showing: App[None]) -> None:
        """Asks which machine its work lands on, where that is a question anybody is asked."""
        if self._place is not None and not pointed(self._place):
            self._said = "the flow settled where this one works"
            return
        where = await showing.push_screen_wait(Anchors(self._named, self._anchor))
        if where is None:
            return
        self._anchor, self._said = where, ""
        self.changed()

    async def _imports(self, showing: App[None]) -> None:
        """Copies a saved agent into this one, name and all but the name."""
        held = _hmz().agents.all()
        if not held:
            self._said = "no agents have been saved yet; /agents saves one"
            return
        chosen = await showing.push_screen_wait(Imports(held))
        if chosen is None:
            return
        one = next((each for each in held if each.name == chosen), None)
        if one is None:
            return
        cli, _, rest = one.runs.spec.partition("/")
        model, _, effort = rest.rpartition(":")
        self._cli, self._model = cli, model
        self._swarm = effort.startswith(SWARM)
        self._effort = effort.removeprefix(SWARM)
        self._provider = one.runs.provider
        self._permission = (
            PERMISSIONS.index(one.runs.permission)
            if one.runs.permission in PERMISSIONS
            else len(PERMISSIONS) - 1
        )
        self._anchor = one.runs.anchor
        
        if self._place is None or not self._place.goal:
            self._goals = one.runs.goals
        self._searches = one.runs.web_search
        self._catalogue, self._read_for = None, ("", "")
        self._said = (
            f"copied from {escape(chosen)}; changing it here changes only this one"
        )
        self.changed()

    async def _saves_as(self, showing: App[None]) -> None:
        """Writes this agent down under a name, new or one already there."""
        agents = _hmz().agents
        if not (self._cli and self._model):
            self._said = "an agent with no model is not one to save"
            return
        name = await showing.push_screen_wait(Names(agents.all(), self._named))
        if not name:
            return

        agents.write(name, self._made())
        self._said = f"saved as {escape(name)}"

class Clis(Picks):
    """Which coding agent takes one agent's turns, out of the ones that could.

    Not always all of them: a flow that hangs a hook on a moment only some backends run said
    so where it declared the place, and a CLI that does not run that moment is one choosing
    would make the flow refuse to start.
    """

    asked = "Select which coding agent takes its turns"
    about = (
        "The CLI behind this agent. Its accounts, its models, its skills and how hard it can "
        "be asked to think are all its own, so choosing another lets go of them."
    )

    def __init__(
        self,
        agents: dict[str, tuple[Model, ...]],
        current: str = "",
        *,
        place: Place | None = None,
        unavailable: frozenset[str] = frozenset(),
    ) -> None:
        """Initializes the choosing.

        Args:
          agents: The backends offered here, and what each of them says it runs.
          current: The one it is now.
          place: What the flow declared about this agent, or None for a saved agent, which
            belongs to no flow and so is refused nothing.
          unavailable: The optional backends that still need installing.
        """
        super().__init__(current)
        self._agents = dict(agents)
        self._place = place
        self._unavailable = unavailable

    def rows(self) -> list[tuple[str, str, str]]:
        """Every CLI that could take this one's turns, and what each of them runs."""
        needs: frozenset[Moment] = (
            self._place.moments if self._place is not None else frozenset()
        )
        pursuing = self._place is not None and self._place.goal
        listed: list[tuple[str, str, str]] = []
        for backend in sorted(self._agents):
            drives = _drives(backend)
            if drives is None or not needs <= drives.moments:
                continue
            if pursuing and not drives.pursues:
                continue
            listed.append(
                (
                    backend,
                    backend,
                    _installing(backend)
                    if backend in self._unavailable
                    else f"{len(self._agents[backend])} models"
                    if self._agents[backend]
                    else "has not said what it runs yet",
                )
            )
        return listed

    def nothing(self) -> str:
        """Says so where the flow has ruled every backend here out, which is worth knowing."""
        return (
            ""
            if self._rows
            else "no coding agent installed here can take this one's turns"
        )

class Accounts(Picks):
    """Which account one agent's turns run as, out of one CLI's own.

    The machine's own is always the first of them: an agent nobody has been asked about runs
    as whoever signed the CLI in, and that is a row rather than a blank. Making one is a key
    here, this being the moment somebody finds out they have none for this CLI.
    """

    asked = "Select the account its turns run as"
    about = (
        "An account is one backend's -- what signs in to Claude Code is not what signs in to "
        "codex -- so these are that CLI's own. Its sessions, its settings and its skills are "
        "the CLI's whichever account it runs as."
    )
    keys = "a to make one · "
    LETTERS: ClassVar = frozenset({"search", "new"})

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),
        Binding("s", "search", "search", priority=True),
        Binding("a", "new", "make one", priority=True),
    ]

    def __init__(self, backend: str, current: str = "") -> None:
        """Initializes the choosing.

        Args:
          backend: The CLI whose accounts these are.
          current: The account it runs as now, or "" for the machine's own.
        """
        super().__init__(current)
        self._backend = backend
        self._said = ""

    def rows(self) -> list[tuple[str, str, str]]:
        """The machine's own first, and then every account that CLI has here."""
        found = _hmz().accounts.all(self._backend)
        if self._backend == "dsh":
            found = [
                one
                for one in found
                if one.way == "key" and one.env.get("DEEPSEEK_API_KEY", "").strip()
            ]
        return [
            (
                "",
                _LOCAL,
                "using credentials and the base URL saved by dsh, or this environment"
                if self._backend == "dsh"
                else "signed in as you signed it in",
            ),
            *((one.name, one.name, _sets(one)) for one in found),
        ]

    def nothing(self) -> str:
        """Says what came of making one, or where they come from for a CLI that has none."""
        if self._said:
            return self._said
        if self._backend == "dsh" and len(self._rows or []) < 2:  
            return (
                "DeepSeek Harness needs an API key; a stores one, or set DEEPSEEK_API_KEY "
                "and reopen hmz"
            )
        if len(self._rows or []) > 1:
            return ""
        return f"{escape(self._backend)} has no accounts here yet; a makes one"

    @work
    async def action_new(self) -> None:
        """Makes an account for this CLI without leaving the question it is chosen in.

        The same walk `/providers` runs, minus the question it has already answered: which
        backend. What comes of it is what this list is now showing, so a new account is chosen
        straight away -- making one here is choosing it -- unless its own way in failed, which
        is said under the list and left for whoever is looking to decide about.
        """
        showing = cast(
            "App[None]",
            self.app,  
        )
        outcome = await made(showing, self._backend)
        if outcome.why:
            self._said = escape(outcome.why)
        if outcome.provider is None:
            self._rows = None  
            self._fill()
            return
        if outcome.status:
            self._said = (
                f"{escape(outcome.provider.name)} is written down, but signing it in "
                f"exited {outcome.status}"
            )
            self._rows = None
            self._fill()
            return
        self.dismiss(outcome.provider.name)

class Catalogue(Picks):
    """Which model one agent runs, out of what its CLI last said it runs as its account.

    The rows are what that CLI said rather than a list written down anywhere: a CLI ships a
    model without asking anybody, and which of them a turn may name is the account's. `r` asks
    it again, which is what somebody who came here for a model that is not in the list wants
    -- and is the whole reason the key is on this sheet rather than somewhere else.
    """

    LETTERS: ClassVar = frozenset({"search", "refresh"})
    keys = "r to ask it again · "

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),
        Binding("s", "search", "search", priority=True),
        Binding("r", "refresh", "ask it what it runs", priority=True),
    ]

    def __init__(
        self,
        backend: str,
        provider: str,
        models: tuple[Model, ...],
        current: str = "",
    ) -> None:
        """Initializes the choosing.

        Args:
          backend: The CLI whose models these are.
          provider: The account it was asked as, or "" for the machine's own.
          models: What it last said it runs as that account, which is nothing at all for one
            that has never been asked -- and is what `r` here fills.
          current: The model it runs now.
        """
        super().__init__(current)
        self.asked = f"Select what {backend} runs"
        self.about = (
            f"Which model of {backend} takes this one's turns, and how hard it may be asked "
            "to think. These are what it last said it runs as this account; r asks it again."
        )
        self._backend = backend
        self._provider = provider
        self._models = models
        self._asking = False
        self._said = ""

    def rows(self) -> list[tuple[str, str, str]]:
        """Every model that CLI named, and what efforts each of them takes."""
        return [
            (
                one.name,
                one.name,
                ", ".join(one.efforts) + (f"{_DOT}swarms" if one.swarms else ""),
            )
            for one in self._models
        ]

    def nothing(self) -> str:
        """What to say where there is no model to say anything else about."""
        if self._asking:
            return f"asking {escape(self._backend)} what it runs…"
        if self._said:
            return self._said
        if self._models:
            return ""  
        whose = f" as {escape(self._provider)}" if self._provider else ""
        return (
            f"{escape(self._backend)} has not said what it runs{whose} yet; r asks it"
        )

    @work
    async def action_refresh(self) -> None:
        """Asks this CLI what it runs as this account, and puts up what it answers.

        Off the event loop, because asking means starting a coding agent and some of them take
        the better part of a minute over it: an interface that stopped redrawing while it ran
        would be one that looked as though it had gone away.
        """
        import asyncio

        if not self._backend or self._asking:
            return
        self._asking, self._said = True, ""
        self._fill()
        try:
            found = await asyncio.to_thread(
                _hmz().accounts.ask, self._backend, self._provider
            )
        except Exception as why:  

            self._said = escape(str(why) or type(why).__name__)
            self._asking = False
            self._fill()
            return
        self._asking, self._models = False, found
        self._said = "" if found else f"{escape(self._backend)} named no models it runs"
        self._rows = None
        self.query_one("#choices", OptionList).highlighted = 0
        self._drawn = 0
        self._fill()

class Imports(Picks):
    """Which saved agent to copy into the one being set up.

    A copy rather than a link: an agent tuned inside a flow is that flow's, and writing the
    changes back into the thing it was copied from would change every other flow that had
    imported it.
    """

    asked = "Select an agent to copy in"
    about = (
        "The agents saved under a name, which /agents keeps. What is copied is everything "
        "the agent is; changing it afterwards changes this one alone."
    )

    def __init__(self, held: Sequence[Kept]) -> None:
        """Initializes the choosing.

        Args:
          held: The agents written down, in the order they are kept in.
        """
        super().__init__()
        self._held = list(held)

    def rows(self) -> list[tuple[str, str, str]]:
        """Every agent written down, and what each of them is."""
        return [(one.name, one.name, reads((), [one.runs])[0]) for one in self._held]

class Names(Sheet[str]):
    """What to save an agent as: a name already there to write over, or one typed.

    Listed rather than typed where there is one to list, because writing over the agent
    somebody meant is the common half of this: a name typed a second time with a letter
    different is a second agent nobody wanted.
    """

    LETTERS: ClassVar = frozenset({"search"})

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),
        Binding("s", "search", "search", priority=True),
    ]

    def __init__(self, held: Sequence[Kept], suggested: str = "") -> None:
        """Initializes the naming.

        Args:
          held: The agents written down already, any of which may be written over.
          suggested: What to offer as a name for a new one, which is what the agent being
            saved is called where it is called anything.
        """
        super().__init__()
        self._held = list(held)
        self._suggested = suggested

    def _ask(self) -> None:
        """Says what saving one does, and puts the names up."""
        self.query_one("#asked", Label).update("Save this agent as")
        self.query_one("#about", Label).update(
            "The name it is imported by. Choosing one already here writes over it; s and "
            "then a name of your own saves it as a new one."
        )
        self._fill()
        self.query_one("#choices", OptionList).focus()

    def _fill(self) -> None:
        """Puts the names up, with whatever has been typed among them as a new one."""
        listing = self.query_one("#choices", OptionList)
        rows = [(one.name, one.name, reads((), [one.runs])[0]) for one in self._held]
        shown = [row for row in rows if self.fits(row[1], row[2])]
        wanted = self._typed.strip() or (
            "" if shown or self._typed else self._suggested
        )
        if wanted and all(row[0] != wanted for row in shown):
            shown.append((wanted, wanted, "a new one under this name"))
        self._counting = len(str(max(len(shown), 1)))
        at = min(listing.highlighted or 0, max(len(shown) - 1, 0))
        listing.set_options(
            Option(
                self._row(seen, label, about, here=seen == at, inforce=False),
                id=f"={answer}",
            )
            for seen, (answer, label, about) in enumerate(shown)
        )
        listing.highlighted = at if shown else None
        self._drawn = at
        self.query_one("#keys", Label).update(
            f"Enter to save · Esc to go back{self.searching()}"
        )

    @on(OptionList.OptionSelected)
    def _took(self, event: OptionList.OptionSelected) -> None:
        """Answers with the name that was picked.

        Args:
          event: What was chosen.
        """
        self.dismiss(str(event.option.id).removeprefix("="))

_GOES, _TAKEN_AGAIN = "goes", "tried"

class Failing(Picks):
    """What to say about one place: where its turns go, and how often they are taken again.

    Its own menu rather than a letter apiece on the list of steps. They are two questions
    about the place under the cursor, and enter -- which every list already means -- is what
    opens them.
    """

    def __init__(self, place: str, step: Step) -> None:
        """Asks about one place.

        Args:
          place: The place, as a step names it.
          step: What is written against it now, which is what the rows say.
        """
        super().__init__()
        self._step = step
        self.asked = place
        self.about = (
            "What happens when a turn at this place cannot be taken. It is taken again as "
            "many times as this says, and then at whatever it falls back to -- in a session "
            "of its own, no backend taking another backend's session id."
        )

    def rows(self) -> list[tuple[str, str, str]]:
        """The two, each saying what it is now."""
        tries = (
            f"{self._step.tries} more tries, {self._step.policy}"
            if self._step.tries
            else "once: a failed turn is a failed turn"
        )
        return [
            (
                _GOES,
                "falls back to",
                self._step.to or "nowhere: a failed turn is a failed turn",
            ),
            (_TAKEN_AGAIN, "taken again", tries),
        ]

class Fallbacks(Drafts[list[str]]):
    """Where a turn goes when the place taking it cannot take it at all.

    A place is three things and no more: the CLI, the account it runs as, and the model it
    runs. That is what a turn can fail for having named -- a model retired, a CLI that will
    not start, a region gone dark, a rate limit on the whole account rather than one request
    -- and it is what a step is written between. How hard the agent thinks and what it may
    reach for are what that agent *is*, settled where it was made, and they come across the
    step unchanged.

    A row also says how many times over a failed turn is taken again before the step happens.
    Both are answers to the one thing that went wrong, so both are here.

    An account falling back to another account of the same CLI is not this. That happens
    inside the conversation that was running, so it is a thing about the account, and it is
    said in `/providers` where the accounts are.
    """

    TABS: ClassVar = ("Fallback",)
    LETTERS: ClassVar = frozenset({"search", "adding", "drop"})

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),
        Binding("s", "search", "search", priority=True),
        Binding("a", "adding", "add one", priority=True),
        Binding("d", "drop", "take one away", priority=True),
    ]

    def __init__(self, agents: dict[str, tuple[Model, ...]]) -> None:
        """Reads every step written down.

        Args:
          agents: The backends offered here, and what each of them says it runs, which is
            what choosing a place is offered out of.
        """
        super().__init__()
        self._agents = dict(agents)
        
        self._steps: list[Step] = list(_hmz().fallbacks.all())
        
        self._was = self._steps[0].spec if self._steps else ""
        self._said = ""

    def _ask(self) -> None:
        """Says what these are, and puts them up."""
        self.query_one("#asked", Label).update("Fallback")
        self.query_one("#about", Label).update(
            "Where a turn goes when the place taking it cannot take it at all. A place is a "
            "CLI, an account and a model; the turn is taken again there as many times as "
            "this says, and then in a session of its own at the place it falls back to."
        )
        self._fill()
        self.query_one("#choices", OptionList).focus()

    def _rows(self) -> list[tuple[str, str, str]]:
        """Every row: its id, the place that fails, and what happens when it does.

        Returns:
          One `(id, what fails, what happens)` apiece.
        """
        return [(one.spec, one.spec, _falling(one)) for one in self._steps]

    def _fill(self) -> None:
        """Puts the steps up, each saying what fails and what happens when it does."""
        listing = self.query_one("#choices", OptionList)
        self._follows(listing)
        rows = [row for row in self._rows() if self.fits(row[1], row[2])]
        self._counting = len(str(max(len(rows), 1)))
        if all(row[0] != self._was for row in rows):
            self._was = rows[0][0] if rows else ""
        at = next((seen for seen, row in enumerate(rows) if row[0] == self._was), 0)
        listing.set_options(
            Option(
                self._row(seen, said, goes, here=seen == at, inforce=True),
                id=f"={named}",
            )
            for seen, (named, said, goes) in enumerate(rows)
        )
        listing.highlighted = at if rows else None
        self._drawn = listing.highlighted
        self.tabbed(self._tab_line())
        self.query_one("#tuning", Label).update(
            f"[$text-muted]{self._said or self._nothing(rows)}[/]"
        )
        self.query_one("#keys", Label).update(
            "Enter for what happens · a adds one · d twice takes one away · "
            f"Esc to close{self.searching()}"
        )

    @staticmethod
    def _nothing(rows: Sequence[object]) -> str:
        """What to say under a page with nothing on it, which is where to start."""
        return "" if rows else "nothing falls back anywhere yet; a says one does"

    def _follows(self, listing: OptionList) -> None:
        """Takes which row the cursor is on off the list, by its id.

        Args:
          listing: The list.
        """
        at = listing.highlighted
        if at is not None and 0 <= at < listing.option_count:
            named = str(listing.get_option_at_index(at).id or "").removeprefix("=")
            if named:
                self._was = named

    def action_adding(self) -> None:
        """Says that one more place falls back to another, which is two places to choose."""
        self._adds()

    def action_drop(self) -> None:
        """Takes the step under the cursor away, once d has been pressed twice."""
        named = self.under()
        if not named:
            return
        if not self._armed(named):
            self._said = f"press d again to take {escape(named)} away"
            self._fill()
            return
        self._steps = [one for one in self._steps if one.spec != named]
        self._said = f"{escape(named)} falls back nowhere when this menu is saved"
        self.changed()
        self._fill()

    @on(OptionList.OptionSelected)
    def _took(self, event: OptionList.OptionSelected) -> None:
        """Asks what happens when the place under the cursor cannot take a turn.

        Args:
          event: What was chosen.
        """
        named = str(event.option.id or "").removeprefix("=")
        if named:
            self._doing(named)

    @work
    async def _adds(self) -> None:
        """Chooses the place that cannot run, and then the place that takes its turns."""
        said = await self._chosen("The place that cannot run")
        if not said:
            return
        at = await self._chosen(f"What takes {said}'s turns")
        if not at:
            return
        self._writes(said, at)

    @work
    async def _doing(self, said: str) -> None:
        """Asks what to do about one step: where it goes, or how often it is taken again.

        Args:
          said: The place, as it is written down.
        """
        showing = cast(
            "App[None]",
            self.app,  
        )
        step = self._step(said)
        chosen = await showing.push_screen_wait(Failing(said, step))
        if chosen is None:
            return  
        if chosen == _GOES:
            at = await self._chosen(f"What takes {said}'s turns")
            if at:
                self._writes(said, at)
            return
        tried = await showing.push_screen_wait(
            Retries(said, step.tries, step.policy, step.timeout)
        )
        if tried is not None:
            self._tries(said, *tried)

    def _step(self, said: str) -> Step:
        """The step written against one place, or an empty one for a place with none."""
        from hmz.fallbacks import Falls

        return next((one for one in self._steps if one.spec == said), Falls(said))

    def _writes(self, said: str, at: str) -> None:
        """Holds where one place goes until the menu is saved, refusing one pointing at itself.

        Args:
          said: The place that cannot run.
          at: The place that takes its turns.
        """
        from dataclasses import replace

        if said == at:
            self._said = "a place cannot fall back to itself"
            self._fill()
            return
        self._held(replace(self._step(said), to=at))

    def _tries(self, said: str, tries: int, policy: str, timeout: float) -> None:
        """Holds how often a failed turn at one place is taken again, until the menu is saved.

        Args:
          said: The place.
          tries: How many goes beyond the first.
          policy: How long to wait between them.
          timeout: The longest the trying again may go on for, or 0.0 for no limit.
        """
        from dataclasses import replace

        self._held(
            replace(self._step(said), tries=tries, policy=policy, timeout=timeout)
        )

    def _held(self, step: Step) -> None:
        """Puts one step in place of whatever was held against that place.

        Args:
          step: The step as it now stands.
        """
        self._steps = [one for one in self._steps if one.spec != step.spec] + [step]
        self._was, self._said = step.spec, ""
        self.changed()
        self._fill()

    async def _chosen(self, asked: str) -> str:
        """Walks the three questions a place is, and answers with what they come to.

        Which CLI, which of that CLI's accounts, and which of the models it says it runs as
        that account. Three and no more: a place is what a turn can fail for having named,
        and how hard an agent thinks is not one of them.

        Args:
          asked: What the walk says it is asking.

        Returns:
          The place as a step names it, or "" for a walk that was left part way through.
        """
        showing = cast(
            "App[None]",
            self.app,  
        )
        cli = await showing.push_screen_wait(Clis(self._agents))
        if not cli:
            return ""
        account = await showing.push_screen_wait(Accounts(cli))
        if account is None:
            return ""
        model = await showing.push_screen_wait(
            Catalogue(cli, account, self._agents.get(cli, ()))
        )
        if not model:
            return ""
        self._said = escape(asked)
        return _hmz().fallbacks.spec(cli, model, account)

    def applied(self) -> None:
        """Writes down every step, and answers with what it did."""
        steps = _hmz().fallbacks
        told: list[str] = []

        was = {one.spec: one for one in steps.all()}
        held = {one.spec: one for one in self._steps}
        for gone in was:
            if gone not in held:
                steps.clear(gone)
                told.append(f"[dim]{escape(gone)} falls back to nowhere[/dim]")
        for said, step in held.items():
            if was.get(said) == step:
                continue
            try:
                steps.points(said, step.to)
                steps.retrying(said, step.tries, step.policy, step.timeout)
            except ValueError as why:
                told.append(f"hmz: {escape(str(why))}")
            else:
                told.append(f"[dim]{escape(said)} {escape(_falling(step))}[/dim]")
        self.dismiss(told)

def _falling(step: Step) -> str:
    """What happens when one place cannot take a turn, as the one line a row has room for.

    Args:
      step: The step.

    Returns:
      How often the turn is taken again there, and where it goes once those are spent.
    """
    goes = f"falls back to {step.to}" if step.to else "falls back nowhere"
    if not step.tries:
        return goes
    over = f", up to {_lasting(step.timeout)}" if step.timeout else ""
    return f"{step.tries} more tries, {step.policy}{over}{_DOT}{goes}"

class Saved(Drafts[list[str]]):
    """Every agent written down under a name, which is what a flow's agents are imported from.

    An agent is a CLI, an account, a model at an effort and what it may do without being
    asked, and none of that is a thing about the flow that happens to be driving it. So it is
    worth saying once and reaching for: the reviewer you always use, the cheap one you fan out
    across, the one on somebody else's gateway.

    Nothing here is being chosen for anything. What it is for is the three things that can
    happen to one -- made, set up, taken away -- so those are the keys, and none of them lands
    until the menu is saved on the way out.
    """

    TABS: ClassVar = ("Agents",)
    LETTERS: ClassVar = frozenset({"search", "adding", "drop"})

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),
        Binding("tab", "next_tab", "next page", priority=True),
        Binding("shift+tab", "prev_tab", "previous page", priority=True),
        Binding("s", "search", "search", priority=True),
        Binding("a", "adding", "add one", priority=True),
        Binding("d", "drop", "take one away", priority=True),
    ]

    def __init__(self, agents: dict[str, tuple[Model, ...]]) -> None:
        """Reads what has been written down.

        Args:
          agents: The backends offered here, and what each of them says it runs.
        """
        super().__init__()
        self._agents = dict(agents)
        
        self._held: list[Kept] = list(_hmz().agents.all())
        
        self._was = self._held[0].name if self._held else ""
        self._said = ""

    def _ask(self) -> None:
        """Says what these are, and puts them up."""
        self.query_one("#asked", Label).update("Agents")
        self.query_one("#about", Label).update(
            "One named agent apiece: the CLI that takes its turns, the account they run as, "
            "the model at an effort and what it may do. A flow imports a copy of one where "
            "its agents are chosen, so changing one here does not change a flow already set "
            "up with it."
        )
        self._fill()
        self.query_one("#choices", OptionList).focus()

    def _fill(self) -> None:
        """Puts the agents up, with the marker beside the one the cursor is on."""
        listing = self.query_one("#choices", OptionList)
        self._follows(listing)
        shown = [one for one in self._held if self.fits(one.name, one.runs.spec)]
        self._counting = len(str(max(len(shown), 1)))
        if all(one.name != self._was for one in shown):
            self._was = shown[0].name if shown else ""
        at = next((seen for seen, one in enumerate(shown) if one.name == self._was), 0)
        listing.set_options(
            Option(
                self._row(
                    seen,
                    one.name,
                    reads((), [one.runs])[0],
                    here=seen == at,
                    inforce=False,
                ),
                id=f"={one.name}",
            )
            for seen, one in enumerate(shown)
        )
        listing.highlighted = at if shown else None
        self._drawn = listing.highlighted
        self.tabbed(self._tab_line())
        said = self._said or ("" if self._held else "no agents saved yet; a saves one")
        self.query_one("#tuning", Label).update(
            f"[$text-muted]{said}[/]" if said else ""
        )
        self.query_one("#keys", Label).update(
            "Enter to set one up · a adds one · d twice takes one away · "
            f"Esc to close{self.searching()}"
        )

    def _follows(self, listing: OptionList) -> None:
        """Takes which agent the cursor is on off the list, by the name it is kept under.

        Args:
          listing: The list.
        """
        at = listing.highlighted
        if at is not None and 0 <= at < listing.option_count:
            named = str(listing.get_option_at_index(at).id or "").removeprefix("=")
            if named:
                self._was = named

    def action_adding(self) -> None:
        """Sets up an agent that is not there yet, and holds it if it is named."""
        spare = opens_on(self._agents)
        self._sets(Kept("", spare[0] if spare else Runs("")), new=True)

    def action_drop(self) -> None:
        """Takes the agent under the cursor away, once d has been pressed twice."""
        name = self.under()
        if not name:
            return
        if not self._armed(name):
            self._said = f"press d again to take {escape(name)} away"
            self._fill()
            return
        self._held = [one for one in self._held if one.name != name]
        self._said = f"{escape(name)} goes when this menu is saved"
        self.changed()
        self._fill()

    @on(OptionList.OptionSelected)
    def _took(self, event: OptionList.OptionSelected) -> None:
        """Sets up the agent under the cursor.

        Args:
          event: What was chosen.
        """
        name = str(event.option.id or "").removeprefix("=")
        one = next((each for each in self._held if each.name == name), None)
        if one is not None:
            self._sets(one, new=False)

    @work
    async def _sets(self, one: Kept, *, new: bool) -> None:
        """Opens one agent, and holds whatever comes back.

        Args:
          one: The agent as it is now.
          new: Whether it is one that is not written down yet, which is what decides between
            adding it and writing over it.
        """
        showing = cast(
            "App[None]",
            self.app,  
        )
        fitted = await showing.push_screen_wait(
            Agent(
                one.name or "a new agent",
                one.runs,
                self._agents,
                name=one.name,
                naming=True,
            )
        )
        if fitted is None:
            return  
        named = fitted.name.strip()
        if not named:
            self._said = "an agent with no name is not one anything can import"
            self._fill()
            return

        at = next(
            (seen for seen, each in enumerate(self._held) if each.name == one.name),
            len(self._held),
        )
        held = [each for each in self._held if each.name not in (named, one.name)]
        at = min(at, len(held)) if not new else len(held)
        self._held = [*held[:at], Kept(named, fitted.runs), *held[at:]]
        self._was, self._said = named, ""
        self.changed()
        self._fill()

    def applied(self) -> None:
        """Writes down exactly what the menu is holding, and says what it now holds."""
        _hmz().agents.keep(self._held)
        self.dismiss(
            [
                f"[dim]{len(self._held)} agents saved: "
                f"{escape(', '.join(one.name for one in self._held))}[/dim]"
                if self._held
                else "[dim]no agents are saved any more[/dim]"
            ]
        )

_CORRECTS, _SIGNS_IN, _FALLS_BACK = "corrects", "signs-in", "falls"

_HELD = "provider.json"

def _tries_moved(cli: str, name: str) -> str:
    """What to say about tries written on one account before they moved, or "" for none.

    How many times over a failed turn is taken again was once a thing about an account and is
    now a thing about a place -- the CLI, the account and the model -- so the accounts store
    stopped reading it. An account written down before that move still holds it, and tries
    that quietly stopped happening are worse than tries nobody ever set: they are a setting
    somebody goes on believing in. So the file is read again as it stands, exactly for the
    key nothing reads, and what is found is said where somebody is looking at the account.

    Args:
      cli: The backend the account belongs to.
      name: What it is called, or "" for the account this machine is already signed into.

    Returns:
      The line to say under the list, or "" for an account holding no such thing -- which is
      every account made since it moved.
    """
    import json

    from hmz.providers import LOCAL, alone, where

    try:
        at = alone(cli) if name == LOCAL else where(cli, name) / _HELD
        said = json.loads(at.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(said, dict):
        return ""
    tries = cast("dict[str, Any]", said).get("retries")
    if not isinstance(tries, int) or isinstance(tries, bool) or tries < 1:
        return ""
    return (
        "the tries written down on this account are no longer read: how often a failed turn "
        "is taken again is said of a place now, on /fallback"
    )

class Account(Picks):
    """What to do with one account: correct it, sign it in again, say where it falls back to.

    Its own menu rather than a letter apiece on the list of accounts. They are three questions
    about the account under the cursor, and a sheet whose keys are `l` and `f` is a sheet
    whose keys have to be learned from a line at the bottom of it -- while enter, which every
    list already means, was doing one of the three.

    How many times over a failed turn is tried again is not among them. That is a thing about
    the place a turn runs at rather than about the credentials it runs with, and `/fallback`
    is the menu it is said on.
    """

    def __init__(self, cli: str, name: str) -> None:
        """Asks about one account.

        Args:
          cli: The backend it belongs to.
          name: What it is called, or "" for the account this machine is already signed into.
        """
        super().__init__()
        self._cli = cli
        self._name = name

        self._stale_tries = _tries_moved(cli, name)
        self.asked = f"{cli}/{name}" if name else f"{cli}, as this machine is signed in"
        self.about = (
            "What to do with this account. Correcting it and saying where it falls back to "
            "land when the accounts menu is saved; signing in happens as it is asked for."
        )

    def rows(self) -> list[tuple[str, str, str]]:
        """The three, less the two there is nothing to do for this machine's own account."""
        held = [
            (
                _FALLS_BACK,
                "falls back to",
                "which account a turn carries on under when this one fails",
            ),
        ]
        if not self._name:
            return held
        return [
            (
                _CORRECTS,
                "correct what it holds",
                "the answers its way in was made with, asked again",
            ),
            (
                _SIGNS_IN,
                "sign in again",
                "run its own way in again; it owns the terminal while it does",
            ),
            *held,
        ]

    def nothing(self) -> str:
        """Why two of them are not here, and what this account holds that nothing reads.

        Returns:
          One line apiece, or "" for the account with neither to say -- which is any account
          humanize made and nobody ever wrote tries on.
        """
        said: list[str] = []
        if not self._name:
            said.append(
                f"this is {escape(self._cli)} as this machine is already signed in: "
                "humanize keeps no credentials for it, so there is nothing to correct or "
                "sign in"
            )
        if self._stale_tries:
            said.append(self._stale_tries)
        return "\n".join(said)

class Providers(Drafts[list[str]]):
    """Every account there is to run an agent as, under a heading per CLI.

    Read rather than chosen from: which account an agent runs as is asked where that agent is
    set up, so nothing here is being picked for anything. What it is for is what can happen to
    one -- made, set up again, signed in again, marked as where a turn goes when another
    account fails, taken away -- and all but the first two of those are one menu, opened with
    enter on the account they are about. A row of letter keys was a row of keys somebody had
    to read off the bottom of the screen while enter, which every list already means, did one
    of the four.

    What is written down without running anything is held until the menu is saved: taking one
    away, marking one as a fallback, correcting what one holds. What cannot be held is what
    runs a command of its own -- making an account and signing one in own the terminal while
    they run, and something that has already happened is not a draft.

    Each row is the name, the way it was made by and the variables it sets. Their names and
    never a value: this is drawn where somebody can read it.
    """

    TABS: ClassVar = ("Providers",)
    LETTERS: ClassVar = frozenset({"search", "adding", "drop"})

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),
        Binding("tab", "next_tab", "next page", priority=True),
        Binding("shift+tab", "prev_tab", "previous page", priority=True),
        Binding("s", "search", "search", priority=True),
        Binding("a", "adding", "make one", priority=True),
        Binding("d", "drop", "take one away", priority=True),
    ]

    def __init__(self) -> None:
        """Reads every account there is."""
        super().__init__()
        self._found: list[Provider] = []
        
        self._gone: set[str] = set()

        self._chains: dict[str, str] = {}
        
        self._edits: dict[str, dict[str, str]] = {}

        self._alike: dict[str, tuple[str, ...]] = {}

        self._was = ""

        self._said = ""
        self._told: list[str] = []

    def _ask(self) -> None:
        """Says what these are, and puts them up."""
        self.query_one("#asked", Label).update("Providers")
        self.query_one("#about", Label).update(
            "One named set of credentials per account, kept apart from the CLI's own and "
            "from each other's. An agent is given one where it is set up, and runs its turns "
            "as that account. Enter opens what there is to do with one. Taking one away, "
            "saying where it falls back to and correcting one land when this menu is saved; "
            "making one and signing one in happen as they are asked for."
        )
        self._read()
        self._fill()
        self.query_one("#choices", OptionList).focus()

    def _read(self) -> None:
        """Reads every account off the disk, which is what the rows are drawn from.

        The account this machine is already signed into is one of them, under each CLI that
        has one of its own: it is what an agent nobody gave an account runs as, and it is
        where that agent's chain begins, so it is a row to press f and t on like any other.
        Under a CLI with no accounts there is nothing for it to fall back to, so it is a row
        there only where something has already been said about it -- a chain that outlived
        the accounts it named, or tries written down before they moved -- which must not be
        a setting somebody believes in that nothing shows.

        Last in each CLI's group rather than first: what somebody came here to read is the
        accounts they made, and this is the one that was always there.
        """
        from hmz.providers import LOCAL

        hmz = _hmz()
        accounts = hmz.accounts
        held = accounts.all()
        whose = {each.cli for each in held}
        mine = [
            one
            for profile in hmz.backends()
            if (one := accounts.find(profile.name, LOCAL)) is not None
            and (
                profile.name in whose
                or one.fallback

                or _tries_moved(profile.name, LOCAL)
            )
        ]
        self._found = sorted(
            [*held, *mine], key=lambda one: (one.cli, not one.name, one.name)
        )

    def _named(self, one: Provider) -> str:
        """One account as it is keyed here, which is by the CLI it belongs to and its name."""
        return f"{one.cli}/{one.name}"

    def _about(self, one: Provider) -> str:
        """What a row says about one account, and what is going to happen to it."""
        said = (
            _sets(one) if one.name else "the CLI as this machine is already signed in"
        )
        if self._named(one) in self._edits:
            said += f"{_DOT}corrected"
        falls = self._chains.get(self._named(one), one.fallback)
        if falls:
            said += f"{_DOT}falls back to {falls}"
        if self._named(one) in self._gone:
            said += f"{_DOT}to be taken away"
        return said

    def _fill(self) -> None:
        """Puts the accounts up under a heading apiece, marked where the cursor is."""
        listing = self.query_one("#choices", OptionList)
        self._follows(listing)
        shown = [one for one in self._found if self.fits(one.name, one.cli, one.way)]
        self._counting = len(str(max(len(shown), 1)))
        if all(self._named(one) != self._was for one in shown):

            self._was = self._named(shown[0]) if shown else ""
        rows: list[Option] = []
        group, landing = "", 0
        for seen, one in enumerate(shown):
            named = self._named(one)
            if one.cli != group:
                group = one.cli

                if rows:
                    rows.append(Option("", disabled=True))
                rows.append(
                    Option(f"{_INDENT}[$primary]{escape(group)}[/]", disabled=True)
                )
            if named == self._was:
                landing = len(rows)
            rows.append(
                Option(
                    self._row(
                        seen,
                        one.name or "as local",
                        self._about(one),
                        here=named == self._was,
                        inforce=False,
                    ),
                    id=f"={named}",
                )
            )
        listing.set_options(rows)
        listing.highlighted = landing if shown else None
        self._drawn = listing.highlighted
        self.tabbed(self._tab_line())
        said = self._said or ("" if self._found else "no accounts yet; a makes one")
        self.query_one("#tuning", Label).update(
            f"[$text-muted]{said}[/]" if said else ""
        )
        self.query_one("#keys", Label).update(
            "Enter for what to do with one · a makes one · d twice takes one away · "
            f"Esc to close{self.searching()}"
        )

    def _follows(self, listing: OptionList) -> None:
        """Takes which account the cursor is on off the list, by `cli/name`.

        The headings between them are rows nothing can land on, so a row number is not an
        account and the id on the row is the only thing that says which one is meant.

        Args:
          listing: The list.
        """
        at = listing.highlighted
        if at is not None and 0 <= at < listing.option_count:
            named = str(listing.get_option_at_index(at).id or "").removeprefix("=")
            if named:
                self._was = named

    def _under(self) -> Provider | None:
        """The account the cursor is on, or None where the list has nothing in it."""
        return next((one for one in self._found if self._named(one) == self._was), None)

    def _machines(self, cli: str, doing: str) -> str:
        """Why the account this machine is signed into is not one to do that to.

        Args:
          cli: The backend it is of.
          doing: What was asked for.

        Returns:
          The line to say under the list. humanize did not make that account and keeps no
          credentials for it -- it is the CLI as whoever is at this machine runs it -- so the
          only things to say about it are where it falls back to and how it is tried again,
          which is what enter offers on it.
        """
        telemetry.snag("key-does-nothing", sheet="Providers", doing=doing)
        return (
            f"there is nothing to {doing}: this is {escape(cli)} as this machine is already "
            "signed in. Enter says what it does take"
        )

    @work
    async def action_fallback(self, one: Provider | None = None) -> None:
        """Asks which account a turn under this one carries on under when it fails.

        Args:
          one: The account, or None for the one the cursor is on.
        """
        one = one or self._under()
        if one is None:
            return
        named = self._named(one)
        showing = cast(
            "App[None]",
            self.app,  
        )
        chosen = await showing.push_screen_wait(
            Falls(one.cli, one.name, self._chains.get(named, one.fallback))
        )
        if chosen is None:
            return  
        if chosen == one.fallback:
            self._chains.pop(named, None)
        else:
            self._chains[named] = chosen
        self._said = ""
        self.changed()
        self._fill()

    def action_drop(self) -> None:
        """Marks the account under the cursor to be taken away, once d is pressed twice."""
        one = self._under()
        if one is None:
            return
        if not one.name:
            self._said = self._machines(one.cli, "take away")
            self._fill()
            return
        named = self._named(one)
        if named in self._gone:
            self._gone.discard(named)  
            self._said = f"{escape(named)} stays"
            self.changed()
            self._fill()
            return
        if not self._armed(named):
            self._said = (
                f"press d again to take {escape(named)} away, credentials and all"
            )
            self._fill()
            return
        self._gone.add(named)
        self._said = f"{escape(named)} goes when this menu is saved"
        self.changed()
        self._fill()

    @on(OptionList.OptionSelected)
    def _took(self, event: OptionList.OptionSelected) -> None:
        """Opens what there is to do with the account under the cursor.

        Args:
          event: What was chosen.
        """
        named = str(event.option.id or "").removeprefix("=")
        one = next((each for each in self._found if self._named(each) == named), None)
        if one is not None:
            self._doing(one)

    @work
    async def _doing(self, one: Provider) -> None:
        """Asks what to do with one account, and does it.

        Args:
          one: The account.
        """
        showing = cast(
            "App[None]",
            self.app,  
        )
        said = await showing.push_screen_wait(Account(one.cli, one.name))
        if said is None:
            return  
        if said == _CORRECTS:
            self._corrects(one)
        elif said == _SIGNS_IN:
            self.action_again(one)
        elif said == _FALLS_BACK:
            self.action_fallback(one)

    @work
    async def _corrects(self, one: Provider) -> None:
        """Asks what one account is to hold, starting from what it holds now.

        A secret is never read back on to the screen, so what is typed here replaces what was
        there rather than being edited into it: a key is written once and read never.

        Args:
          one: The account.
        """
        from dataclasses import replace

        if not one.name:
            self._said = self._machines(one.cli, "correct")
            self._fill()
            return
        way = _hmz().accounts.way(one.cli, one.way)
        if way is None:
            self._said = f"{escape(one.way)} is not a way in {escape(one.cli)} has"
            self._fill()
            return
        showing = cast(
            "App[None]",
            self.app,  
        )
        signs = await showing.push_screen_wait(
            Signing(one.cli, way, name=one.name, held=one.env)
        )
        if signs is None:
            return  
        named = self._named(one)
        self._edits[named] = signs.answers

        corrected = replace(one, env=signs.answers)
        among = _hmz().accounts.serves(corrected)
        self._alike.pop(named, None)
        if among:
            chosen = await showing.push_screen_wait(Alike(corrected, among))
            if chosen:
                self._alike[named] = tuple(chosen)
        self._said = f"{escape(named)} is corrected when this menu is saved"
        if self._alike.get(named):
            self._said += f", for {escape(', '.join(self._alike[named]))} as well"
        self.changed()
        self._fill()

    @work
    async def action_adding(self) -> None:
        """Asks which CLI, and then walks that backend's own way in.

        Two questions rather than one, because the second is only answerable once the first
        has been: a backend's ways in are its own. What comes of it has already happened by
        the time it lands -- a login owns the terminal while it runs -- so it is not one of
        the things this menu holds until it is saved.

        The list of CLIs is also where a CLI of your own is written down: somebody who cannot
        find their agent in it is somebody whose agent is not one humanize drives, and that
        is a thing to say where the question was asked rather than on a key of the sheet
        before it.
        """
        showing = cast(
            "App[None]",
            self.app,  
        )
        while True:
            cli = await showing.push_screen_wait(Backends())
            if cli is None:
                return  
            if cli == _SPEAKS:
                await self._speaks()
                return
            outcome = await made(showing, cli)

            if outcome.provider is not None or outcome.why:
                break
        one = outcome.provider
        if one is None:  
            self._said = escape(outcome.why)
            self._fill()
            return
        self._told.append(
            f"[dim]{escape(one.cli)}/{escape(one.name)} is written down at "
            f"{escape(str(one.at))}[/dim]"
        )
        if outcome.way_runs and not outcome.status:

            self._told.append(
                f"[dim]{escape(one.cli)}/{escape(one.name)} is signed in[/dim]"
            )
        elif outcome.status:
            self._told.append(f"hmz: signing it in exited {outcome.status}")
        if outcome.copied:
            self._told.append(
                f"[dim]{escape(one.name)} is written down for "
                f"{escape(', '.join(outcome.copied))} too[/dim]"
            )
        self._said = self._landed(one, outcome.status, runs=outcome.runs)
        if outcome.copied:
            self._said += f"{_DOT}and runs {escape(', '.join(outcome.copied))} too"
        self._read()
        self._was = self._named(one)
        self._fill()

    def _landed(self, one: Provider, status: int, *, runs: int) -> str:
        """What to say about an account that has just been made or signed in again.

        Args:
          one: The account.
          status: What its way in exited with, or 0 for one that ran nothing.
          runs: How many models its CLI then said it runs as it.

        Returns:
          The line to say under the list.
        """
        if status:
            return f"signing {escape(one.name)} in exited {status}"
        if runs:
            return f"{escape(one.cli)} says it runs {runs} models as {escape(one.name)}"
        return (
            f"{escape(one.cli)} did not say what it runs as {escape(one.name)}; "
            "r on its models asks again"
        )

    @work
    async def action_again(self, one: Provider | None = None) -> None:
        """Runs one account's own way in again, asking for whatever it still needs.

        Args:
          one: The account, or None for the one the cursor is on.
        """
        accounts = _hmz().accounts
        one = one or self._under()
        if one is None:
            return
        if not one.name:
            self._said = self._machines(one.cli, "sign in")
            self._fill()
            return
        way = accounts.way(one.cli, one.way)
        if way is None or not way.argv:
            self._said = (
                f"{escape(one.name)} was made by {escape(one.way)}, which has nothing to "
                "run; enter corrects what it holds instead"
            )
            self._fill()
            return
        showing = cast(
            "App[None]",
            self.app,  
        )

        answers = dict(one.env)
        if accounts.asks(way, answers):
            signs = await showing.push_screen_wait(Signing(one.cli, way, name=one.name))
            if signs is None:
                return  
            answers |= signs.answers
        try:
            with handed_over(showing):
                status = accounts.sign_in(one, way, answers)
        except OSError as why:  
            self._said = escape(f"{way.argv[0]}: {why}")
            self._fill()
            return

        self._said = self._landed(
            one, status, runs=0 if status else await asks(one.cli, one.name)
        )
        self._told.append(
            f"[dim]{escape(one.cli)}/{escape(one.name)} is signed in[/dim]"
            if not status
            else f"hmz: {escape(way.argv[0])} exited {status}"
        )
        self._fill()

    async def _speaks(self) -> None:
        """Asks for a CLI of your own that speaks ACP, and writes it down as a backend.

        Reached from the list of backends a new account is for, because that is the moment
        somebody finds out that the agent they want to run is not one humanize drives. What
        is written down outlives the run, so it is a backend from the next prompt on, in this
        workspace and every other -- which is why it is not one of the things this menu holds
        until it is saved.
        """
        from hmz import backends

        showing = cast(
            "App[None]",
            self.app,  
        )
        said = await showing.push_screen_wait(Speaks())
        if said is None:
            return
        command, name = said
        try:
            backends.remember(name, shlex.split(command))
        except (OSError, ValueError) as why:
            self._said = escape(str(why))
            self._fill()
            return
        self._said = f"{escape(name)} is a backend from here on"
        self._told.append(
            f"[dim]{escape(name)} is written down: `{escape(command)}` starts it, "
            "and it is a backend from here on[/dim]"
        )
        self._fill()

    def applied(self) -> None:
        """Does everything the menu was holding, and answers with what became of each."""
        accounts = _hmz().accounts
        told = list(self._told)

        for taken in sorted(self._gone):
            cli, _, name = taken.partition("/")
            try:
                gone = accounts.remove(cli, name)
            except ValueError as why:  
                told.append(f"hmz: {escape(str(why))}")
                continue
            told.append(
                f"[dim]{escape(taken)} is gone, credentials and all[/dim]"
                if gone
                else f"hmz: no provider {escape(taken)}"
            )
        for one in self._found:
            named = self._named(one)
            if named in self._gone:
                continue  
            if (answers := self._edits.get(named)) is not None:
                try:
                    corrected = accounts.write(one.cli, one.name, one.way, answers)
                except (OSError, ValueError) as why:
                    told.append(f"hmz: {escape(str(why))}")
                    continue
                told.append(f"[dim]{escape(named)} is corrected[/dim]")
                for cli in self._alike.get(named, ()):
                    try:
                        accounts.copies(corrected, cli)
                    except (OSError, ValueError) as why:
                        told.append(f"hmz: {escape(str(why))}")
                        continue
                    told.append(
                        f"[dim]{escape(cli)}/{escape(one.name)} is corrected with it[/dim]"
                    )
            if (falls := self._chains.get(named)) is not None:
                try:
                    accounts.points(one.cli, one.name, falls)
                except ValueError as why:
                    told.append(f"hmz: {escape(str(why))}")
                else:
                    told.append(
                        f"[dim]{escape(named)} falls back to {escape(falls)}[/dim]"
                        if falls
                        else f"[dim]{escape(named)} falls back to nowhere[/dim]"
                    )
        self.dismiss(told)

    def leaving(self) -> None:
        """Asks about what is held, and answers with what happened where nothing is.

        A menu that made an account and then held nothing still has something to say: what it
        did, it did as it was asked to, and the transcript is where that is said.
        """
        if not self._changed:
            self.dismiss(self._told or None)
            return
        self.asks_to_save()

carries_on, _COLLECTS, _WHERE_IT_IS = "carry-on", "collect", "where"

_ENOUGH_TASK = 60

class Doing(NamedTuple):
    """What somebody asked to have done with one run that has already happened.

    Attributes:
      epic: The run, by the directory it is written in, or None where this is only what the
        sheet has to say on the way out.
      doing: What to do with it, which is what the menu under it answered, and "" where the
        sheet did it itself.
      said: What happened while the sheet was open, for the transcript: a menu that gathered
        a trace and said nothing afterwards is one nobody can read back.
    """

    epic: Path | None = None
    doing: str = ""
    said: tuple[str, ...] = ()

def _many(count: int, thing: str) -> str:
    """How many of something there were, said as English says it.

    Args:
      count: How many.
      thing: What they are, in the singular.

    Returns:
      The two words -- `1 session`, `3 sessions` -- since a sheet is prose and `1 sessions`
      is a sheet that reads as a template somebody forgot to finish.
    """
    return f"{count} {thing}" if count == 1 else f"{count} {thing}s"

def _asked_for(task: str) -> str:
    """What a run was asked to do, as much of it as a row has room for.

    Args:
      task: The whole of it, which is however long whoever started the run made it.

    Returns:
      Its first line's worth, on one line, cut with an ellipsis where it was cut.
    """
    said = " ".join(task.split())
    return said if len(said) <= _ENOUGH_TASK else f"{said[: _ENOUGH_TASK - 1]}…"

def _when(said: str) -> str:
    """One of the moments an epic writes down, as a row of a list says one.

    Args:
      said: The moment, as it was written -- `2026-08-16T03:04:05.123Z`.

    Returns:
      It, to the minute, and whatever was written where that is not what it is.
    """
    if len(said) < len("YYYY-MM-DDTHH:MM"):
        return said
    return said[:16].replace("T", " ")

class Does(Picks):
    """What to do with one run that has already happened.

    Which is a second question rather than more keys on the first: a list of runs is a list
    somebody is reading, and what there is to do with one of them depends on the one under
    the cursor -- a flow that says it can be picked up is picked up, and one that says
    nothing is a run to read rather than a run to continue.
    """

    def __init__(self, ran: Ran, *, resumable: bool) -> None:
        """Asks about one run.

        Args:
          ran: The run, as it was written down.
          resumable: Whether its flow says now that it can be picked up, which is asked of
            the flow rather than of the run: a flow may have been rewritten since.
        """
        super().__init__()
        self._ran = ran
        self._resumable = resumable
        self.asked = f"{_when(ran.began)}{_DOT}{ran.flow}"
        self.about = (
            f"What to do with this run. It {_how(ran)}, driving "
            f"{_many(len(ran.agents), 'agent')} through "
            f"{_many(len(ran.sessions), 'session')}."
        )

    def rows(self) -> list[tuple[str, str, str]]:
        """Carrying on where it stopped, where that is a thing this flow can do, and reading."""
        held: list[tuple[str, str, str]] = []
        if self._resumable:
            held.append(
                (
                    carries_on,
                    "carry on from here",
                    "run the flow again on what this run left behind",
                )
            )
        held.append(
            (
                _COLLECTS,
                "collect a trace",
                "its sessions, and the programs it ran, as one trace to read",
            )
        )
        held.append(
            (
                _WHERE_IT_IS,
                "where it is",
                "the directory this run is written in, sessions and all",
            )
        )
        return held

    def nothing(self) -> str:
        """Why carrying on is not one of the things there are to do, where it is not."""
        if self._resumable:
            return ""
        return (
            f"{escape(self._ran.flow)} does not say it can be picked up, so there is "
            "nothing to carry on from"
        )

def _how(ran: Ran) -> str:
    """How one run ended, as a line about it reads.

    Args:
      ran: The run.

    Returns:
      What became of it, in words: a run with no end written down is one that was abandoned
      where it stood -- the machine it was on went, or the interface came down under it.
    """
    return {
        "done": "finished",
        "failed": "failed",
        "stopped": "was stopped",
    }.get(ran.how, "was left unfinished")

def collected(ran: Ran) -> tuple[Path, str]:
    """Gathers what one run left behind into a trace file, and says what is in it.

    That run's own sessions and no others: a directory may have been run in a hundred times,
    and a trace filed under one of those runs while holding the other ninety-nine is a trace
    of nothing anybody asked about. They are asked for by the ids the run wrote down rather
    than by directory, so a flow that worked in a machine's mirror is in its own trace too.

    Beside the run rather than in this directory: an epic is what a run was, and the trace of
    that run belongs with the sessions it points at and the state it left. A trace of what a
    directory holds whoever opened it is `hmz trace collect --all`, and a trace to attach to
    an issue is `--output`: both are a command line, there being no run here to hang either
    on.

    Args:
      ran: The run.

    Returns:
      Where the trace was written, and a line saying what it holds.
    """
    where, document = _hmz().epics.traced(ran.at)
    said = document["otherData"]
    held = f"{said.get('sessions', '0')} sessions, {said.get('slices', '0')} slices"
    if said.get("programs"):
        held += f", {said['programs']} programs"
    return where, held

class Epics(Sheet[Doing]):
    """Every run of a flow in this directory, newest first, and what to do with one.

    A run is written down as it happens -- which flow, on what, by which agents, and which
    sessions each of them opened -- and until now nothing showed them. What they are for is
    two things: reading one back afterwards, which is what the links to its sessions are, and
    carrying one on, which is what a flow that says it can be picked up is for.

    Read rather than chosen from, so enter opens what there is to do with the run under the
    cursor rather than doing any of it.
    """

    LETTERS: ClassVar = frozenset({"search"})

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),
        Binding("s", "search", "search", priority=True),
    ]

    def __init__(self, workspace: Path | None = None, *, running: bool = False) -> None:
        """Reads every run of this directory.

        Args:
          workspace: Which directory's, defaulting to this one.
          running: Whether a flow is running now, which is what makes carrying one on a
            thing to say no to rather than a thing to offer.
        """
        super().__init__()
        from hmz.sdk import Hmz

        runs = Hmz(workspace).epics

        self._ran = [
            one
            for one in (runs.read(at) for at in reversed(runs.all()))
            if one is not None
        ]
        self._underway = running

        self._was = ""
        
        self._said = ""

        self._resumes: dict[str, bool] = {}
        
        self._told: list[str] = []

    def _ask(self) -> None:
        """Says what these are, and puts them up."""
        self.query_one("#asked", Label).update("Epics")
        self.query_one("#about", Label).update(
            "Every run of a flow in this directory, newest first: what it was, how it went, "
            "and how many sessions it opened. Enter says what there is to do with one."
        )
        self._fill()
        self.query_one("#choices", OptionList).focus()

    def _about(self, ran: Ran) -> str:
        """What a row says about one run: what it was asked to do, how it went, and its size.

        How it went only where it went some other way than finishing: a list of runs is
        mostly runs that finished, and a column saying so of nearly all of them is a column
        that says nothing while taking the room the ones that did not need.
        """
        said = _asked_for(ran.task) if ran.task else "no task"
        held = f"{said}{_DOT}{_many(len(ran.sessions), 'session')}"
        if ran.how != "done":
            held += f"{_DOT}{_how(ran)}"

        return f"{held}{_DOT}can be picked up" if self._picks_up(ran.flow) else held

    def _fill(self) -> None:
        """Puts the runs up, marked where the cursor is."""
        listing = self.query_one("#choices", OptionList)
        self._follows(listing)
        shown = [one for one in self._ran if self.fits(one.flow, one.task, one.name)]
        self._counting = len(str(max(len(shown), 1)))
        if all(one.name != self._was for one in shown):
            self._was = shown[0].name if shown else ""
        listing.set_options(
            Option(
                self._row(
                    seen,
                    f"{_when(one.began)}{_DOT}{one.flow}",
                    _briefly(self._about(one), self.size.width),
                    here=one.name == self._was,
                    inforce=False,
                ),
                id=f"={one.name}",
            )
            for seen, one in enumerate(shown)
        )
        listing.highlighted = (
            next((at for at, one in enumerate(shown) if one.name == self._was), 0)
            if shown
            else None
        )
        self._drawn = listing.highlighted
        said = self._said or ("" if self._ran else self._nothing())
        self.query_one("#tuning", Label).update(
            f"[$text-muted]{said}[/]" if said else ""
        )
        self.query_one("#keys", Label).update(
            f"Enter for what to do with one · Esc to close{self.searching()}"
        )

    def leaving(self) -> None:
        """Leaves, saying in the transcript whatever was gathered while this was open."""
        self.dismiss(Doing(said=tuple(self._told)) if self._told else None)

    def _nothing(self) -> str:
        """What an empty list says, which is that nothing has been run here yet."""
        return "no flow has been run in this directory yet"

    async def _collects(self, ran: Ran) -> None:
        """Gathers what one run left behind into a trace, beside the run itself.

        Off the event loop: reading a run's sessions back is every log every backend wrote
        for it, which is seconds on a long run -- and an interface that stopped redrawing
        while it ran would be one that looked as though it had gone away.

        Args:
          ran: The run.
        """
        import asyncio

        self._said = f"collecting {escape(ran.name)}…"
        self._fill()
        try:
            at, held = await asyncio.to_thread(collected, ran)
        except (OSError, ValueError) as why:
            self._said = escape(str(why))
            self._fill()
            return
        self._said = f"{escape(str(at))}{_DOT}{escape(held)}"
        self._told.append(f"[dim]{escape(str(at))} — {escape(held)}[/dim]")
        self._fill()

    def _follows(self, listing: OptionList) -> None:
        """Takes which run the cursor is on off the list, by the directory it is written in."""
        at = listing.highlighted
        if at is not None and 0 <= at < listing.option_count:
            named = str(listing.get_option_at_index(at).id or "").removeprefix("=")
            if named:
                self._was = named

    def _under(self) -> Ran | None:
        """The run the cursor is on, or None where the list has nothing in it."""
        return next((one for one in self._ran if one.name == self._was), None)

    def _picks_up(self, flow: str) -> bool:
        """Whether one flow says now that it can be picked up.

        Asked of the flow rather than of the run that recorded it: a flow is a directory on
        disk and may have been rewritten since, and what can happen next is what it says now.
        Asked once per flow, since reading one means running its file.

        Args:
          flow: The flow, as the run named it.

        Returns:
          Whether it is resumable, and False for one that will not load at all -- a flow that
          cannot be read cannot be run, which is what carrying on would come to.
        """
        if flow not in self._resumes:
            try:
                self._resumes[flow] = _hmz().flows.resumes(flow)
            except Exception:  
                self._resumes[flow] = False
        return self._resumes[flow]

    @on(OptionList.OptionSelected)
    def _took(self, event: OptionList.OptionSelected) -> None:
        """Opens what there is to do with the run under the cursor.

        Args:
          event: What was chosen.
        """
        named = str(event.option.id or "").removeprefix("=")
        one = next((each for each in self._ran if each.name == named), None)
        if one is not None:
            self._doing(one)

    @work
    async def _doing(self, ran: Ran) -> None:
        """Asks what to do with one run, and does it or answers with it.

        Args:
          ran: The run.
        """
        showing = cast(
            "App[None]",
            self.app,  
        )
        said = await showing.push_screen_wait(
            Does(ran, resumable=self._picks_up(ran.flow))
        )
        if said is None:
            return  
        if said == _WHERE_IT_IS:
            self._said = escape(str(ran.at))
            self._fill()
            return
        if said == _COLLECTS:
            await self._collects(ran)
            return
        if said == carries_on and self._underway:

            self._said = "a flow is running; ctrl+c twice stops it before another can be picked up"
            self._fill()
            return
        self.dismiss(Doing(ran.at, said, tuple(self._told)))

_BOX = ("┌", "┐", "└", "┘", "─", "│")

_DOWN, _UP, _NEITHER = "↓", "↑", "┆"

_WIDEST = 64

_UNDER, _LAST_UNDER = "├╴", "└╴"
_FLEET_WORKING, _FLEET_DONE = "◆", "◇"

_FLEET_SHOWN = 4

EVERY = ""

_ON_BOARD = "\x00"

_BOARDED = "\x01"

_ON_IT = "◈"

class Drawn(NamedTuple):
    """One agent of a run, as the diagram on `/status` draws it.

    Attributes:
      who: The agent id, which is what attaching to it names and what the graph counts it
        under.
      named: What the flow calls it, or "" for a flow that names none of its agents.
      runs: What it runs, as the line that says what each agent is says it.
      working: Whether it has a turn open right now.
      reading: Whether its transcript is the one on the screen behind this sheet.
    """

    who: str
    named: str = ""
    runs: str = ""
    working: bool = False
    reading: bool = False

def _boxed(said: list[tuple[str, str]], width: int) -> list[str]:
    """One agent, drawn as a box the width of the diagram.

    Args:
      said: The lines to put in it, each as the colour to draw it in and the plain words to
        draw. Plain, so that what will not fit can be cut before any markup is put round it:
        a bracket an agent's name happens to hold is a bracket, and an escape of one is
        characters that are not columns.
      width: How wide to draw it, borders included.

    Returns:
      The box, a line at a time, each exactly as wide as the last.
    """
    left, right, under_left, under_right, across, side = _BOX
    room = width - 4
    return [
        f"[$text-muted]{left}{across * (width - 2)}{right}[/]",
        *(
            f"[$text-muted]{side}[/] [{colour}]{escape(_fits(line, room))}[/]"
            f"{' ' * max(0, room - len(_fits(line, room)))} [$text-muted]{side}[/]"
            for colour, line in said
        ),
        f"[$text-muted]{under_left}{across * (width - 2)}{under_right}[/]",
    ]

def _fits(said: str, room: int) -> str:
    """One line of a box, cut to the room there is for it rather than running past its side.

    Args:
      said: The words, as they are.
      room: How many columns there are between the two sides of the box.

    Returns:
      Them, or as much of them as fits with an ellipsis where the rest was.
    """
    return said if len(said) <= room else said[: room - 1] + "…"

def _joins(down: int, up: int, width: int) -> list[str]:
    """The arrows between two boxes, saying which way the flow went and how often.

    Args:
      down: How many times the agent above handed to the one below.
      up: How many times it came back the other way.
      width: How wide the boxes are, so the arrows sit under them rather than beside them.

    Returns:
      The one line between the two boxes.
    """
    ways = [f"{_DOWN} {down}" for _ in range(1) if down] + [
        f"{_UP} {up}" for _ in range(1) if up
    ]
    said = "   ".join(ways) or _NEITHER
    return [f"[$text-muted]{' ' * min(4, max(0, width // 2 - 2))}{said}[/]"]

def diagram(drawn: Sequence[Drawn], shape: Shape, width: int) -> list[list[str]]:
    """The agents of a run and the handovers between them, as one box apiece.

    The shape of a flow is not written anywhere: a flow is a Python file that may branch any
    way it likes, so what it did is read off the turns going past. Drawn down the page in the
    order the flow takes its agents, with the handovers between neighbours as the arrows that
    join them -- which is the shape of nearly every flow there is, since a flow is written as
    one agent after another. The rest are said under it rather than drawn as lines crossing
    the page, there being no way to draw those in a terminal that reads as anything.

    Args:
      drawn: The agents, in the order the flow takes them.
      shape: The run as a graph, which says who is working and who handed to whom.
      width: How much room there is across.

    Returns:
      One block of lines per agent, in the same order: the arrows above it and then its box,
      so that a list of blocks is the diagram from top to bottom.
    """
    across = max(24, min(_WIDEST, width - len(_INDENT) - 5))
    blocks: list[list[str]] = []
    for at, one in enumerate(drawn):
        block = (
            []
            if at == 0
            else _joins(
                shape.handovers.get((drawn[at - 1].who, one.who), 0),
                shape.handovers.get((one.who, drawn[at - 1].who), 0),
                across,
            )
        )
        taken = shape.turns.get(one.who, 0)
        mark = _WORKING if one.working else _IDLE
        head = _DOT.join(part for part in (one.named, short(one.who)) if part)
        under = _DOT.join(
            part
            for part in (
                one.runs,
                f"{taken} turn{'' if taken == 1 else 's'}" if taken else "",
                "reading" if one.reading else "",
            )
            if part
        )
        colour = "$secondary" if one.working else "$text-muted"
        drawn_block = (
            block
            + _boxed([(colour, f"{mark} {head}"), ("$text-muted", under)], across)
            + fleet(shape.under.get(one.who, ()), across)
        )
        blocks.append([f"{_INDENT}{line}" for line in drawn_block])
    return blocks

def fleet(under: Sequence[Under], width: int) -> list[str]:
    """The agents one agent started of its own, hanging off the bottom of its box.

    Drawn rather than listed elsewhere because that is where they are: a subagent is a thing
    the agent above it is doing, so it belongs under that agent and nowhere else. And drawn
    without a box of its own, for the reason it has no transcript: a box is what a flow's own
    agents wear, and one round a subagent would say it was another agent to attach to.

    Args:
      under: The fleet, oldest first.
      width: How wide the boxes are, so these sit under them rather than beside them.

    Returns:
      One line per subagent, and nothing at all for an agent that has started none. A fleet
      too long to draw is cut, with a line saying how many were left off: a turn that started
      forty is a turn nobody wants forty rows about.
    """
    if not under:
        return []
    shown = list(under[:_FLEET_SHOWN])
    rest = len(under) - len(shown)
    room = max(width - 6, 12)
    lines: list[str] = []
    for at, one in enumerate(shown):
        last = at == len(shown) - 1 and not rest
        mark = _FLEET_WORKING if one.working else _FLEET_DONE
        colour = "$secondary" if one.working else "$text-muted"
        lines.append(
            f"[$text-muted]{'  ' + (_LAST_UNDER if last else _UNDER)}[/]"
            f"[{colour}]{mark}[/] [$text-muted]{escape(_fits(one.about, room))}[/]"
        )
    if rest:
        lines.append(f"[$text-muted]  {_LAST_UNDER}{_FLEET_DONE} and {rest} more[/]")
    return lines

def elsewhere(drawn: Sequence[Drawn], shape: Shape) -> list[str]:
    """The handovers the diagram has no arrow for, said rather than drawn.

    Which are the ones between agents the boxes did not put next to each other: a line
    crossing the page from the first box to the fourth is a line nothing in a terminal draws
    readably, so it is a row under the diagram instead.

    Args:
      drawn: The agents, in the order the flow takes them.
      shape: The run as a graph.

    Returns:
      One line per handover the boxes have no arrow for, which is nothing at all for the
      flows that are one agent after another.
    """
    order = {one.who: at for at, one in enumerate(drawn)}
    return [
        f"{escape(short(sender))} → {escape(short(taker))}{_DOT}×{often}"
        for (sender, taker), often in sorted(shape.handovers.items())
        if abs(order.get(sender, -1) - order.get(taker, -1)) != 1
        or sender not in order
        or taker not in order
    ]

class Entry(Sheet[tuple[str, str]]):
    """One line of the board, typed: what it is called, and then what it says.

    Typed rather than picked, because it is words: a line of a board is what somebody wants
    said, and a list has nothing to offer them. Two questions in one sheet, since a line
    nobody named is not a line and a name with nothing under it is a line that says nothing.
    """

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),
        Binding("enter", "onward", "next", priority=True),
    ]

    def __init__(self, key: str, value: str, board: Board) -> None:
        """Initializes the writing.

        Args:
          key: The line being changed, or "" for one being put up now -- which is what makes
            this ask for a name first.
          value: What it says now, which is what is offered to change.
          board: The board it goes on, so a name already taken can be said about here rather
            than found on the way out.
        """
        super().__init__()
        self._key = key
        self._value = value
        self._board = board

        self._naming = not key
        self._typed = key if self._naming else value
        self._said = ""

    def _ask(self) -> None:
        """Says what a line of the board is, and takes what is typed from here on."""
        self.query_one("#asked", Label).update(
            "A line of the board" if self._naming else f"{escape(self._key)}"
        )
        self.query_one("#about", Label).update(
            "What you and the flow both write on. The flow reads it whenever it likes and "
            "changes nothing you are typing; nothing here waits on the flow either."
        )
        self._fill()
        self.query_one("#choices", OptionList).focus()

    def _fill(self) -> None:
        """Puts up what has been typed, which is the whole of what this sheet shows."""
        listing = self.query_one("#choices", OptionList)
        asked = "what to call it" if self._naming else "what it says"
        listing.set_options(
            [
                Option(
                    f"{_INDENT}[$text-muted]{asked}[/]  "
                    f"[$secondary]{escape(self._typed)}[/][reverse] [/reverse]",
                    id="=typed",
                )
            ]
        )
        listing.highlighted = 0
        self._drawn = 0
        self.query_one("#tuning", Label).update(
            f"[$text-muted]{self._said}[/]" if self._said else ""
        )
        self.query_one("#keys", Label).update(
            "Enter for what it says · Esc to go back"
            if self._naming
            else "Enter to write it down · Esc to go back"
        )

    def on_key(self, event: events.Key) -> None:
        """Takes every printable key as what is being typed, which is what this sheet is.

        Not a search that has to be asked for, as it is on a sheet of choices: there is
        nothing here to choose between, so the letters have nowhere else to go.

        Args:
          event: The key.
        """
        if event.key == "backspace":
            self._typed = self._typed[:-1]
        elif event.is_printable and event.character:
            self._typed += event.character
        else:
            return
        event.prevent_default()
        event.stop()
        self._said = ""
        self._fill()

    def action_onward(self) -> None:
        """Takes the name and asks what it says, or writes the line down."""
        if not self._naming:
            self.dismiss((self._key, self._typed))
            return
        named = self._typed.strip()
        if not named:
            self._said = "a line of the board is named"
            self._fill()
            return
        held = self._board.held(named)
        if held is not None and held.whose == FLOW:
            self._said = f"{escape(named)} is the flow's to change, not yours"
            self._fill()
            return
        self._key, self._naming = named, False
        self._typed = held.value if held is not None else self._value
        self._said = ""
        self.query_one("#asked", Label).update(escape(self._key))
        self._fill()

    @on(OptionList.OptionSelected)
    def _took(self, _event: OptionList.OptionSelected) -> None:
        """A click on the one row means what enter means, there being one thing to do."""
        self.action_onward()

class Status(Sheet[str]):
    """How the run is going, and the shape of it: who is working, and who handed to whom.

    Which is where the column that used to sit beside the transcript went. What a flow is
    doing is worth a look now and then and not worth a fifth of the screen the whole time:
    the transcript is what is being read, and the column was taking width off it to say
    something that mostly had not changed since the last glance.

    It is also where an agent is picked out to be read. tab steps between the ones working,
    which is what somebody wants while several are thinking; this is the whole flow drawn at
    once, so it is where the one that has stopped is reached. Enter or a click on a box
    answers with that agent, and the first row is the transcript every agent's work appears
    on, which is the way back to watching the flow.

    And it is where the board is, for a flow that talks to the person: the lines they and the
    flow both write on, under the diagram, changed here and read by the flow whenever it
    likes. It belongs beside how far through the run is rather than on a sheet of its own --
    a board somebody has to go and open is a board nobody reads.

    Redrawn while it is open, since what it is about moves without anybody touching it.
    """

    LETTERS: ClassVar = frozenset({"adding", "drop"})

    BINDINGS: ClassVar = [
        ("escape", "back", "back"),
        Binding("a", "adding", "add a line", priority=True),
        Binding("d", "drop", "take a line away", priority=True),
    ]

    def __init__(
        self,
        flow: str,
        named: tuple[str, ...],
        models: list[Runs],
        monitor: Monitor,
        config: BaseModel | None = None,
        drawn: Callable[[], Sequence[Drawn]] = tuple,
        reading: str = EVERY,
        board: Callable[[], Board | None] = lambda: None,
    ) -> None:
        """Reads one run.

        Args:
          flow: The flow being run.
          named: What that flow calls each agent it drives, "" apiece where it names none.
          models: What each of its agents runs, and where its turns land.
          monitor: The run itself, read again each time this is redrawn.
          config: What the flow was set up with, for a flow that takes any setting up.
          drawn: Asked, each time this is redrawn, for the agents there are to read, in the
            order the flow takes them. Asked rather than given, because the answer moves
            while the sheet is open: an agent appears as its first turn starts, which is what
            makes this a picture of the run growing.
          reading: Which transcript is on the screen behind this, so the diagram can say so.
          board: Asked for what the flow and whoever is at the prompt both write on, or for
            None on a run whose flow does not talk to the person. Asked rather than given, as
            the boxes are: a run may start while this is open. It is here rather than on a
            sheet of its own because it is about the run -- what there is to do next belongs
            beside how far through the run is, and a board somebody has to go and open is a
            board nobody reads.
        """
        super().__init__()
        self._flow = flow
        self._named = named
        self._models = models
        self._monitor = monitor
        self._config = config

        self._boxing = drawn
        self._boxes: list[Drawn] = list(drawn())
        self._reading = reading
        self._boarding = board
        
        self._said = ""

        self._was = reading

        self._shown: list[str] = []

    def _ask(self) -> None:
        """Says what this is, puts the flow up, and starts redrawing."""
        self.query_one("#asked", Label).update("Status")
        self.query_one("#about", Label).update(
            "How the run is going, and the shape of it. Enter reads an agent -- whether or "
            "not it is working, which is what tab is held to."
        )
        self._fill()
        self.query_one("#choices", OptionList).focus()
        self.set_interval(_LIVE, self._fill)

    def _fill(self) -> None:
        """Puts the flow up as it stands, keeping the cursor where it was.

        Put up again rather than adjusted, because everything on it moves: an agent starts a
        turn, a handover happens, a token is spent. The cursor is held by agent rather than
        by row so that it stays on the same box while that happens.
        """
        listing = self.query_one("#choices", OptionList)
        at = listing.highlighted
        if at is not None and 0 <= at < listing.option_count:
            self._was = str(listing.get_option_at_index(at).id or "")

        self._boxes = list(self._boxing())
        working = sum(1 for one in self._boxes if one.working)
        rows = [
            Option(
                f"{_INDENT}[{'$secondary' if working else '$text-muted'}]▣[/] every agent"
                f"{_DOT}[$text-muted]"
                f"{working} of {len(self._boxes)} working[/]"
                + (f"{_DOT}[$text-muted]reading[/]" if self._reading == EVERY else ""),
                id=EVERY,
            )
        ]
        shape = self._monitor.shape()
        rows += [
            Option("\n".join(block), id=one.who)
            for one, block in zip(
                self._boxes, diagram(self._boxes, shape, self.size.width), strict=True
            )
        ]
        rows += self._lines()
        drawn = [str(row.prompt) for row in rows]
        if drawn != self._shown:
            self._shown = drawn
            listing.clear_options()
            listing.add_options(rows)
            listing.highlighted = next(
                (one for one, row in enumerate(rows) if str(row.id or "") == self._was),
                0,
            )
            self.shortens()

        self._says(shape)

    def _says(self, shape: Shape) -> None:
        """Puts what the run has come to under the diagram: what it is, and what it has cost.

        Args:
          shape: The run as a graph, taken at the same moment the boxes were.
        """
        over = (self._monitor.until or time.monotonic()) - self._monitor.began
        spending = self._monitor.spending()

        groups: list[list[tuple[str, list[str]]]] = [
            [
                ("Flow", _flowing(self._flow)),
                ("Agents", reads(self._named, self._models) or ["none installed"]),

                ("Set", [escape(one) for one in setting(self._config)]),
            ],
            [
                (
                    "Working",
                    [short(who) for who in sorted(shape.working)]
                    or ["[$text-muted]nobody[/]"],
                ),
                ("Running", [f"{over:.0f}s"]),
                
                ("Also", elsewhere(self._boxes, shape)),
            ],
            [
                (
                    "Tokens",
                    [
                        f"{escape(spend.model):<26}{thousands(spend.tokens):>8}"
                        f"   [$text-muted]{spend.rate:.0f}/s[/]"
                        for spend in spending
                    ]
                    or ["[$text-muted]nothing spent yet[/]"],
                ),
            ],
        ]
        lines: list[str] = []
        for group in groups:
            for field, values in group:
                for at, value in enumerate(values):

                    head = f"{field}:" if at == 0 else ""
                    lines.append(f"[$text-muted]{head:<{_FIELD}}[/]{value}")
            lines.append("")
        if self._said:
            lines.append(f"[$text-muted]{self._said}[/]")
        self.query_one("#tuning", Label).update("\n".join(lines))
        self.query_one("#keys", Label).update(
            f"↑↓ move{_DOT}enter read{_DOT}esc close"
            if self._boarding() is None
            else f"↑↓ move{_DOT}enter read or change{_DOT}a adds a line"
            f"{_DOT}d twice takes one away{_DOT}esc close"
        )

    def _lines(self) -> list[Option]:
        """The board, as the rows under the diagram.

        Returns:
          A heading and one row per line, or nothing at all for a run whose flow does not
          talk to the person -- there being no board on one nobody can write to.
        """
        board = self._boarding()
        if board is None:
            return []
        held = board.items()
        rows = [
            Option(
                f"{_INDENT}[$primary]Board[/]"
                f"{_DOT}[$text-muted]what you and the flow both write on[/]",
                id=_BOARDED,
                disabled=True,
            )
        ]
        if not held:
            return [
                *rows,
                Option(
                    f"{_INDENT}  [$text-muted]nothing on it yet; a puts a line up[/]",
                    id=f"{_BOARDED}{_BOARDED}",
                    disabled=True,
                ),
            ]
        room = max(24, min(_WIDEST, self.size.width - len(_INDENT) - 5) - _LABEL - 6)

        rows += [
            Option(
                f"{_INDENT}  [{'$text-muted' if one.whose == FLOW else '$secondary'}]"
                f"{_ON_IT}[/] {escape(named)}{' ' * max(0, _LABEL - len(named))}"
                f"[$text-muted]{escape(_fits(one.value or one.about, room))}[/]"
                + (
                    f"{_DOT}[$text-muted]{one.whose}'s[/]"
                    if one.whose != ANYONE
                    else ""
                ),
                id=f"{_ON_BOARD}{one.key}",
            )
            for one in held
            if (named := _fits(one.key, _LABEL))
        ]
        return rows

    def action_adding(self) -> None:
        """Puts a line on the board, which is what somebody with more to say does."""
        if self._boarding() is None:
            self._said = "this flow does not talk to you, so there is no board on it"
            self._fill()
            return
        self._writes("")

    def action_drop(self) -> None:
        """Takes the line under the cursor off the board, once d has been pressed twice."""
        named = self.under()
        board = self._boarding()
        if board is None or not named.startswith(_ON_BOARD):
            return
        key = named.removeprefix(_ON_BOARD)
        held = board.held(key)
        if held is None:
            return
        if held.whose == FLOW:
            self._said = f"{escape(key)} is the flow's to change, not yours"
            self._fill()
            return
        if not self._armed(named):
            self._said = f"press d again to take {escape(key)} off the board"
            self._fill()
            return
        board.drop(key, by=USER)
        self._said = f"{escape(key)} is off the board"
        self._shown = []  
        self._fill()

    @on(OptionList.OptionSelected)
    def _took(self, event: OptionList.OptionSelected) -> None:
        """Reads an agent, or changes a line of the board: whichever row this was."""
        named = str(event.option.id or EVERY)
        if not named.startswith(_ON_BOARD):
            self.dismiss(named)
            return
        key = named.removeprefix(_ON_BOARD)
        board = self._boarding()
        held = board.held(key) if board is not None else None
        if held is not None and held.whose == FLOW:
            self._said = f"{escape(key)} is the flow's to change, not yours"
            self._fill()
            return
        self._writes(key)

    @work
    async def _writes(self, key: str) -> None:
        """Asks what a line is to say, and writes it down.

        Args:
          key: Which line, or "" for one being put up now.
        """
        board = self._boarding()
        if board is None:
            return
        showing = cast(
            "App[None]",
            self.app,  
        )
        held = board.held(key) if key else None
        said = await showing.push_screen_wait(
            Entry(key, held.value if held is not None else "", board)
        )
        if said is None:
            return  
        named, value = said
        try:
            board.put(named, value, by=USER)
        except (PermissionError, ValueError) as why:
            self._said = escape(str(why))
        else:
            self._said = f"{escape(named)} is on the board"
        self._shown = []  
        self._fill()
