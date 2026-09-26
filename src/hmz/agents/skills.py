"""The skills a session carries: the ones its CLI installs, and the ones a flow mounts.

Two halves, and the difference between them is who they belong to.

The first is a reading. A skill installed on this machine is the CLI's own -- installed the
way that CLI installs one and switched off the way that CLI switches one off -- so humanize
lists them so that whoever is setting an agent up can see what it will be carrying, and
changes nothing about any of them. Nothing is asked of the CLI itself, for the reason nothing
else here asks it either: starting one costs seconds a prompt does not have. Where each of
them looks is written down in :mod:`hmz.backends`, and reading a `SKILL.md` is reading its
front matter and stopping.

The second is a mounting. A flow brings the skills it works by, and every session its agents
open gets them: put where that backend reads a project's own skills for as long as the session
lives, and taken away again after. Mounted rather than installed, so a flow that works by three
skills is not a flow that leaves three skills on somebody's machine -- and per session, so a
flow rewritten between turns is a flow whose next session carries the rewritten one.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml

from hmz.backends import named

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "CARD",
    "SKILLS",
    "Loaded",
    "Mounted",
    "Skill",
    "mount",
    "skills",
    "unmount",
]

_FRONT = 4096

_FENCE = "---"
_ABOUT = 90

SKILLS = "skills"
CARD = "SKILL.md"

@dataclass(frozen=True, slots=True)
class Skill:
    """One skill a backend would load.

    Attributes:
      name: What the CLI knows it by, which is what a turn is told about: the name in its
        front matter, or the directory it is in where it states none -- which is the rule
        every one of these CLIs reads a skill by.
      about: The line it describes itself with, which is what says whether to switch it off.
      whose: Where it came from: yours, or this project's.
    """

    name: str
    about: str
    whose: str

def skills(backend: str, where: Path | str | None = None) -> list[Skill]:
    """The skills one backend would load here, the way that backend finds them.

    Args:
      backend: The CLI, by any name it answers to.
      where: The project whose own skills to include, defaulting to this directory.

    Returns:
      One entry per skill, yours before the project's and each set in the order it is on disk.
      Empty for a backend that is not known here, keeps none, or has none installed.
    """
    profile = named(backend)
    if profile is None:
        return []
    found: list[Skill] = []
    seen: set[str] = set()
    for root, globs, whose in (
        (profile.directory(), profile.skills, "yours"),

        (profile.configuration(), profile.config, "yours"),

        (Path.home(), profile.shared, "yours"),
        (Path(where or Path.cwd()), profile.works, "this project"),
    ):
        for pattern in globs:
            for path in sorted(root.glob(pattern)):
                skill = _skill(path, whose)

                if skill is not None and skill.name not in seen:
                    seen.add(skill.name)
                    found.append(skill)
    return found

def _skill(path: Path, whose: str) -> Skill | None:
    """One `SKILL.md`, read as the CLI reads it: its front matter, and nothing else.

    Args:
      path: The file.
      whose: Where it came from.

    Returns:
      The skill, or None for a file that cannot be read at all.
    """
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            head = stream.read(_FRONT)
    except OSError:
        return None
    front: Any = {}
    if head.startswith(_FENCE):
        _, _, rest = head.partition("\n")
        block, _, _ = rest.partition(f"\n{_FENCE}")
        with contextlib.suppress(yaml.YAMLError):
            front = yaml.safe_load(block)
    said = cast("dict[str, Any]", front) if isinstance(front, dict) else {}

    name = str(said.get("name") or path.parent.name).strip()
    about = " ".join(str(said.get("description") or "").split())
    return Skill(
        name=name,
        about=about[:_ABOUT] + ("…" if len(about) > _ABOUT else ""),
        whose=whose,
    )

@dataclass(frozen=True, slots=True)
class Loaded:
    """One skill a flow brings, to be mounted onto every session its agents open.

    Attributes:
      name: What it is called, which is the directory it is in -- and the name the CLI will
        know it by once it is there.
      at: Where it is now: inside the flow, or inside the clone of a repository the flow
        named. It is copied from there rather than pointed at, so that a session on another
        machine carries it too.
      whose: What brought it, for whoever is being shown what a flow works by: the flow
        itself, or the repository it was named in.
    """

    name: str
    at: Path
    whose: str = ""

@dataclass(frozen=True, slots=True)
class Mounted:
    """What one session put where its backend reads skills, to be taken away again.

    Attributes:
      at: The directories this session planted, which are the ones it takes away. A skill
        that was already there is not among them: a project's own skill of that name is the
        project's, and one already mounted by another session of the same flow belongs to
        that session until the last of them is done with it.
    """

    at: tuple[Path, ...] = ()

_PLANTED: dict[Path, tuple[Path, int]] = {}
_PLANTING = threading.Lock()

_MADE: set[Path] = set()

def mount(backend: str, workspace: Path | str, loaded: Iterable[Loaded]) -> Mounted:
    """Puts a flow's skills where one backend reads a project's own, for one session.

    Copied rather than linked: a session whose turns land on another machine reads the mirror
    of this directory, and a link into humanize's own home is a link that machine cannot
    follow. Copied afresh per session, which is what makes a skill edited between turns the
    skill the next session carries.

    Args:
      backend: The CLI, by any name it answers to.
      workspace: The directory the session works in, as that backend will see it.
      loaded: The skills to mount.

    Returns:
      What was planted, to be handed back to :func:`unmount`. Nothing at all for a backend
      that reads no such directory -- one whose skills are all its own -- and for a flow that
      brings none.
    """
    profile = named(backend)
    if profile is None or not profile.mounts:
        return Mounted()
    into = Path(workspace) / profile.mounts
    planted: list[Path] = []
    for one in loaded:
        at = into / one.name
        with _PLANTING:
            held = _PLANTED.get(at)
            if held is not None:
                whence, count = held
                if whence != one.at:

                    _clashed()
                    continue

                _PLANTED[at] = (whence, count + 1)
                planted.append(at)
                continue
            if at.exists():

                _clashed()
                continue

            making = [one for one in (into.parent, into) if not one.exists()]
            if not _copied(one.at, at):

                continue
            _MADE.update(making)
            _PLANTED[at] = (one.at, 1)
            planted.append(at)
    return Mounted(tuple(planted))

def _copied(skill: Path, at: Path) -> bool:
    """Copies one skill into place whole, or leaves nothing of it where it could not.

    Beside it and then moved into place, because a copy that stops partway -- a disk that
    filled, a file that could not be read -- would leave a directory that is not a skill where
    a skill goes. Nothing here would ever remove it: it is not in the table of what was
    planted, so from then on every session reads it as one the project owns and mounts its own
    over it never.

    Args:
      skill: Where the skill is now.
      at: Where it is to be.

    Returns:
      Whether it is there.
    """
    at.parent.mkdir(parents=True, exist_ok=True)
    try:
        beside = Path(tempfile.mkdtemp(dir=at.parent, prefix=f".{at.name}."))
    except OSError:
        return False
    try:
        shutil.copytree(skill, beside, dirs_exist_ok=True)
        beside.rename(at)
    except OSError:
        shutil.rmtree(beside, ignore_errors=True)
        return False
    return True

def _clashed() -> None:
    """Says that a skill a flow brought is not the skill of that name the session will read.

    Not an error -- there is a skill of that name there and the turn runs with it -- and not
    what whoever wrote the flow meant either, which is the half of the feedback a stack trace
    never carries. That it happened is the whole of what is said: what the skill is called is
    a word out of somebody's own flow, and those do not leave the machine.
    """
    from hmz import telemetry

    telemetry.snag("skill-name-taken")

def unmount(one: Mounted) -> None:
    """Takes away what a session mounted, once no other session is holding it.

    Args:
      one: What :func:`mount` answered with. Called more than once for a session that is
        closed and then collected, which is why it counts down rather than deletes.
    """
    for at in one.at:
        with _PLANTING:
            held = _PLANTED.get(at)
            if held is None:
                continue
            whence, count = held
            if count > 1:
                _PLANTED[at] = (whence, count - 1)
                continue
            shutil.rmtree(at, ignore_errors=True)
            if at.exists():

                _PLANTED[at] = (whence, 0)
                continue
            del _PLANTED[at]

    with _PLANTING:
        for empty in sorted(_MADE, reverse=True):
            try:
                empty.rmdir()
            except OSError:
                continue  
            _MADE.discard(empty)
