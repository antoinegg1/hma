"""The skills a flow brings with it, and where they are once they have been fetched.

A flow is a directory, and `skills/` inside it is the skills that flow works by -- laid out
the way every one of these CLIs lays a skill out, one directory apiece with a `SKILL.md` in
it. They travel with the flow: fork it, edit one, and the next run is driven by the edited
one, which is the whole point of a flow being a directory rather than a file.

A flow may also name skills that live somewhere else, by writing them where it is declared::

    @flow(skills=("/path/to/local/flows#example",))

which is a git repository anything can clone and, after the `#`, which of the skills in it is
wanted -- matched against the `skills/*` that repository holds, by the directory each is in.
Without one, every skill that repository holds is brought. A repository is cloned once into
`~/.humanize/skills/<name>/` and fetched again the next time a run asks for it, so a skill
somebody else maintains is a skill that keeps up.

Nothing here installs anything. What is fetched is put where humanize keeps it, and what a
session does with it is :func:`hmz.agents.skills.mount`: put where that backend reads a
project's own skills for as long as the session lives, and taken away again after. The skills
the person at this machine installed are untouched, being theirs.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from hmz import home
from hmz.agents.skills import CARD, SKILLS, Loaded

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["CARD", "SKILLS", "brought", "cached", "fetched", "under"]

_WANTED = "#"

def under() -> Path:
    """Where the skills fetched from somewhere else are kept, under humanize's own home."""
    return home() / SKILLS

def brought(at: Path | str, declared: Iterable[str] = ()) -> list[Loaded]:
    """Every skill one flow brings: its own first, then whatever it named.

    Args:
      at: The flow's own directory, which is where its `skills/` is, or "" for a flow that is
        one file and so has none of its own.
      declared: What it named where it was declared -- a git URL apiece, each with an
        optional `#<skill>` saying which of that repository's skills is wanted.

    Returns:
      One per skill, the flow's own in the order they are on disk and the fetched ones in the
      order they were named. A name declared twice is the first of them: the flow's own beats
      a repository's, since a fork that edited a skill meant the edited one.

    Raises:
      OSError: If a repository cannot be fetched, or holds no skill of the name a flow asked
        it for. Said where the flow is being got ready rather than left for the first turn: a
        flow that works by a skill it has not got is not a flow to start and find out about
        an hour in.
    """
    found: list[Loaded] = []
    seen: set[str] = set()

    for one in _inside(Path(at) / SKILLS) if at else []:
        seen.add(one.name)
        found.append(Loaded(one.name, one, "this flow"))
    for said in declared:
        url, _, wanted = said.partition(_WANTED)
        if not url.strip():
            continue
        where = fetched(url.strip())
        inside = _inside(where / SKILLS)
        wanted = wanted.strip()
        if wanted and not any(one.name == wanted for one in inside):

            raise OSError(
                f"{url.strip()} holds no skill called {wanted!r}"
                + (
                    f"; it holds {', '.join(one.name for one in inside)}"
                    if inside
                    else ""
                )
            )
        for one in inside:
            if (wanted and one.name != wanted) or one.name in seen:
                continue
            seen.add(one.name)
            found.append(Loaded(one.name, one, said))
    return found

def _inside(at: Path) -> list[Path]:
    """Every skill directory inside one, alphabetically.

    Args:
      at: A `skills/` directory, which may not be there at all.

    Returns:
      One per directory holding a `SKILL.md`, which is what makes a directory a skill.
      Nothing at all where there is no such place -- a flow that brings none has no `skills/`,
      which is not a thing to raise about.
    """
    try:
        return sorted(one for one in at.iterdir() if (one / CARD).is_file())
    except OSError:
        return []

def cached(url: str) -> Path:
    """Where the repository at this URL is kept once it has been fetched.

    Args:
      url: The repository, as a flow named it.

    Returns:
      The directory, whether or not anything has been fetched into it. Named after the
      repository and the owner above it, so that whoever looks in there can see what is
      there -- and ended with a digest of the whole URL, because those two names are not
      unique: `acme/skills` on one host and `acme/skills` on another are two repositories,
      and one directory for both is a flow silently working by somebody else's skills.
    """
    said = PurePosixPath(url.rstrip("/"))
    name = said.name.removesuffix(".git") or "skills"
    whose = said.parent.name
    
    plain = [one for one in (_safe(whose), _safe(name)) if one]
    digest = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:12]
    return under() / "-".join([*plain, digest])

def _safe(said: str) -> str:
    """One part of a URL, as much of it as may be a directory name."""
    return "".join(one for one in said if one.isalnum() or one in "._-").strip(".-")

def fetched(url: str) -> Path:
    """Clones a repository of skills, or brings the clone of it up to date.

    Args:
      url: Where it is, as git takes it.

    Returns:
      The directory it was fetched into.

    Raises:
      OSError: If git is not there, or the fetch failed. What git said is attached, and a
        clone that failed leaves nothing behind to be taken for a fetched one.
    """
    from .verses import clone, refresh

    at = cached(url)
    if (at / ".git").exists():
        try:
            refresh(at)
        except OSError:

            return at
        return at
    at.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(at, ignore_errors=True)  

    beside = Path(tempfile.mkdtemp(dir=at.parent, prefix=f".{at.name}."))
    beside.rmdir()  
    try:
        clone(url, beside)
        try:
            beside.rename(at)
        except OSError:
            
            shutil.rmtree(beside, ignore_errors=True)
    except BaseException:
        shutil.rmtree(beside, ignore_errors=True)
        raise
    return at
