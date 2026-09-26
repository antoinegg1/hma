"""What humanize reports about itself when something goes wrong, and what it never reports.

humanize is early. A crash on somebody else's machine is a crash nobody here sees, and an
interaction that reads as obvious to whoever wrote it and as nonsense to whoever met it is not
a crash at all -- so both are reported, and both are answered for once, by hand, on the first
start.

Two rules shape everything here.

**Nothing is sent that a person would be surprised by.** What a flow was told, what an agent
said back, what is in a file, what is in an account: none of it leaves this machine, and none
of it is reachable from what does. What is sent is the shape of a failure -- the exception and
where in humanize it happened -- and the shape of the run it happened in: which flow, which
backends at which models, which accounts by name, which skills by name. Names and never
values, counts and never contents. The three switches that would upload the rest are off and
say why where they are set.

**Nothing is sent that nobody said yes to.** The setting is `enable_sentry` and it has three
answers rather than two: on, off, and the absence that means nobody has been asked yet. Only
the interface asks, because only the interface has somebody to ask; a headless run reports if
the answer is already yes and is silent otherwise. `HUMANIZE_SENTRY=on|off` answers for one
process without writing anything down, which is what a scripted install and this suite use.

A leaf: it names nothing of humanize but the settings it reads and the home they are in, so
every layer may report and none of them has to be reached into to do it. What goes with a
report is the layers' own to say, and each says it by handing over a callable -- see
:func:`about` -- which is only ever run when a report is actually being sent.
"""

from __future__ import annotations

import contextlib
import os
import re
import sysconfig
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "SENT",
    "about",
    "again",
    "asked",
    "crash",
    "enabled",
    "held",
    "snag",
    "start",
    "stop",
]

DSN = ""

SAYS = "HUMANIZE_SENTRY"

SENT = (
    "the error and where in humanize it happened",
    "which flow was running, and what each of its agents was set up to run",
    "which coding agents are installed here, and which accounts exist by name",
    "which skills and flowverses are in play, by name",
    "what humanize did that you then undid, refused or walked away from",
    "the version of humanize, of Python, and the kind of machine this is",
)

KEPT = (
    "nothing you typed: no task, no prompt, no line at the prompt",
    "nothing an agent said, and nothing out of any transcript or session log",
    "no file, no path outside humanize itself, and no directory name",
    "no key, no token and no account credential -- not even the names of the variables",
)

_HOME = re.compile(r"/(?:home|homes|Users)/[^/\s:'\"]+")
_KEYS = re.compile(
    r"\b(?:sk|pk|ghp|gho|github_pat|xai|sk-ant|sk-proj)[-_][A-Za-z0-9_\-]{8,}\b"
)
_CREDS = re.compile(r"(?<=//)[^/\s@]+(?=@)")
_LONG = 500

_RAN = re.compile(r"Command\s+'.*?'\s+(?=returned|timed out|died)", re.DOTALL)

_ROOTS = tuple(
    at
    for at in dict.fromkeys(
        [Path(__file__).resolve().parent.parent]
        + [
            Path(said).resolve()
            for said in (
                sysconfig.get_paths().get("purelib"),
                sysconfig.get_paths().get("platlib"),
                sysconfig.get_paths().get("stdlib"),
            )
            if said
        ]
    )
)

_THEIRS = "<not humanize>"

_ABOUT: dict[str, Callable[[], object]] = {}
_TELLING = threading.Lock()

_started: list[bool] = []

_answered: list[bool | None] = []

def enabled() -> bool | None:
    """Telemetry is disabled in the anonymous HMA package."""
    return False

def again() -> None:
    """Forgets what was read, for whoever has just written it down."""
    _answered.clear()

def asked(*, enable_sentry: bool) -> None:
    """Writes down the answer, which is asked once and holds wherever humanize is run.

    It also takes effect now. An answer of no from somebody who has been reporting all
    session is somebody saying stop, and a reporter that went on until the next start would
    be answering a question that was not asked.

    Args:
      enable_sentry: What was answered.
    """
    from hmz.settings import Settings

    Settings().answers(enable_sentry=enable_sentry)
    again()
    if enable_sentry:
        start()
    else:
        stop()

def start() -> bool:
    """Starts reporting, if it is on and has not been started already.

    Returns:
      Whether anything is reporting from here on. False for an answer that is no, for a
      machine nobody has been asked on, and for an SDK that will not start -- none of which
      is a reason to stop: humanize's own failures being unreported is a smaller thing than
      humanize not running.
    """
    if _started or not enabled():
        return bool(_started)
    try:
        import sentry_sdk
        from sentry_sdk.integrations.argv import ArgvIntegration
    except ImportError:  
        return False
    try:
        sentry_sdk.init(
            dsn=DSN,

            send_default_pii=False,

            include_local_variables=False,

            enable_logs=False,

            server_name="",
            traces_sample_rate=1.0,
            profile_session_sample_rate=1.0,

            profile_lifecycle="trace",

            disabled_integrations=[ArgvIntegration()],
            release=_version(),
            before_send=_before_send,
            before_send_transaction=_before_send,
        )
    except Exception:  
        return False
    _started.append(True)
    return True

def stop() -> None:
    """Stops reporting for the rest of this process, whatever was started earlier.

    What is already on its way is left to go: a report of something that had already happened
    is not something an answer given afterwards can recall. Everything after this is silent,
    and `start()` would have to be asked again.
    """
    if not _started:
        return
    _started.clear()
    try:
        import sentry_sdk
    except ImportError:  
        return

    with contextlib.suppress(Exception):
        sentry_sdk.get_client().close(timeout=1.0)
    with contextlib.suppress(Exception):
        sentry_sdk.init(dsn="")

def about(name: str, said: Callable[[], object]) -> None:
    """Says what to attach to a report, by handing over something that knows.

    The layers that know what a run is are above this one and must stay there, so what is
    attached is a callable rather than a value: a flow says how to describe itself, an
    interface says how to describe the machine, and neither is asked until a report is
    actually being sent. Which also means nothing is gathered on a machine that reports
    nothing.

    Args:
      name: What the attachment is called, which is the filename it arrives under.
      said: What to call for it. Anything it answers with is written as YAML and put through
        the same scrubbing everything else is. It must not raise; one that does is left out
        of the report rather than taking the report with it.
    """
    with _TELLING:
        _ABOUT[name] = said

def held() -> dict[str, object]:
    """Everything registered, asked now, for whoever is about to send or show a report.

    Returns:
      One entry per thing that answered, by name. A caller that raises is left out: a report
      that could not describe the run is still a report worth having.
    """
    with _TELLING:
        asking = dict(_ABOUT)
    found: dict[str, object] = {}
    for name, said in asking.items():
        try:
            found[name] = said()
        except Exception:  
            continue
    return found

def crash(why: BaseException, **said: object) -> None:
    """Reports one failure, with everything the layers said about the run it happened in.

    Args:
      why: What went wrong.
      said: Anything else worth knowing, as short strings: what was being done, and by which
        part of humanize.
    """
    if not start():
        return
    import sentry_sdk

    with sentry_sdk.isolation_scope() as scope:
        for name, value in said.items():
            scope.set_tag(name, _plainer(str(value)))
        _attaches(scope)
        sentry_sdk.capture_exception(why)

def snag(name: str, **said: object) -> None:
    """Reports something that is not a failure and is not what anybody meant either.

    A key that did nothing, a menu answered and then thrown away, a line refused, a run
    stopped seconds after it started: none of it is an error, and all of it is somebody
    finding out that humanize does not work the way they expected. That is the half of the
    feedback a stack trace never carries, and on an early tool it is the more useful half.

    Args:
      name: What happened, as one hyphenated word -- `dead-key`, `changes-dropped`.
      said: What is worth knowing about it: counts, which sheet, which key. Never what was
        typed, and never what anything was called by anybody but humanize.
    """
    if not start():
        return
    import sentry_sdk

    with sentry_sdk.isolation_scope() as scope:
        scope.set_tag("snag", name)
        for held_, value in said.items():
            scope.set_tag(held_, _plainer(str(value)))
        _attaches(scope)
        sentry_sdk.capture_message(f"snag: {name}", level="warning")

def _attaches(scope: Any) -> None:
    """Puts what the layers said about the run onto one report.

    Args:
      scope: The scope the report is being made on.
    """
    import yaml

    for name, value in held().items():
        try:
            written = yaml.safe_dump(
                _plainly(value), sort_keys=False, allow_unicode=True
            )
        except yaml.YAMLError:
            continue  
        scope.add_attachment(bytes=written.encode("utf-8"), filename=f"{name}.yaml")

def _plainly(said: object) -> object:
    """One thing to attach, with every string in it put through the scrubbing.

    Value by value rather than over the document: a document scrubbed as one string is a
    document that can be cut in half by the length limit, and half a YAML file is a file
    nobody can read.

    Args:
      said: Whatever a layer answered with.

    Returns:
      The same, with the strings in it plainer.
    """
    if isinstance(said, str):
        return _plainer(said)
    if isinstance(said, dict):
        return {
            str(_plainer(str(name))): _plainly(value)
            for name, value in cast("dict[object, object]", said).items()
        }
    if isinstance(said, (list, tuple)):
        return [_plainly(one) for one in cast("list[object]", said)]
    return said

def _before_send(event: Any, hint: Any) -> Any:
    """The last thing every report goes through, whoever made it.

    Belt as well as braces: the settings above already say that no frame carries its
    variables and that nothing about this machine or the person at it is attached, and this
    takes them off again -- an SDK that grows a new way of collecting one of them should find
    it taken away here rather than sent.

    Args:
      event: The report, as the SDK built it.
      hint: What it was built from, which is not read.

    Returns:
      The report to send, with what must not leave taken out of it.
    """
    del hint

    for gone in ("server_name", "user", "request", "modules", "extra"):
        event.pop(gone, None)
    for one in event.get("exception", {}).get("values", []):
        for frame in one.get("stacktrace", {}).get("frames", []):
            frame.pop("vars", None)
            frame.pop("pre_context", None)
            frame.pop("post_context", None)
            frame.pop("context_line", None)
            _framed(frame)
        one["value"] = _plainer(str(one.get("value") or ""))

    event.pop("breadcrumbs", None)
    return event

def _framed(frame: dict[str, Any]) -> None:
    """One frame of a traceback, named the way a report may name it.

    A frame in humanize or in something humanize is installed beside is named by where it is
    under that root -- `hmz/agents/base.py`, `textual/app.py` -- which is what makes the report
    worth reading and says nothing about the machine it came off. A frame in anything else is a
    file of theirs: their flow, in their project, under names they chose. Its line number stays
    and the rest of it goes, because a directory named after the work is exactly what the
    promise above says humanize does not take.

    Args:
      frame: The frame, changed in place.
    """
    said = str(frame.get("abs_path") or frame.get("filename") or "")
    where = _under(said)
    if where is None:
        frame["abs_path"] = frame["filename"] = _THEIRS
        
        frame.pop("module", None)
        frame.pop("function", None)
        return
    frame["abs_path"] = frame["filename"] = where
    if frame.get("module"):
        frame["module"] = _plainer(str(frame["module"]))

def _under(said: str) -> str | None:
    """One file, as the path under whichever root humanize knows it by.

    Args:
      said: The file, as the SDK found it.

    Returns:
      Its path under that root -- so `hmz/telemetry.py` rather than wherever humanize is
      installed -- or None for a file under no root humanize knows, which is somebody's own.
    """
    if not said:
        return None
    try:
        at = Path(said).resolve()
    except (OSError, ValueError):  
        return None
    for root in _ROOTS:
        if at.is_relative_to(root):
            return at.relative_to(root).as_posix()
    return None

def _plainer(said: str) -> str:
    """One string, with what must not leave a machine taken out of it.

    Args:
      said: Whatever was about to be sent.

    Returns:
      It, with home directories, credentials in URLs and anything shaped like a key replaced
      -- and cut short, since a long string in a report is a file somebody pasted.
    """
    said = _RAN.sub("A command ", said)
    said = _HOME.sub("~", said)
    said = _CREDS.sub("…", said)
    said = _KEYS.sub("…", said)
    return said if len(said) <= _LONG else said[:_LONG] + "…"

def _version() -> str:
    """What humanize this is, as the release a report is filed under."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return f"hmz@{version('hmz')}"
    except PackageNotFoundError:  
        return "hmz@unknown"
