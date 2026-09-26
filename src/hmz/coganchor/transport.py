"""Getting a :class:`~hmz.coganchor.proto.Channel` to ``serve`` on the target.

Four ways in:

``ssh://[user@]host[:port]``
    Ship a self-contained zipapp of coganchor to the host and run its ``serve``
    side over the ssh pipe.  Nothing needs to be installed there beyond
    Python 3.
``docker://container``
    The same, into a running container, over ``docker exec``.  A container is a
    machine like any other here; it needs no port, no secret and no cooperation
    beyond a ``python3``.
``tcp://host:port``
    Attach to an ``hmz anchor serve --listen`` someone already started.
``local[:REAL]``
    Run ``serve`` as a child process on this machine.  Used for development and
    by the test suite, where ``REAL`` is the directory standing in for the
    target's copy of the workspace.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zipapp
from dataclasses import dataclass
from pathlib import Path

from hmz import coganchor
from hmz.coganchor.proto import Channel

__all__ = ["Target", "Transport", "build_bundle", "connect"]

log = logging.getLogger(__name__)

REMOTE_CACHE = "~/.cache/humanize"

_INSTALL = (
    "if [ ! -s {file} ]; then cat > {file}.part && mv {file}.part {file}; "
    "else cat > /dev/null; fi"
)

_SSH_OPTIONS = ("-T", "-o", "BatchMode=no", "-o", "ServerAliveInterval=30")

@dataclass(frozen=True, slots=True)
class Target:
    """Where the target is."""

    scheme: str
    host: str = ""
    port: int = 0
    path: str = ""

    @classmethod
    def parse(cls, spec: str) -> Target:
        if spec == "local" or spec.startswith("local:"):
            _, _, path = spec.partition(":")
            return cls("local", path=path)
        if spec.startswith("ssh://"):
            authority = spec[len("ssh://") :]
            host, _, port = authority.rpartition(":")
            if host and port.isdigit():
                return cls("ssh", host=host, port=int(port))
            return cls("ssh", host=authority)
        if spec.startswith("docker://") and (container := spec[len("docker://") :]):
            return cls("docker", host=container)
        if spec.startswith("tcp://"):
            host, _, port = spec[len("tcp://") :].rpartition(":")
            if not host or not port.isdigit():
                raise ValueError(f"malformed target {spec!r}; expected tcp://HOST:PORT")
            return cls("tcp", host=host, port=int(port))
        raise ValueError(
            f"unsupported target {spec!r}; expected ssh://HOST, docker://CONTAINER, "
            "tcp://HOST:PORT or local[:PATH]"
        )

    def describe(self) -> str:
        if self.scheme == "ssh":
            return f"ssh://{self.host}" + (f":{self.port}" if self.port else "")
        if self.scheme == "docker":
            return f"docker://{self.host}"
        if self.scheme == "tcp":
            return f"tcp://{self.host}:{self.port}"
        return f"local{':' + self.path if self.path else ''}"

@dataclass(slots=True)
class Transport:
    """An open channel plus whatever process is keeping it alive."""

    channel: Channel
    process: subprocess.Popen[bytes] | None = None

    def close(self) -> None:
        self.channel.close()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

                self.process.wait()

def connect(target: Target, exports: list[str], token: str | None = None) -> Transport:
    """Open a channel to the ``serve`` side described by ``target``."""
    if target.scheme == "tcp":
        return _connect_tcp(target)
    if target.scheme == "ssh":
        return _connect_ssh(target, exports, token)
    if target.scheme == "docker":
        return _connect_docker(target, exports)
    return _connect_local(target, exports, token)

def _connect_tcp(target: Target) -> Transport:
    sock = socket.create_connection((target.host, target.port), timeout=30.0)
    sock.settimeout(None)
    return Transport(Channel.from_socket(sock))

def _connect_local(target: Target, exports: list[str], token: str | None) -> Transport:  
    command = [
        sys.executable,
        "-m",
        "hmz",
        "anchor",
        "serve",
        "--stdio",
        *_export_args(exports),
    ]
    return _spawn(command, token)

def _connect_ssh(target: Target, exports: list[str], token: str | None) -> Transport:
    payload = build_bundle().read_bytes()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    remote_file = f"{REMOTE_CACHE}/humanize-{digest}.pyz"
    ssh = [
        "ssh",
        *_SSH_OPTIONS,
        *(["-p", str(target.port)] if target.port else []),
        target.host,
    ]

    upload = f"mkdir -p {REMOTE_CACHE} && " + _INSTALL.format(file=remote_file)
    result = subprocess.run(
        [*ssh, upload], input=payload, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise ConnectionError(
            f"could not install humanize on {target.host}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )

    remote_command = " ".join(
        [
            "exec",
            "python3",
            remote_file,
            "anchor",
            "serve",
            "--stdio",
            *_export_args(exports, quote=True),
        ]
    )
    return _spawn([*ssh, remote_command], token)

def _connect_docker(target: Target, exports: list[str]) -> Transport:
    """Serve from inside a running container, over ``docker exec``.

    The bundle is pushed the way ``ssh://`` pushes it, into the container's ``/tmp`` rather than
    a home directory it may not have.  The exec inherits the container's own user and working
    directory, so a container is served as whoever it runs as.
    """
    payload = build_bundle().read_bytes()
    digest = hashlib.sha256(payload).hexdigest()[:16]

    remote_file = f"/tmp/humanize-{digest}.pyz"  
    exec_in = ["docker", "exec", "-i", target.host]

    result = subprocess.run(
        [*exec_in, "sh", "-c", _INSTALL.format(file=remote_file)],
        input=payload,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ConnectionError(
            f"could not install humanize in {target.host}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )

    command = [
        *exec_in,
        "python3",
        remote_file,
        "anchor",
        "serve",
        "--stdio",
        *_export_args(exports),
    ]
    return _spawn(command, None)

def _spawn(command: list[str], token: str | None) -> Transport:
    """Start a child that serves over its own stdin and stdout.

    The token travels in the environment the child inherits, which works for
    ``local``.  For ``ssh`` and ``docker`` the child is the client rather than
    the target and neither forwards the environment, so those sessions are
    authenticated by ssh and by the docker socket themselves and the token goes
    unused.
    """
    log.debug("starting the target: %s", " ".join(command))
    env = dict(os.environ)
    if token:
        env["HUMANIZE_TOKEN"] = token
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        env=env,
        close_fds=True,
    )
    assert process.stdin is not None  
    assert process.stdout is not None  
    return Transport(Channel(process.stdout, process.stdin), process)

def _export_args(exports: list[str], *, quote: bool = False) -> list[str]:
    """Build ``--export`` arguments, quoting them for a remote shell if needed.

    The ssh transport hands its command to the target's shell as one string, so
    a workspace path containing a quote or a space has to survive that; the
    local and docker transports pass argv straight through and must not be
    quoted.
    """
    args: list[str] = []
    for export in exports:
        args += ["--export", shlex.quote(export) if quote else export]
    return args

def build_bundle(destination: Path | None = None) -> Path:
    """Package coganchor, and the command line reaching it, as a zipapp for the target.

    The whole subpackage ships, tracer half included, because pruning it would
    be a list to keep in step with the source tree.  Nothing is lost by that:
    the target only ever runs ``anchor serve``, which :func:`hmz.cli.main` reaches
    without importing the modules that need ptrace or an x86-64 register map --
    nor any other subpackage, none of which is here -- so the bundle runs on a
    target of any architecture.  It is pure stdlib, so a host needs nothing but
    ``python3``.
    """
    if destination is None:
        destination = Path(tempfile.gettempdir()) / f"humanize-{os.getuid()}.pyz"
    with tempfile.TemporaryDirectory(prefix="humanize-bundle-") as staging:
        root = Path(staging)

        parts = coganchor.__name__.split(".")
        shutil.copytree(
            Path(coganchor.__file__).parent,
            root.joinpath(*parts),
            ignore=shutil.ignore_patterns("__pycache__", "*.md"),
        )
        for depth in range(1, len(parts)):

            init = root.joinpath(*parts[:depth]) / "__init__.py"
            init.write_text(
                f'"""{".".join(parts[:depth])}, cut down to {parts[depth]}."""\n'
            )

        package = Path(coganchor.__file__).parent.parent
        shutil.copytree(
            package / "cli",
            root.joinpath(*parts[:-1]) / "cli",
            ignore=shutil.ignore_patterns("__pycache__", "*.md"),
        )

        (root / "__main__.py").write_text(
            "from hmz.cli import main\n\nraise SystemExit(main())\n"
        )

        stamp = time.mktime((1980, 1, 2, 0, 0, 0, 0, 2, -1))
        for path in root.rglob("*"):

            path.chmod(0o755 if path.is_dir() else 0o644)
            os.utime(path, (stamp, stamp))

        handle, staged = tempfile.mkstemp(
            dir=destination.parent, prefix=f"{destination.name}."
        )
        os.close(handle)
        try:
            zipapp.create_archive(
                root, Path(staged), interpreter="/usr/bin/env python3"
            )
            os.replace(staged, destination)
        except BaseException:
            os.unlink(staged)  
            raise
    return destination
