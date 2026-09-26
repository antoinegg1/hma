"""A container of one image, holding the workspace at the path it already has here.

The project directory is mounted rather than copied, and the container runs as the user who
started the flow, so what a turn writes there is this user's file in this user's directory and
survives the container it was written from. Everything else -- the tools, the interpreters, the
libraries a command reaches for -- is the image's, which is what the isolation is.

Driven through the `docker` command rather than a client library, because that is what a turn
reaches the container through: coganchor's `docker://` target runs its own half over
`docker exec`, and one machine is best spoken to in one voice.
"""

from __future__ import annotations

import errno
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from hmz.coganchor import AnchorConfig, check

from .base import MachineBase, MachineConfig

_IDLE = ("python3", "-c", "import time; time.sleep(2**31)")

_LABEL = "humanize"

@dataclass(frozen=True, kw_only=True)
class DockerConfig(MachineConfig):
    """What the container is run from.

    Attributes:
      image: The image to run, which needs a `python3` for coganchor's target half and whatever
        else the agent is expected to reach for.
      workspace: The project directory to give the container, defaulting to this one. It is the
        directory itself that goes there, not a copy of it, so the work outlives the container.
    """

    image: str = "python:3.12"
    workspace: str | None = None

    def create(self) -> Docker:
        """Builds the backend, without starting a container yet."""
        return Docker(self)

class Docker(MachineBase):
    """One container, and the mirror the agent works in while its turns land there."""

    _config: DockerConfig

    def __init__(self, config: DockerConfig) -> None:
        """Initializes a backend holding no container.

        Args:
          config: The image and workspace the container is started with.
        """
        super().__init__(config)
        self._mirror: tempfile.TemporaryDirectory[str] | None = None
        self._name: str | None = None

    def start(self) -> AnchorConfig:
        """Starts the container and asks it what it is holding.

        Returns:
          The anchor that reaches it, which names the workspace by the path it has here.

        Raises:
          FileNotFoundError: If there is no workspace directory to give the container, or no
            `docker` to give it to.
          RuntimeError: If the container cannot be started -- an image with no `python3` in it
            is refused here. What docker said is attached.
          OSError: If the container cannot serve the workspace it was mounted, which is a turn
            that would fail on its first file, reported before the first turn instead.
        """

        workspace = os.path.abspath(self._config.workspace or os.getcwd())  
        if not Path(workspace).is_dir():

            raise FileNotFoundError(
                errno.ENOENT, "no directory to give the container", workspace
            )

        self._mirror = tempfile.TemporaryDirectory(
            prefix="humanize-", ignore_cleanup_errors=True
        )
        self._name = Path(self._mirror.name).name
        try:
            started = subprocess.run(
                [
                    "docker",
                    "run",
                    "--detach",
                    "--name",
                    self._name,

                    "--label",
                    f"{_LABEL}={os.getuid()}",
                    "--user",
                    f"{os.getuid()}:{os.getgid()}",
                    "--workdir",
                    workspace,

                    "--env",
                    "HOME=/tmp",
                    "--volume",
                    f"{workspace}:{workspace}",
                    self._config.image,
                    *_IDLE,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if started.returncode != 0:

                raise RuntimeError(  
                    f"could not start a container of {self._config.image}: "
                    f"{started.stderr.strip()}"
                )
            anchor = AnchorConfig(
                target=f"docker://{self._name}",
                workspace=workspace,
                shadow=str(Path(self._mirror.name) / "shadow"),
            )
            check(anchor)  
        except BaseException:
            self.stop()
            raise
        return anchor

    def stop(self) -> None:
        """Removes the container and the mirror, leaving the workspace as the turns left it."""
        if self._name is not None:
            
            subprocess.run(
                ["docker", "rm", "--force", self._name],
                capture_output=True,
                check=False,
            )
        if self._mirror is not None:
            self._mirror.cleanup()
