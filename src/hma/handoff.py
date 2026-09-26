"""Atomic metadata writes and verified file copies used by HMA."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import time
from pathlib import Path
from typing import Any

def atomic_json(path: Path, value: Any) -> None:
    'Persist protected controller metadata with an atomic, durable rename.'
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(name, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)

def copy_verified(
    source: Path, target: Path, *, deadline: float | None = None
) -> dict[str, Any]:
    'Copy regular non-symlink files, break hard links, verify the destination.'
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as incoming:
        metadata = os.fstat(incoming.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"not a regular file: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        hashed = hashlib.sha256()
        with target.open("xb") as outgoing:
            while chunk := incoming.read(1024 * 1024):
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError("global deadline during handoff")
                outgoing.write(chunk)
                hashed.update(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        after = os.fstat(incoming.fileno())
        if (metadata.st_size, metadata.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"file changed during handoff: {source}")
    with target.open("rb") as copied:
        target_hash = hashlib.sha256()
        while chunk := copied.read(1024 * 1024):
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError("global deadline during verification")
            target_hash.update(chunk)
        verified = target_hash.hexdigest()
    if verified != hashed.hexdigest():
        raise RuntimeError(f"handoff hash mismatch: {target}")
    target.chmod(0o755 if metadata.st_mode & 0o111 else 0o644)
    return {"sha256": verified, "size": metadata.st_size}
