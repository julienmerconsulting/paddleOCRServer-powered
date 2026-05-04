"""paddleocrserver_powered — multi-language PaddleOCR HTTP server."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

from .server import main

__all__ = ["main", "spawn_server", "__version__"]

__version__ = "0.1.0"


def spawn_server(
    port: Optional[int] = None,
    host: Optional[str] = None,
    languages_config: Optional[str] = None,
    python_executable: Optional[str] = None,
    env: Optional[dict] = None,
    **popen_kwargs,
) -> subprocess.Popen:
    """
    Launch the server as a subprocess via `python -m paddleocrserver_powered`.

    Returns the Popen handle so the caller can wait on it, terminate it,
    or capture its output.
    """
    cmd = [python_executable or sys.executable, "-m", "paddleocrserver_powered"]
    if host is not None:
        cmd += ["--host", str(host)]
    if port is not None:
        cmd += ["--port", str(port)]
    if languages_config is not None:
        cmd += ["--languages-config", str(languages_config)]

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    return subprocess.Popen(cmd, env=merged_env, **popen_kwargs)
