"""paddleocrserver_powered — multi-language PaddleOCR HTTP server."""

from __future__ import annotations

import os
import subprocess
import sys

__all__ = ["main", "spawn_server", "__version__"]

__version__ = "0.1.0"


def __getattr__(name):
    """
    Lazy attribute access so `import paddleocrserver_powered` does not pull
    in the heavy runtime deps (paddleocr, cv2, ...) just for resource lookups
    or version introspection. The deps are only required when something
    actually calls `main` or `spawn_server`'s subprocess starts up.
    """
    if name == "main":
        from .server import main as _main
        return _main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def spawn_server(
    port: int | None = None,
    host: str | None = None,
    languages_config: str | None = None,
    python_executable: str | None = None,
    env: dict | None = None,
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
