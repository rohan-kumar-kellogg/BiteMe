from __future__ import annotations

import os
import platform
from pathlib import Path


def normalize_path(p: str) -> str:
    """
    Normalize path for robust comparisons across relative/absolute and separators.
    - resolves relative paths
    - normalizes separators
    - strips leading './'
    - lowercases on macOS/windows
    """
    raw = str(p).strip()
    if raw.startswith("./"):
        raw = raw[2:]
    path = Path(raw).expanduser().resolve()
    norm = os.path.normpath(str(path)).replace("\\", "/")
    if platform.system().lower() in {"darwin", "windows"}:
        norm = norm.lower()
    return norm

