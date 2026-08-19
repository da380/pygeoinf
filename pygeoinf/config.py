"""
pygeoinf/config.py
Shared configuration constants for the pygeoinf library.
"""

import os
from os.path import dirname, expanduser, join as joinpath

# The package's bundled data directory. This ships inside the wheel and is
# treated as read-only: it holds the seed datasets distributed with pygeoinf.
DATADIR = joinpath(dirname(__file__), "data")


def _default_cache_dir() -> str:
    """Return the platform-appropriate user cache directory for pygeoinf."""
    override = os.environ.get("PYGEOINF_CACHE_DIR")
    if override:
        return expanduser(override)

    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or expanduser("~\\AppData\\Local")
        return joinpath(base, "pygeoinf", "Cache")

    base = os.environ.get("XDG_CACHE_HOME") or expanduser("~/.cache")
    return joinpath(base, "pygeoinf")


# Writable location for datasets downloaded at runtime. Downloads must never be
# written into DATADIR, which may live in a read-only site-packages tree.
# Override with the PYGEOINF_CACHE_DIR environment variable.
CACHEDIR = _default_cache_dir()
