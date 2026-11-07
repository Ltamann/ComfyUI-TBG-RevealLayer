"""Install missing extension dependencies on first execution, never at import."""

import importlib
import importlib.metadata
import logging
from pathlib import Path
import subprocess
import sys
import threading

_LOCK = threading.Lock()
_READY = False
REQUIREMENTS = Path(__file__).with_name("requirements.txt")


def missing_requirements():
    try:
        from packaging.requirements import Requirement
    except ImportError:
        return ["packaging"]
    missing = []
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(line)
            continue
        if installed not in requirement.specifier:
            missing.append(line)
    return missing


def ensure_dependencies():
    global _READY
    with _LOCK:
        if _READY:
            return
        missing = missing_requirements()
        if missing:
            logging.info("TBG RevealLayer: installing requirements: %s", ", ".join(missing))
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(REQUIREMENTS)],
                check=True,
            )
            importlib.invalidate_caches()
            # ComfyUI may already hold older modules in memory. A clean restart
            # is required after installation rather than mixing package versions.
            raise RuntimeError("TBG RevealLayer installed its requirements. Restart ComfyUI once, then queue again.")
        _READY = True
