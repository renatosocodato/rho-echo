"""Load the canonical Rho-Echo implementation without copying its equations."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


PYTHON_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SOURCE = PYTHON_ROOT / "windowed_rhoa_model.py"
_MODULE_NAME = "_rho_echo_canonical_kernel_v1"


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a source file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_canonical_module() -> ModuleType:
    """Load the single canonical model module from its declared source path."""
    if not CANONICAL_SOURCE.is_file():
        raise ImportError(f"Canonical Rho-Echo source is missing: {CANONICAL_SOURCE}")

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, CANONICAL_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load canonical Rho-Echo source: {CANONICAL_SOURCE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
