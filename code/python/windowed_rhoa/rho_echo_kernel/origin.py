"""Static origin record and source-integrity verification for the Rho-Echo kernel."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from ._canonical import PYTHON_ROOT, sha256_file


ORIGIN_RESOURCE = "origin_manifest.json"


def get_origin_manifest(*, verify_sources: bool = False) -> dict[str, Any]:
    """Return the versioned origin manifest, optionally checking every source digest."""
    resource = files(__package__).joinpath(ORIGIN_RESOURCE)
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if verify_sources:
        verify_origin_sources(payload)
    return payload


def verify_origin_sources(manifest: dict[str, Any] | None = None) -> None:
    """Raise when a source file differs from the bytes frozen by the interface."""
    payload = manifest or get_origin_manifest()
    mismatches: list[str] = []
    for relative_path, expected in payload["canonical_source_sha256"].items():
        path = PYTHON_ROOT / Path(relative_path)
        if not path.is_file():
            mismatches.append(f"{relative_path}: missing")
            continue
        observed = sha256_file(path)
        if observed != expected:
            mismatches.append(f"{relative_path}: expected {expected}, observed {observed}")
    if mismatches:
        raise RuntimeError("Rho-Echo origin verification failed: " + "; ".join(mismatches))
