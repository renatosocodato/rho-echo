"""Golden trajectory generation and independent replay for the kernel interface."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .api import KERNEL_API_VERSION, MODEL_VERSION, simulate
from .origin import get_origin_manifest
from .schemas import InputProgram, KernelParameters, LpsPulse, RectangularPulse, SimulationRequest


GOLDEN_FIXTURE_RESOURCE = "fixtures/r2_golden_v1.json"


@dataclass(frozen=True)
class ReplayReport:
    fixture_id: str
    passes: bool
    max_absolute_error: float
    max_scaled_error: float
    compared_values: int


def golden_request() -> SimulationRequest:
    """Return the fixed multi-input request used by the v1 numerical contract."""
    parameters = KernelParameters.defaults().with_updates(src_mode="abeta_modulated")
    return SimulationRequest(
        sample_times=(0.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 35.0, 40.0, 50.0, 60.0),
        inputs=InputProgram(
            inflammatory=LpsPulse(onset=8.0, amplitude=0.8, duration=10.0, decay_tau=12.0),
            narciclasine=RectangularPulse(onset=30.0, amplitude=0.25, duration=8.0),
            amyloid_beta=RectangularPulse(onset=0.0, amplitude=0.7, duration=40.0),
        ),
        parameters=parameters,
    )


def build_golden_fixture() -> dict[str, Any]:
    request = golden_request()
    trajectory = simulate(request)
    expected = trajectory.to_dict()
    expected.pop("solver_message")
    return {
        "schema_version": "1.0.0",
        "fixture_id": "rho-echo-r2-multi-input-v1",
        "kernel_api_version": KERNEL_API_VERSION,
        "model_version": MODEL_VERSION,
        "origin_source_sha256": get_origin_manifest()["canonical_source_sha256"],
        "tolerances": {"relative": 5e-8, "absolute": 5e-10},
        "request": request.to_dict(),
        "expected": expected,
    }


def load_golden_fixture() -> dict[str, Any]:
    resource = files(__package__).joinpath(GOLDEN_FIXTURE_RESOURCE)
    return json.loads(resource.read_text(encoding="utf-8"))


def _arrays(payload: Mapping[str, Any]):
    yield "times", np.asarray(payload["times"], dtype=float)
    for family in ("states", "observables"):
        for name, values in payload[family].items():
            yield f"{family}.{name}", np.asarray(values, dtype=float)


def verify_golden_fixture(fixture: Mapping[str, Any] | None = None, *, portable: bool = False) -> ReplayReport:
    """Replay a fixture through the canonical solver and check its numerical envelope."""
    payload = dict(fixture or load_golden_fixture())
    if payload["kernel_api_version"] != KERNEL_API_VERSION:
        raise RuntimeError("Golden fixture targets a different kernel API version")
    if payload["model_version"] != MODEL_VERSION:
        raise RuntimeError("Golden fixture targets a different Rho-Echo model version")
    get_origin_manifest(verify_sources=True)
    if payload["origin_source_sha256"] != get_origin_manifest()["canonical_source_sha256"]:
        raise RuntimeError("Golden fixture source hashes do not match the origin manifest")

    observed = simulate(SimulationRequest.from_dict(payload["request"])).to_dict()
    expected = payload["expected"]
    relative = float(payload["tolerances"]["relative"])
    absolute = float(payload["tolerances"]["absolute"])
    if portable:
        # Adaptive solver error accumulates across input discontinuities.
        # Keep this cross-platform check separate from the original fixture.
        relative, absolute = 1e-6, 1e-8
    max_absolute = 0.0
    max_scaled = 0.0
    compared = 0
    passes = bool(observed["success"]) == bool(expected["success"])

    observed_arrays = dict(_arrays(observed))
    for name, expected_array in _arrays(expected):
        observed_array = observed_arrays[name]
        delta = np.abs(observed_array - expected_array)
        scale = absolute + relative * np.abs(expected_array)
        passes = passes and bool(np.all(delta <= scale))
        max_absolute = max(max_absolute, float(np.max(delta)))
        max_scaled = max(max_scaled, float(np.max(delta / scale)))
        compared += int(expected_array.size)

    for key in ("kernel_api_version", "model_version", "parameter_sha256"):
        passes = passes and observed[key] == expected[key]
    return ReplayReport(
        fixture_id=str(payload["fixture_id"]),
        passes=passes,
        max_absolute_error=max_absolute,
        max_scaled_error=max_scaled,
        compared_values=compared,
    )


def write_golden_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_golden_fixture(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Replay the packaged fixture")
    parser.add_argument("--stdout", action="store_true", help="Print a newly generated fixture")
    parser.add_argument("--output", type=Path, help="Write a newly generated fixture to this path")
    parser.add_argument("--portable", action="store_true", help="Use the cross-platform envelope with --check")
    args = parser.parse_args(argv)
    if args.portable and not args.check:
        parser.error("--portable requires --check")
    if args.check:
        report = verify_golden_fixture(portable=args.portable)
        print(json.dumps(report.__dict__, indent=2, sort_keys=True))
        return 0 if report.passes else 1
    if args.stdout:
        print(json.dumps(build_golden_fixture(), indent=2, sort_keys=True))
        return 0
    if args.output:
        write_golden_fixture(args.output)
        return 0
    parser.error("choose --check, --stdout, or --output")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
