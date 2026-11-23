"""Typed, JSON-safe schemas for the versioned Rho-Echo kernel boundary."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

from ._canonical import load_canonical_module


ParameterValue: TypeAlias = float | int | bool | str
DEFAULT_INITIAL_STATE = (0.6, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0)


def _require_finite(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _canonical_json(payload: Mapping[str, Any] | list[Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class LpsPulse:
    """Canonical inflammatory-input pulse with an exponential decay tail."""

    onset: float = 20.0
    amplitude: float = 1.0
    duration: float = 5.0
    decay_tau: float = 10.0

    def __post_init__(self) -> None:
        for name in ("onset", "amplitude", "duration", "decay_tau"):
            object.__setattr__(self, name, _require_finite(name, getattr(self, name)))
        if self.duration <= 0.0:
            raise ValueError("duration must be positive")
        if self.decay_tau <= 0.0:
            raise ValueError("decay_tau must be positive")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LpsPulse":
        return cls(**dict(payload))


@dataclass(frozen=True)
class RectangularPulse:
    """Finite rectangular pulse used for narciclasine or amyloid-beta input."""

    onset: float = 0.0
    amplitude: float = 1.0
    duration: float = 40.0

    def __post_init__(self) -> None:
        for name in ("onset", "amplitude", "duration"):
            object.__setattr__(self, name, _require_finite(name, getattr(self, name)))
        if self.duration <= 0.0:
            raise ValueError("duration must be positive")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RectangularPulse":
        return cls(**dict(payload))


@dataclass(frozen=True)
class InputProgram:
    """External inputs presented to the canonical seven-state controller."""

    inflammatory: LpsPulse | None = None
    narciclasine: RectangularPulse | None = None
    amyloid_beta: RectangularPulse | None = None

    def to_dict(self) -> dict[str, dict[str, float] | None]:
        return {
            "inflammatory": None if self.inflammatory is None else self.inflammatory.to_dict(),
            "narciclasine": None if self.narciclasine is None else self.narciclasine.to_dict(),
            "amyloid_beta": None if self.amyloid_beta is None else self.amyloid_beta.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InputProgram":
        inflammatory = payload.get("inflammatory")
        narciclasine = payload.get("narciclasine")
        amyloid_beta = payload.get("amyloid_beta")
        return cls(
            inflammatory=None if inflammatory is None else LpsPulse.from_dict(inflammatory),
            narciclasine=None if narciclasine is None else RectangularPulse.from_dict(narciclasine),
            amyloid_beta=None if amyloid_beta is None else RectangularPulse.from_dict(amyloid_beta),
        )


@dataclass(frozen=True)
class KernelParameters:
    """Complete explicit parameter set for one reproducible kernel simulation.

    Values are copied into an immutable mapping. Environment-dependent parameter
    loading is deliberately excluded from this boundary: every simulation carries
    the full flat parameter set used by the canonical ``Params`` dataclass.
    """

    values: Mapping[str, ParameterValue]

    def __post_init__(self) -> None:
        canonical = load_canonical_module()
        valid_names = {item.name for item in fields(canonical.Params)}
        copied = dict(self.values)
        unknown = sorted(set(copied).difference(valid_names))
        missing = sorted(valid_names.difference(copied))
        if unknown:
            raise ValueError(f"Unknown Rho-Echo parameters: {', '.join(unknown)}")
        if missing:
            raise ValueError(f"Missing Rho-Echo parameters: {', '.join(missing)}")
        invalid_types = sorted(
            name for name, value in copied.items() if not isinstance(value, (float, int, bool, str))
        )
        if invalid_types:
            raise TypeError(f"Non-scalar Rho-Echo parameters: {', '.join(invalid_types)}")
        nonfinite = sorted(
            name
            for name, value in copied.items()
            if isinstance(value, (float, int))
            and not isinstance(value, bool)
            and not math.isfinite(float(value))
        )
        if nonfinite:
            raise ValueError(f"Non-finite Rho-Echo parameters: {', '.join(nonfinite)}")

        params = canonical.Params(**copied)
        if not params.validate_ordering():
            raise ValueError("RhoA thresholds must satisfy R_surv < R_src < R_low < R_high")
        if params.src_mode not in {"threshold_only", "abeta_modulated"}:
            raise ValueError(f"Unsupported src_mode: {params.src_mode}")
        object.__setattr__(self, "values", MappingProxyType(copied))

    @classmethod
    def defaults(cls) -> "KernelParameters":
        """Return the explicit in-code canonical defaults, with no environment overrides."""
        canonical = load_canonical_module()
        return cls(asdict(canonical.Params()))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "KernelParameters":
        values = payload.get("values", payload)
        if not isinstance(values, Mapping):
            raise TypeError("KernelParameters payload must contain a parameter mapping")
        return cls(values)

    def with_updates(self, **updates: ParameterValue) -> "KernelParameters":
        values = dict(self.values)
        values.update(updates)
        return KernelParameters(values)

    def to_dict(self) -> dict[str, dict[str, ParameterValue]]:
        return {"values": dict(self.values)}

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SimulationRequest:
    """Fully specified deterministic request for the canonical R2 ODE solver."""

    sample_times: tuple[float, ...]
    initial_state: tuple[float, ...] = DEFAULT_INITIAL_STATE
    inputs: InputProgram = InputProgram()
    parameters: KernelParameters | None = None

    def __post_init__(self) -> None:
        times = tuple(_require_finite("sample_time", value) for value in self.sample_times)
        state = tuple(_require_finite("initial_state", value) for value in self.initial_state)
        if len(times) < 2:
            raise ValueError("sample_times must contain at least two values")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("sample_times must be strictly increasing")
        if len(state) != 7:
            raise ValueError("initial_state must follow the seven-state order R,H,A,S,Y9,P,Y10")
        object.__setattr__(self, "sample_times", times)
        object.__setattr__(self, "initial_state", state)
        if self.parameters is None:
            object.__setattr__(self, "parameters", KernelParameters.defaults())

    @classmethod
    def regular_grid(
        cls,
        *,
        start: float,
        stop: float,
        samples: int,
        initial_state: tuple[float, ...] = DEFAULT_INITIAL_STATE,
        inputs: InputProgram = InputProgram(),
        parameters: KernelParameters | None = None,
    ) -> "SimulationRequest":
        if samples < 2:
            raise ValueError("samples must be at least 2")
        start = _require_finite("start", start)
        stop = _require_finite("stop", stop)
        if stop <= start:
            raise ValueError("stop must be greater than start")
        step = (stop - start) / (samples - 1)
        times = tuple(start + index * step for index in range(samples))
        return cls(times, initial_state, inputs, parameters)

    def to_dict(self) -> dict[str, Any]:
        assert self.parameters is not None
        return {
            "sample_times": list(self.sample_times),
            "initial_state": list(self.initial_state),
            "inputs": self.inputs.to_dict(),
            "parameters": self.parameters.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SimulationRequest":
        parameter_payload = payload.get("parameters")
        return cls(
            sample_times=tuple(payload["sample_times"]),
            initial_state=tuple(payload.get("initial_state", DEFAULT_INITIAL_STATE)),
            inputs=InputProgram.from_dict(payload.get("inputs", {})),
            parameters=(
                None if parameter_payload is None else KernelParameters.from_dict(parameter_payload)
            ),
        )


@dataclass(frozen=True)
class KernelTrajectory:
    """Immutable, JSON-safe state and observable trajectory returned by the kernel."""

    times: tuple[float, ...]
    states: Mapping[str, tuple[float, ...]]
    observables: Mapping[str, tuple[float, ...]]
    success: bool
    solver_message: str
    kernel_api_version: str
    model_version: str
    parameter_sha256: str

    def __post_init__(self) -> None:
        length = len(self.times)
        states = {name: tuple(values) for name, values in self.states.items()}
        observables = {name: tuple(values) for name, values in self.observables.items()}
        if any(len(values) != length for values in (*states.values(), *observables.values())):
            raise ValueError("Every state and observable must align with times")
        object.__setattr__(self, "states", MappingProxyType(states))
        object.__setattr__(self, "observables", MappingProxyType(observables))

    def to_dict(self) -> dict[str, Any]:
        return {
            "times": list(self.times),
            "states": {name: list(values) for name, values in self.states.items()},
            "observables": {name: list(values) for name, values in self.observables.items()},
            "success": self.success,
            "solver_message": self.solver_message,
            "kernel_api_version": self.kernel_api_version,
            "model_version": self.model_version,
            "parameter_sha256": self.parameter_sha256,
        }
