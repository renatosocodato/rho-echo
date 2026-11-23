"""Versioned public interface to the canonical Rho-Echo ODE controller."""

from .api import (
    KERNEL_API_VERSION,
    MODEL_NAME,
    MODEL_VERSION,
    OBSERVABLE_NAMES,
    STATE_NAMES,
    simulate,
)
from .origin import get_origin_manifest, verify_origin_sources
from .schemas import (
    DEFAULT_INITIAL_STATE,
    InputProgram,
    KernelParameters,
    KernelTrajectory,
    LpsPulse,
    RectangularPulse,
    SimulationRequest,
)


def load_golden_fixture():
    """Load the packaged numerical contract without importing it at package startup."""
    from .benchmark import load_golden_fixture as _load

    return _load()


def verify_golden_fixture(fixture=None):
    """Replay the packaged numerical contract against the canonical equations."""
    from .benchmark import verify_golden_fixture as _verify

    return _verify(fixture)


__all__ = [
    "DEFAULT_INITIAL_STATE",
    "InputProgram",
    "KERNEL_API_VERSION",
    "KernelParameters",
    "KernelTrajectory",
    "LpsPulse",
    "MODEL_NAME",
    "MODEL_VERSION",
    "OBSERVABLE_NAMES",
    "RectangularPulse",
    "STATE_NAMES",
    "SimulationRequest",
    "get_origin_manifest",
    "load_golden_fixture",
    "simulate",
    "verify_golden_fixture",
    "verify_origin_sources",
]

__version__ = KERNEL_API_VERSION
