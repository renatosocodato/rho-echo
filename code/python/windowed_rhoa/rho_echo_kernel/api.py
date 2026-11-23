"""Stable simulation API delegating all equations to canonical Rho-Echo code."""

from __future__ import annotations

from typing import Final

import numpy as np

from ._canonical import load_canonical_module
from .schemas import InputProgram, KernelTrajectory, SimulationRequest


KERNEL_API_VERSION: Final = "1.0.0"
MODEL_NAME: Final = "Rho-Echo"
MODEL_VERSION: Final = "windowed-rhoa-v3"
STATE_NAMES: Final = ("R", "H", "A", "S", "Y9", "P", "Y10")
OBSERVABLE_NAMES: Final = (
    "N_tox",
    "M_morph",
    "I_infl",
    "ROCK",
    "mDia",
    "Gamma",
    "Psi",
    "Xi",
    "Omega",
    "Phi",
)


def _input_functions(program: InputProgram):
    canonical = load_canonical_module()
    if program.inflammatory is None:
        inflammatory = canonical.zero_input()
    else:
        pulse = program.inflammatory
        inflammatory = canonical.make_lps(
            t_onset=pulse.onset,
            amp=pulse.amplitude,
            dur=pulse.duration,
            tau=pulse.decay_tau,
        )

    if program.narciclasine is None:
        narciclasine = canonical.zero_input()
    else:
        pulse = program.narciclasine
        narciclasine = canonical.make_narci(
            t_onset=pulse.onset,
            amp=pulse.amplitude,
            dur=pulse.duration,
        )

    if program.amyloid_beta is None:
        amyloid_beta = canonical.zero_input()
    else:
        pulse = program.amyloid_beta
        amyloid_beta = canonical.make_abeta_pulse(
            t_onset=pulse.onset,
            amp=pulse.amplitude,
            dur=pulse.duration,
        )
    return inflammatory, narciclasine, amyloid_beta


def simulate(request: SimulationRequest) -> KernelTrajectory:
    """Run the canonical seven-state Rho-Echo controller.

    Solver semantics are intentionally fixed to ``solve_R2``: BDF integration,
    ``rtol=1e-8``, ``atol=1e-10``, and ``max_step=0.1``. The interface does not
    duplicate or reinterpret any governing equation.
    """
    canonical = load_canonical_module()
    assert request.parameters is not None
    params = canonical.Params(**dict(request.parameters.values))
    inflammatory, narciclasine, amyloid_beta = _input_functions(request.inputs)
    times = np.asarray(request.sample_times, dtype=float)
    solution = canonical.solve_R2(
        params,
        L_func=inflammatory,
        N_func=narciclasine,
        A_ext_func=amyloid_beta,
        t_span=(float(times[0]), float(times[-1])),
        y0=list(request.initial_state),
        t_eval=times,
    )
    if not solution.success:
        raise RuntimeError(f"Canonical Rho-Echo integration failed: {solution.message}")

    computed = canonical.compute_observables(solution.y, params)
    states = {
        name: tuple(float(value) for value in solution.y[index])
        for index, name in enumerate(STATE_NAMES)
    }
    observables = {
        name: tuple(float(value) for value in np.asarray(computed[name]))
        for name in OBSERVABLE_NAMES
    }
    return KernelTrajectory(
        times=tuple(float(value) for value in solution.t),
        states=states,
        observables=observables,
        success=True,
        solver_message=str(solution.message),
        kernel_api_version=KERNEL_API_VERSION,
        model_version=MODEL_VERSION,
        parameter_sha256=request.parameters.sha256,
    )
