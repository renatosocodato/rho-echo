"""Focused numerical and serialization tests for the versioned Rho-Echo kernel."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from windowed_rhoa.rho_echo_kernel import (
    KERNEL_API_VERSION,
    MODEL_VERSION,
    InputProgram,
    KernelParameters,
    LpsPulse,
    RectangularPulse,
    SimulationRequest,
    get_origin_manifest,
    simulate,
    verify_golden_fixture,
)
from windowed_rhoa.rho_echo_kernel._canonical import load_canonical_module
from windowed_rhoa.rho_echo_kernel.benchmark import build_golden_fixture, load_golden_fixture


def compact_request() -> SimulationRequest:
    return SimulationRequest(
        sample_times=(0.0, 1.0, 3.0, 6.0, 10.0),
        inputs=InputProgram(
            inflammatory=LpsPulse(onset=2.0, amplitude=0.6, duration=3.0, decay_tau=4.0),
            amyloid_beta=RectangularPulse(onset=0.0, amplitude=0.5, duration=8.0),
        ),
        parameters=KernelParameters.defaults().with_updates(src_mode="abeta_modulated"),
    )


def test_version_and_origin_sources_are_frozen() -> None:
    manifest = get_origin_manifest(verify_sources=True)
    assert manifest["kernel_api_version"] == KERNEL_API_VERSION == "1.0.0"
    assert manifest["model_equation_version"] == MODEL_VERSION == "windowed-rhoa-v3"
    assert manifest["state_order"] == ["R", "H", "A", "S", "Y9", "P", "Y10"]
    assert len(manifest["canonical_source_sha256"]) == 7


def test_request_round_trip_and_parameter_boundary() -> None:
    request = compact_request()
    restored = SimulationRequest.from_dict(request.to_dict())
    assert restored.to_dict() == request.to_dict()
    assert restored.parameters is not None
    assert restored.parameters.sha256 == request.parameters.sha256
    with pytest.raises(ValueError, match="Unknown Rho-Echo parameters"):
        KernelParameters({**request.parameters.values, "invented_edge": 1.0})
    with pytest.raises(ValueError, match="RhoA thresholds"):
        request.parameters.with_updates(R_low=0.8, R_high=0.7)


def test_public_api_is_numerically_identical_to_canonical_solver() -> None:
    request = compact_request()
    trajectory = simulate(request)
    canonical = load_canonical_module()
    assert request.parameters is not None
    params = canonical.Params(**dict(request.parameters.values))
    inputs = request.inputs
    direct = canonical.solve_R2(
        params,
        L_func=canonical.make_lps(
            t_onset=inputs.inflammatory.onset,
            amp=inputs.inflammatory.amplitude,
            dur=inputs.inflammatory.duration,
            tau=inputs.inflammatory.decay_tau,
        ),
        N_func=canonical.zero_input(),
        A_ext_func=canonical.make_abeta_pulse(
            t_onset=inputs.amyloid_beta.onset,
            amp=inputs.amyloid_beta.amplitude,
            dur=inputs.amyloid_beta.duration,
        ),
        t_span=(request.sample_times[0], request.sample_times[-1]),
        y0=list(request.initial_state),
        t_eval=np.asarray(request.sample_times),
    )
    np.testing.assert_array_equal(np.asarray(trajectory.times), direct.t)
    for index, state_name in enumerate(("R", "H", "A", "S", "Y9", "P", "Y10")):
        np.testing.assert_array_equal(np.asarray(trajectory.states[state_name]), direct.y[index])
    direct_observables = canonical.compute_observables(direct.y, params)
    for name, values in trajectory.observables.items():
        np.testing.assert_array_equal(np.asarray(values), direct_observables[name])


def test_golden_fixture_is_static() -> None:
    packaged = load_golden_fixture()
    regenerated = build_golden_fixture()
    assert regenerated["request"] == packaged["request"]
    assert regenerated["fixture_id"] == packaged["fixture_id"]
    assert regenerated["origin_source_sha256"] == packaged["origin_source_sha256"]


def test_golden_fixture_strict_replay() -> None:
    report = verify_golden_fixture()
    assert report.passes
    assert report.max_absolute_error == 0.0
    assert report.max_scaled_error == 0.0
    assert report.compared_values > 100


def test_result_and_parameters_are_immutable() -> None:
    request = compact_request()
    trajectory = simulate(request)
    with pytest.raises(TypeError):
        request.parameters.values["k_H"] = 9.0
    with pytest.raises(TypeError):
        trajectory.states["R"] = (0.0,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        trajectory.success = False
