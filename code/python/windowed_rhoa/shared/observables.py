"""Shared observable and regime helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .math_utils import W_R


def classify_regime(R: float, thresholds) -> str:
    """Classify the RhoA regime using threshold-bearing params or threshold groups."""
    R_surv = getattr(thresholds, "R_surv", None)
    R_src = getattr(thresholds, "R_src", None)
    R_low = getattr(thresholds, "R_low", None)
    R_high = getattr(thresholds, "R_high", None)
    if hasattr(thresholds, "thresh"):
        thresh = thresholds.thresh
        R_surv = thresh.R_surv
        R_src = thresh.R_src
        R_low = thresh.R_low
        R_high = thresh.R_high

    if R > R_high:
        return "restrained"
    if R >= R_low:
        return "reactive_competent"
    if R >= R_src:
        return "intermediate_vulnerable"
    if R >= R_surv:
        return "vulnerable"
    return "dysregulated_toxic"


def compute_core_observables(
    *,
    R,
    H,
    A,
    S,
    Y9,
    P,
    Y10,
    eps: float,
) -> Dict[str, np.ndarray]:
    """Compute shared morphology and inflammatory observables."""
    R = np.asarray(R)
    H = np.asarray(H)
    A = np.asarray(A)
    S = np.asarray(S)
    Y9 = np.asarray(Y9)
    P = np.asarray(P)
    Y10 = np.asarray(Y10)

    n_tox = S * (1.0 - H)
    m_morph = P / (P + Y9 + eps)
    i_infl = (0.3 * A + 0.4 * S + 0.3 * Y10) / (1.0 + eps)
    return {
        "N_tox": n_tox,
        "M_morph": m_morph,
        "I_infl": i_infl,
    }


def compute_control_variables(*, R, Y9, P, Y10, H, thresholds, obs_eps, h_crit, rock, mdia):
    """Compute the shared control variables used in the legacy summary."""
    phi = W_R(
        R,
        thresholds.R_low,
        thresholds.R_high,
        thresholds.w1,
        thresholds.w2,
    )
    gamma = Y9 / (Y9 + P + obs_eps)
    psi = rock / (mdia + obs_eps)
    xi = H / h_crit
    omega = Y10 / (Y9 + Y10 + obs_eps)
    return {
        "Gamma": gamma,
        "Psi": psi,
        "Xi": xi,
        "Omega": omega,
        "Phi": phi,
    }
