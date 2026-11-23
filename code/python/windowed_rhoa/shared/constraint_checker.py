"""Shared hard-constraint helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .math_utils import W_R, smooth_step
from .observables import compute_core_observables


def validate_hc1(thresholds) -> Dict:
    R_range = np.linspace(0.0, 1.0, 1000)
    W = W_R(R_range, thresholds.R_low, thresholds.R_high, thresholds.w1, thresholds.w2)
    peak_r = float(R_range[np.argmax(W)])
    peak_w = float(np.max(W))
    passes = peak_w > 0.9 and float(W[0]) < 0.1 and float(W[-1]) < 0.1
    return {
        "constraint": "HC-1: Windowed RhoA",
        "passes": passes,
        "peak_R": peak_r,
        "peak_W": peak_w,
        "W_at_R0": float(W[0]),
        "W_at_R1": float(W[-1]),
    }


def validate_hc2(params) -> Dict:
    R_test = params.thresh.R_src - 0.05
    gate = float(smooth_step(np.array([params.thresh.R_src - R_test]), params.thresh.w_surv)[0])
    return {
        "constraint": "HC-2: Src derepression",
        "passes": gate > 0.8,
        "src_gate_at_R_test": gate,
        "R_test": R_test,
    }


def validate_hc3(params) -> Dict:
    R_test = params.thresh.R_surv - 0.02
    L_test = 1.0
    surv_dmg = params.l1.a_R * float(
        smooth_step(np.array([params.thresh.R_surv - R_test]), params.thresh.w_surv)[0]
    )
    dH = params.l1.k_H * (1.0 - 0.5) - params.l1.a_L * L_test * 0.5 - surv_dmg * 0.5
    return {
        "constraint": "HC-3: Survival floor",
        "passes": dH < 0.0,
        "dH_at_test": float(dH),
        "R_test": R_test,
    }


def validate_hc5_from_trajectory(params, sol=None) -> Dict:
    if sol is None:
        return {"constraint": "HC-5: Morphology non-identifiability", "passes": None, "note": "Requires trajectory"}

    obs = compute_core_observables(
        R=sol.y[0],
        H=sol.y[1],
        A=sol.y[2],
        S=sol.y[3],
        Y9=sol.y[4],
        P=sol.y[5],
        Y10=sol.y[6],
        eps=params.obs.eps,
    )
    M = obs["M_morph"]
    I = obs["I_infl"]
    rng = np.random.default_rng(42)
    found = False
    for _ in range(1000):
        i, j = rng.integers(0, len(M), 2)
        if abs(M[i] - M[j]) < params.obs.eps_M and abs(I[i] - I[j]) > params.obs.delta_I:
            found = True
            break
    return {
        "constraint": "HC-5: Morphology non-identifiability",
        "passes": found,
        "M_range": [float(np.min(M)), float(np.max(M))],
        "I_range": [float(np.min(I)), float(np.max(I))],
    }
