"""Helpers for the optional Aβ-modulated Src reconciliation branch."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp

from windowed_rhoa.shared.math_utils import smooth_step


def src_coupling_factor(R: float, eta_coup: float, K_coup: float) -> float:
    """Return the RhoA-dependent modulation factor for Aβ-driven Src activation."""
    denom = max(K_coup + R, 1e-9)
    return float(1.0 - eta_coup * R / denom)


def compute_src_terms(
    *,
    R: float,
    S: float,
    A_ext: float,
    src_mode: str,
    k_S_thresh: float,
    k_S_abeta: float,
    eta_coup: float,
    K_coup: float,
    d_S: float,
    R_src: float,
    w_surv: float,
) -> Dict[str, float]:
    """Compute Src branch terms without modifying other model layers."""
    threshold_gate = float(smooth_step(np.array([R_src - R]), w_surv)[0])
    coupling = src_coupling_factor(R, eta_coup, K_coup)
    threshold_drive = k_S_thresh * threshold_gate * (1.0 - S)

    if src_mode not in {"threshold_only", "abeta_modulated"}:
        raise ValueError(f"Unknown src_mode: {src_mode}")

    abeta_drive = 0.0
    if src_mode == "abeta_modulated":
        abeta_drive = k_S_abeta * A_ext * coupling * (1.0 - S)

    dS = abeta_drive + threshold_drive - d_S * S
    return {
        "threshold_gate": threshold_gate,
        "coupling_factor": coupling,
        "abeta_drive": abeta_drive,
        "threshold_drive": threshold_drive,
        "dS": dS,
    }


def simulate_fixed_r_src(
    *,
    R_level: float,
    t_eval: np.ndarray,
    S0: float,
    src_mode: str,
    k_S_thresh: float,
    k_S_abeta: float,
    eta_coup: float,
    K_coup: float,
    d_S: float,
    R_src: float,
    w_surv: float,
    A_ext_func: Optional[Callable[[float], float]] = None,
) -> Dict[str, np.ndarray]:
    """Simulate the Src branch alone at fixed R for comparison against FRET traces."""
    A_ext_func = A_ext_func or (lambda _t: 0.0)

    def rhs(t: float, y: np.ndarray) -> list[float]:
        terms = compute_src_terms(
            R=R_level,
            S=float(np.clip(y[0], 0.0, 1.0)),
            A_ext=float(A_ext_func(t)),
            src_mode=src_mode,
            k_S_thresh=k_S_thresh,
            k_S_abeta=k_S_abeta,
            eta_coup=eta_coup,
            K_coup=K_coup,
            d_S=d_S,
            R_src=R_src,
            w_surv=w_surv,
        )
        return [terms["dS"]]

    sol = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        [S0],
        t_eval=t_eval,
        method="BDF",
        rtol=1e-8,
        atol=1e-10,
    )
    return {
        "success": bool(sol.success),
        "t": sol.t,
        "S": sol.y[0],
        "message": sol.message,
    }
