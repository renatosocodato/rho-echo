"""
windowed_rhoa_model.py — v3 Master Model

"RhoA defines a viability-constrained reactive window, while Myh9 and
Myh10 split morphology from inflammatory state."

Three-layer architecture:
  Layer 1: Windowed RhoA + Survival (R, H, A)
  Layer 2: Motor-resolved outputs (S, Y9, P, Y10)
  Layer 3: Disease extension (B, K_casp) — demoted to hypothesis

All model variants must satisfy HC-1 through HC-5 (hard-constraint ledger).
"""

import json
import os
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from windowed_rhoa.shared.constraint_checker import (
    validate_hc1 as shared_validate_hc1,
    validate_hc2 as shared_validate_hc2,
    validate_hc3 as shared_validate_hc3,
    validate_hc5_from_trajectory as shared_validate_hc5,
)
from windowed_rhoa.shared.math_utils import (
    W_R as shared_W_R,
    sigmoid as shared_sigmoid,
    smooth_step as shared_smooth_step,
)
from windowed_rhoa.shared.observables import (
    classify_regime as shared_classify_regime,
    compute_control_variables,
    compute_core_observables,
)
from windowed_rhoa.reconciliation.src_dynamics import compute_src_terms
from windowed_rhoa.shared.parameters import AllParams as SharedAllParams
from windowed_rhoa.shared.parameters import default_params as shared_default_params


ROOT = Path(__file__).resolve().parents[2]
# A released code archive can be mounted beside, rather than inside, its large
# data package.  Preserve the repository-local path by default, but permit the
# R6 launcher to bind the calibration output explicitly to its supplied data
# root.  The model equations remain unchanged; this only makes the declared
# calcium-informed parameter source discoverable in a separated installation.
DATA_ROOT = Path(os.environ.get("RHO_ECHO_DATA_ROOT", ROOT))
CALCIUM_CROSS_BRANCH_PATH = (
    DATA_ROOT / "results" / "exports" / "calcium_hc3_inference" / "calcium_hc3_cross_branch_integration.json"
)


# ══════════════════════════════════════════════════════════
# WINDOW FUNCTION
# ══════════════════════════════════════════════════════════

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return shared_sigmoid(x)


def W_R(R: np.ndarray, R_low: float, R_high: float,
        w1: float = 0.05, w2: float = 0.05) -> np.ndarray:
    """
    Windowed-RhoA function (Eq. L1-1).

    Returns ~1 when R ∈ [R_low, R_high], ~0 outside.
    Productive reactivity is only possible inside this window.

    W_R(R) = σ((R − R_low)/w₁) · σ((R_high − R)/w₂)
    """
    return shared_W_R(R, R_low, R_high, w1, w2)


def smooth_step(x: np.ndarray, width: float = 0.02) -> np.ndarray:
    """Smooth Heaviside: Θ(x) ≈ σ(x/width)."""
    return shared_smooth_step(x, width)


# ══════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════

@dataclass
class Params:
    """
    All parameters, organized by layer.

    Threshold ordering (hard prior): R_surv < R_src < R_low < R_high
    """

    # ── Thresholds (Layer 1, constrained ordering) ──
    R_surv: float = 0.05    # Below this: survival collapse
    R_src: float = 0.15     # Below this: Src derepressed
    R_low: float = 0.30     # Lower bound of reactive window
    R_high: float = 0.70    # Upper bound of reactive window
    w1: float = 0.05        # Window sigmoid width (lower)
    w2: float = 0.05        # Window sigmoid width (upper)
    w_surv: float = 0.02    # Survival step smoothing width

    # ── Layer 1: RhoA dynamics ──
    k_R_GEF: float = 0.4    # GEF-H1 → RhoA activation
    alpha_S: float = 0.5    # Src → GEF-H1 positive feedback (Azoitei 2019)
    k_R_N: float = 0.3      # Narciclasine → RhoA (direct arm)
    k_R_L: float = 0.8      # LPS → RhoA decrease (increased from v2)
    k_R_off: float = 0.3    # GAP-mediated RhoA inactivation

    # ── Layer 1: Productive reactivity ──
    k_A: float = 0.8        # Reactivity activation rate
    d_A: float = 0.3        # Reactivity decay

    # ── Layer 1: Homeostatic reserve ──
    k_H: float = 0.014      # Recovery rate (refitted to significant late-phase Ca2+)
    a_L: float = 0.22       # LPS-mediated H degradation
    a_R: float = 0.42       # Low-RhoA-mediated H collapse (Chronic Attractor drive)

    # ── Layer 2: Src ──
    k_S_thresh: float = 0.5 # KO-level Src activation when R < R_src
    d_S: float = 0.2        # Src deactivation
    k_S_abeta: float = 0.35 # Aβ-driven Src activation branch
    eta_coup: float = 0.35  # Degree of RhoA modulation of Aβ→Src coupling
    K_coup: float = 0.25    # Half-max for coupling modulation
    src_mode: str = "threshold_only"

    # ── Layer 2: Myh9 ──
    a_9: float = 0.8        # ROCK → Myh9
    d_9: float = 0.4        # Myh9 turnover
    V_K: float = 1.0        # Maximal ROCK activation
    K_R_rock: float = 0.3   # RhoA half-max for ROCK

    # ── Layer 2: Protrusive support ──
    a_P: float = 0.5        # mDia → protrusion
    a_C: float = 0.3        # Cofilin turnover → protrusion extension
    b_P: float = 0.3        # Y9 antagonism of protrusion
    d_P: float = 0.2        # Protrusion decay
    V_D: float = 1.0        # Maximal mDia activation
    K_R_mdia: float = 0.25  # RhoA half-max for mDia

    # ── Layer 2: Myh10 ──
    a_10: float = 0.4       # Inflammatory drive → Myh10
    w_L_Y10: float = 0.3    # Weight of LPS in F_infl
    w_S_Y10: float = 0.5    # Weight of Src in F_infl
    w_A_Y10: float = 0.2    # Weight of productive reactivity in F_infl
    b_10: float = 0.15      # Self-reinforcement (commitment feedback)
    K_hyst: float = 0.3     # Commitment threshold
    d_10: float = 0.1       # Myh10 turnover (slow → memory)

    # ── Layer 2: MT (simplified) ──
    k_T_rec: float = 0.1    # MT recovery
    k_T_K: float = 0.15     # ROCK → MT catastrophe

    # ── Layer 3: Disease extension ──
    k_restore: float = 0.15
    k_desens: float = 0.25
    K_half_desens: float = 0.5
    n_hill: int = 3
    k_casp: float = 0.3     # ⚠ HYPOTHESIS: untested in microglia
    d_casp: float = 0.08
    enable_kcasp: bool = True

    # ── Narciclasine ──
    rho_narci: float = 3.0  # Arm ratio (sweep 0.5–10)
    k_N_offtar: float = 0.1 # Off-target Src suppression (Zhao 2021)

    # ── Misc ──
    eps: float = 0.01
    H_crit: float = 0.5     # Operational reserve-transition reference, not a damage threshold

    def validate_ordering(self) -> bool:
        """Enforce R_surv < R_src < R_low < R_high."""
        return self.R_surv < self.R_src < self.R_low < self.R_high

    @property
    def k_S(self) -> float:
        """Legacy alias for the KO-threshold Src rate."""
        return self.k_S_thresh

    @k_S.setter
    def k_S(self, value: float) -> None:
        self.k_S_thresh = value


AllParams = SharedAllParams


def default_params() -> Params:
    """Legacy-friendly default parameter constructor."""
    params = Params()
    if CALCIUM_CROSS_BRANCH_PATH.exists():
        try:
            payload = json.loads(CALCIUM_CROSS_BRANCH_PATH.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if bool(payload.get("propagation_enabled")) and isinstance(payload.get("shared_layer1_updates"), dict):
            updates = payload["shared_layer1_updates"]
            for key in ("a_L", "a_R", "k_H"):
                if key in updates:
                    setattr(params, key, float(updates[key]))
    frozen_hcrit = os.environ.get("RHO_ECHO_R6_H_CRIT")
    if frozen_hcrit is not None:
        params.H_crit = float(frozen_hcrit)
    return params


# ══════════════════════════════════════════════════════════
# EXTERNAL INPUTS
# ══════════════════════════════════════════════════════════

def zero_input():
    return lambda t: 0.0

def make_lps(t_onset=20.0, amp=1.0, dur=5.0, tau=10.0):
    def f(t):
        if t < t_onset: return 0.0
        elif t < t_onset + dur: return amp
        else: return amp * np.exp(-(t - t_onset - dur) / tau)
    return f

def make_narci(t_onset=10.0, amp=0.5, dur=10.0):
    def f(t):
        return amp if t_onset <= t < t_onset + dur else 0.0
    return f

def make_abeta_pulse(t_onset=0.0, amp=1.0, dur=40.0):
    def f(t):
        return amp if t_onset <= t < t_onset + dur else 0.0
    return f

def make_chronic_stress(rate=0.005, ceiling=0.8):
    def f(t):
        return min(rate * t, ceiling)
    return f

def make_caspase(basal=0.05, rate=0.003, ceiling=0.5):
    def f(t):
        return min(basal + rate * t, ceiling)
    return f

def make_const(val):
    return lambda t: val


# ══════════════════════════════════════════════════════════
# LAYER 1: WINDOWED RhoA + SURVIVAL (R1)
# ══════════════════════════════════════════════════════════

def R1_rhs(t, y, p: Params, L_func, N_func):
    """
    Layer 1 minimal model: R, H, A (3 variables).
    Demonstrates HC-1 (windowed reactivity) and HC-3 (survival floor).
    """
    R, H, A = y
    R = np.clip(R, 0, 1)
    H = np.clip(H, 0, 1)
    A = np.clip(A, 0, 1)

    L = L_func(t)
    N = N_func(t)

    # Window function
    W = float(W_R(np.array([R]), p.R_low, p.R_high, p.w1, p.w2)[0])

    # RhoA (simplified: no MT/Src feedback in R1)
    dR = (p.k_R_GEF * (1 - R)
          + p.k_R_N * N * (1 - R)
          - p.k_R_L * L * R
          - p.k_R_off * R)

    # Homeostatic reserve (L1-4)
    surv_damage = p.a_R * float(smooth_step(np.array([p.R_surv - R]), p.w_surv)[0])
    dH = p.k_H * (1 - H) - p.a_L * L * H - surv_damage * H

    # Productive reactivity (L1-3)
    dA = p.k_A * L * W * H * (1 - A) - p.d_A * A

    return [dR, dH, dA]


def solve_R1(p: Params, L_func=None, N_func=None,
             t_span=(0, 100), y0=(0.6, 1.0, 0.0), **kw):
    L_func = L_func or zero_input()
    N_func = N_func or zero_input()
    t_eval = kw.pop('t_eval', np.linspace(*t_span, 3000))
    return solve_ivp(R1_rhs, t_span, y0, args=(p, L_func, N_func),
                     t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-10, **kw)


# ══════════════════════════════════════════════════════════
# LAYER 2: MOTOR-RESOLVED OUTPUTS (R2, 7 variables)
# ══════════════════════════════════════════════════════════

def R2_rhs(t, y, p: Params, L_func, N_func, A_ext_func=None):
    """
    Layer 1+2: R, H, A, S, Y9, P, Y10 (7 variables).
    """
    R, H, A, S, Y9, P, Y10 = y
    R  = np.clip(R, 0, 1)
    H  = np.clip(H, 0, 1)
    A  = np.clip(A, 0, 1)
    S  = np.clip(S, 0, 1)
    Y9 = max(Y9, 0)
    P  = max(P, 0)
    Y10 = max(Y10, 0)

    L = L_func(t)
    N = N_func(t)
    A_ext = A_ext_func(t) if A_ext_func is not None else 0.0

    W = float(W_R(np.array([R]), p.R_low, p.R_high, p.w1, p.w2)[0])

    # ── Layer 1 ──
    gef_h1 = (1 - 0.5) * (1 + p.alpha_S * S)  # T fixed at 0.5 in R2
    dR = (p.k_R_GEF * gef_h1 * (1 - R)
          + p.k_R_N * N * (1 - R)
          - p.k_R_L * L * R
          - p.k_R_off * R)

    surv_dmg = p.a_R * float(smooth_step(np.array([p.R_surv - R]), p.w_surv)[0])
    dH = p.k_H * (1 - H) - p.a_L * L * H - surv_dmg * H

    dA = p.k_A * L * W * H * (1 - A) - p.d_A * A

    # ── Layer 2 ──
    src_terms = compute_src_terms(
        R=R,
        S=S,
        A_ext=A_ext,
        src_mode=p.src_mode,
        k_S_thresh=p.k_S_thresh,
        k_S_abeta=p.k_S_abeta,
        eta_coup=p.eta_coup,
        K_coup=p.K_coup,
        d_S=p.d_S,
        R_src=p.R_src,
        w_surv=p.w_surv,
    )
    dS = src_terms['dS']

    # ROCK
    ROCK = p.V_K * R / (p.K_R_rock + R)

    # Myh9 (L2-3)
    dY9 = p.a_9 * ROCK - p.d_9 * Y9

    # Protrusive support (L2-4)
    mDia = p.V_D * R / (p.K_R_mdia + R)
    dP = p.a_P * mDia + p.a_C - p.b_P * Y9 * P - p.d_P * P

    # Myh10 (L2-5) — driven by weighted inflammatory input
    F_infl = p.w_L_Y10 * L + p.w_S_Y10 * S + p.w_A_Y10 * A
    h_Y10 = Y10**2 / (p.K_hyst**2 + Y10**2) if Y10 > p.eps else 0.0
    dY10 = p.a_10 * F_infl + p.b_10 * h_Y10 - p.d_10 * Y10

    return [dR, dH, dA, dS, dY9, dP, dY10]


def solve_R2(p: Params, L_func=None, N_func=None, A_ext_func=None,
             t_span=(0, 100), y0=None, **kw):
    L_func = L_func or zero_input()
    N_func = N_func or zero_input()
    A_ext_func = A_ext_func or zero_input()
    if y0 is None:
        y0 = [0.6, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0]
    t_eval = kw.pop('t_eval', np.linspace(*t_span, 5000))
    return solve_ivp(R2_rhs, t_span, y0, args=(p, L_func, N_func, A_ext_func),
                     t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-10,
                     max_step=0.1, **kw)


# ══════════════════════════════════════════════════════════
# OBSERVABLES AND CONTROL VARIABLES
# ══════════════════════════════════════════════════════════

def compute_observables(y, p: Params, L_func=None, t=None) -> Dict:
    """
    Compute observables and control variables from state trajectories.

    For R2: y has shape (7, N) = [R, H, A, S, Y9, P, Y10]
    """
    R, H, A, S, Y9, P, Y10 = y
    eps = p.eps

    rock = p.V_K * R / (p.K_R_rock + R)
    mdia = p.V_D * R / (p.K_R_mdia + R)
    outputs = compute_core_observables(R=R, H=H, A=A, S=S, Y9=Y9, P=P, Y10=Y10, eps=eps)
    controls = compute_control_variables(
        R=R,
        Y9=Y9,
        P=P,
        Y10=Y10,
        H=H,
        thresholds=p,
        obs_eps=eps,
        h_crit=p.H_crit,
        rock=rock,
        mdia=mdia,
    )

    return {
        **outputs,
        'ROCK': rock,
        'mDia': mdia,
        **controls,
    }


def classify_regime(R: float, p: Params) -> str:
    """Classify RhoA regime."""
    return shared_classify_regime(R, p)


# ══════════════════════════════════════════════════════════
# HARD-CONSTRAINT VALIDATION
# ══════════════════════════════════════════════════════════

def validate_HC1(p: Params) -> Dict:
    """HC-1: Windowed RhoA. A_ss must be non-monotonic in R."""
    return shared_validate_hc1(p)


def validate_HC2(p: Params) -> Dict:
    """HC-2: Src derepression. S rises when R < R_src."""
    return shared_validate_hc2(_legacy_to_allparams(p))


def validate_HC3(p: Params) -> Dict:
    """HC-3: Survival floor. H degrades when R < R_surv under LPS."""
    return shared_validate_hc3(_legacy_to_allparams(p))


def validate_HC4(p: Params) -> Dict:
    """HC-4: Motor orthogonality. Y9 and Y10 must be decorrelated in at least one regime."""
    # Run R2 at different fixed R values, check Y9/Y10 correlation
    results = {}
    for regime, R_fixed in [('restrained', 0.8), ('reactive', 0.5), ('dysregulated', 0.1)]:
        # At steady state with constant L=1
        ROCK = p.V_K * R_fixed / (p.K_R_rock + R_fixed)
        Y9_ss = p.a_9 * ROCK / p.d_9

        src_gate = float(smooth_step(np.array([p.R_src - R_fixed]), p.w_surv)[0])
        S_ss = p.k_S_thresh * src_gate / (p.k_S_thresh * src_gate + p.d_S) if src_gate > 0.01 else 0.0

        W = float(W_R(np.array([R_fixed]), p.R_low, p.R_high, p.w1, p.w2)[0])
        A_ss = p.k_A * 1.0 * W * 1.0 / (p.k_A * 1.0 * W * 1.0 + p.d_A) if W > 0.01 else 0.0

        F_infl = p.w_L_Y10 * 1.0 + p.w_S_Y10 * S_ss + p.w_A_Y10 * A_ss
        Y10_ss = p.a_10 * F_infl / p.d_10

        results[regime] = {'Y9_ss': Y9_ss, 'Y10_ss': Y10_ss, 'R': R_fixed}

    # Check: Y9 and Y10 must NOT vary proportionally across regimes
    y9_vals = [r['Y9_ss'] for r in results.values()]
    y10_vals = [r['Y10_ss'] for r in results.values()]

    # Correlation (handle edge cases)
    if np.std(y9_vals) > 0.01 and np.std(y10_vals) > 0.01:
        corr = np.corrcoef(y9_vals, y10_vals)[0, 1]
    else:
        corr = 0.0

    passes = abs(corr) < 0.95  # Must NOT be perfectly correlated

    return {
        'constraint': 'HC-4: Motor orthogonality',
        'passes': passes,
        'correlation': corr,
        'regime_values': results,
    }


def validate_HC5(p: Params, sol=None) -> Dict:
    """
    HC-5: Morphology non-identifiability.
    Find time points where morphology is similar but inflammation differs.
    Requires a solved R2 trajectory.
    """
    return shared_validate_hc5(_legacy_to_allparams(p), sol)


def _legacy_to_allparams(p: Params) -> SharedAllParams:
    """Translate legacy flat Params into the shared attachment-style parameter object."""
    shared = shared_default_params()
    shared.thresh.R_surv = p.R_surv
    shared.thresh.R_src = p.R_src
    shared.thresh.R_low = p.R_low
    shared.thresh.R_high = p.R_high
    shared.thresh.w1 = p.w1
    shared.thresh.w2 = p.w2
    shared.thresh.w_surv = p.w_surv

    shared.l1.k_R_GEF = p.k_R_GEF
    shared.l1.alpha_S = p.alpha_S
    shared.l1.k_R_N = p.k_R_N
    shared.l1.k_R_L = p.k_R_L
    shared.l1.k_R_off = p.k_R_off
    shared.l1.k_A = p.k_A
    shared.l1.d_A = p.d_A
    shared.l1.k_H = p.k_H
    shared.l1.a_L = p.a_L
    shared.l1.a_R = p.a_R

    shared.l2.k_S = p.k_S_thresh
    shared.l2.d_S = p.d_S
    shared.l2.a_9 = p.a_9
    shared.l2.d_9 = p.d_9
    shared.l2.V_K = p.V_K
    shared.l2.K_R_rock = p.K_R_rock
    shared.l2.a_P = p.a_P
    shared.l2.a_C = p.a_C
    shared.l2.b_P = p.b_P
    shared.l2.d_P = p.d_P
    shared.l2.V_D = p.V_D
    shared.l2.K_R_mdia = p.K_R_mdia
    shared.l2.a_10 = p.a_10
    shared.l2.w_L_Y10 = p.w_L_Y10
    shared.l2.w_S_Y10 = p.w_S_Y10
    shared.l2.w_A_Y10 = p.w_A_Y10
    shared.l2.b_10 = p.b_10
    shared.l2.K_hyst = p.K_hyst
    shared.l2.d_10 = p.d_10

    shared.l3.k_restore = p.k_restore
    shared.l3.k_desens = p.k_desens
    shared.l3.K_half_desens = p.K_half_desens
    shared.l3.n_hill = p.n_hill
    shared.l3.k_casp = p.k_casp
    shared.l3.d_casp = p.d_casp
    shared.l3.enable_kcasp = p.enable_kcasp

    if hasattr(shared, 'src'):
        shared.src.k_S_thresh = p.k_S_thresh
        shared.src.k_S_abeta = p.k_S_abeta
        shared.src.eta_coup = p.eta_coup
        shared.src.K_coup = p.K_coup
        shared.src.src_mode = p.src_mode

    shared.obs.eps = p.eps
    shared.obs.H_crit = p.H_crit
    return shared


def validate_all(p: Params, sol=None) -> Dict:
    """Run all 5 hard-constraint checks."""
    assert p.validate_ordering(), f"Threshold ordering violated: {p.R_surv} < {p.R_src} < {p.R_low} < {p.R_high}"

    results = {
        'HC1': validate_HC1(p),
        'HC2': validate_HC2(p),
        'HC3': validate_HC3(p),
        'HC4': validate_HC4(p),
        'HC5': validate_HC5(p, sol),
    }

    all_pass = all(r['passes'] for r in results.values() if r['passes'] is not None)
    return {'all_pass': all_pass, 'results': results}


# ══════════════════════════════════════════════════════════
# SCENARIO RUNNERS
# ══════════════════════════════════════════════════════════

def run_three_regimes(p: Params, t_span=(0, 80)):
    """
    Demonstrate three RhoA regimes by running R2 with LPS that
    progressively reduces RhoA through all three zones.
    """
    # Strong, sustained LPS to push R from 0.6 down through R_low, R_src
    lps = make_lps(t_onset=10, amp=1.5, dur=60, tau=100)
    return solve_R2(p, L_func=lps, t_span=t_span)


def run_fixed_R_comparison(p: Params):
    """
    Compare steady-state observables at fixed R values in each regime.
    Direct test of motor orthogonality and non-identifiability.
    """
    results = {}
    for label, R_val in [('restrained', 0.8), ('reactive', 0.5),
                          ('vulnerable', 0.2), ('dysregulated', 0.08)]:
        # Approximate steady states at constant L=1.0, fixed R
        ROCK = p.V_K * R_val / (p.K_R_rock + R_val)
        mDia = p.V_D * R_val / (p.K_R_mdia + R_val)
        Y9_ss = p.a_9 * ROCK / p.d_9

        src_gate = float(smooth_step(np.array([p.R_src - R_val]), p.w_surv)[0])
        S_ss = p.k_S_thresh * src_gate / (p.k_S_thresh * src_gate + p.d_S) if src_gate > 0.01 else 0.0

        W = float(W_R(np.array([R_val]), p.R_low, p.R_high, p.w1, p.w2)[0])
        A_ss = p.k_A * 1.0 * W * 1.0 / (p.k_A * W + p.d_A) if W > 0.01 else 0.0

        surv_dmg = p.a_R * float(smooth_step(np.array([p.R_surv - R_val]), p.w_surv)[0])
        H_ss = p.k_H / (p.k_H + p.a_L * 1.0 + surv_dmg)

        F_infl = p.w_L_Y10 * 1.0 + p.w_S_Y10 * S_ss + p.w_A_Y10 * A_ss
        Y10_ss = p.a_10 * F_infl / p.d_10

        P_ss = (p.a_P * mDia + p.a_C) / (p.b_P * Y9_ss + p.d_P)

        N_tox = S_ss * (1 - H_ss)
        M_morph = P_ss / (P_ss + Y9_ss + p.eps)
        I_infl = 0.3 * A_ss + 0.4 * S_ss + 0.3 * Y10_ss

        results[label] = {
            'R': R_val, 'H': H_ss, 'A': A_ss, 'S': S_ss,
            'Y9': Y9_ss, 'P': P_ss, 'Y10': Y10_ss,
            'N_tox': N_tox, 'M_morph': M_morph, 'I_infl': I_infl,
            'regime': classify_regime(R_val, p),
        }
    return results


def generate_validation_summary(
    p: Optional[Params] = None,
    include_attachment_compare: bool = False,
) -> Dict[str, Any]:
    """Run the current validation workflow and return a structured summary."""
    p = p or Params()
    if not p.validate_ordering():
        raise ValueError(
            f"Threshold ordering violated: {p.R_surv} < {p.R_src} < {p.R_low} < {p.R_high}"
        )

    summary: Dict[str, Any] = {
        'model_version': 'v3',
        'thresholds': {
            'R_surv': p.R_surv,
            'R_src': p.R_src,
            'R_low': p.R_low,
            'R_high': p.R_high,
        },
        'window_samples': [],
    }

    for R_val in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        W = float(W_R(np.array([R_val]), p.R_low, p.R_high, p.w1, p.w2)[0])
        summary['window_samples'].append(
            {
                'R': R_val,
                'W_R': W,
                'regime': classify_regime(R_val, p),
            }
        )

    hc_static = validate_all(p)
    regimes = run_fixed_R_comparison(p)
    sol = run_three_regimes(p)

    dynamic_summary: Dict[str, Any] = {
        'success': bool(sol.success),
        'samples': [],
    }
    if sol.success:
        obs = compute_observables(sol.y, p)
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            idx = int(frac * (len(sol.t) - 1))
            dynamic_summary['samples'].append(
                {
                    't': float(sol.t[idx]),
                    'R': float(sol.y[0, idx]),
                    'regime': classify_regime(float(sol.y[0, idx]), p),
                    'Phi': float(obs['Phi'][idx]),
                    'Xi': float(obs['Xi'][idx]),
                    'Omega': float(obs['Omega'][idx]),
                    'N_tox': float(obs['N_tox'][idx]),
                }
            )
        dynamic_summary['hc5'] = validate_HC5(p, sol)
    else:
        dynamic_summary['solver_message'] = sol.message
        dynamic_summary['hc5'] = validate_HC5(p, None)

    y9_list = [regimes[r]['Y9'] for r in regimes]
    y10_list = [regimes[r]['Y10'] for r in regimes]
    if np.std(y9_list) > 0.01 and np.std(y10_list) > 0.01:
        motor_corr = float(np.corrcoef(y9_list, y10_list)[0, 1])
    else:
        motor_corr = None

    summary.update(
        {
            'hard_constraints_static': hc_static,
            'fixed_R_regimes': regimes,
            'dynamic_validation': dynamic_summary,
            'motor_orthogonality': {
                'correlation': motor_corr,
                'status': (
                    'orthogonal'
                    if motor_corr is not None and abs(motor_corr) < 0.95
                    else 'coupled_or_insufficient_variance'
                ),
            },
            'validation_complete': True,
        }
    )
    if include_attachment_compare:
        from windowed_rhoa.step1.hc4_orthogonality import run_step1

        summary['attachment_compare'] = {
            'step1_dynamic_cofilin': run_step1(shared_default_params())
        }
    return summary


def format_validation_summary(summary: Dict[str, Any]) -> List[str]:
    """Render the validation summary in the original CLI format."""
    thresholds = summary['thresholds']
    hc_static = summary['hard_constraints_static']
    regimes = summary['fixed_R_regimes']
    dynamic = summary['dynamic_validation']
    motor = summary['motor_orthogonality']

    lines = [
        "=" * 65,
        "WINDOWED-RhoA MODEL v3 — VALIDATION",
        "=" * 65,
        "",
        (
            f"Threshold ordering: R_surv={thresholds['R_surv']} < "
            f"R_src={thresholds['R_src']} < R_low={thresholds['R_low']} "
            f"< R_high={thresholds['R_high']}"
        ),
        "  ✓ Ordering valid",
        "",
        "--- Window function W_R ---",
    ]
    for sample in summary['window_samples']:
        lines.append(
            f"  R={sample['R']:.1f}: W_R={sample['W_R']:.3f} ({sample['regime']})"
        )

    lines.extend(["", "--- Hard-constraint validation (static) ---"])
    for result in hc_static['results'].values():
        status = '✓' if result['passes'] else ('?' if result['passes'] is None else '✗')
        lines.append(f"  {status} {result['constraint']}")

    lines.extend(
        [
            "",
            "--- Fixed-R regime comparison (L=1.0 steady states) ---",
            (
                f"  {'Regime':<16} {'R':>4} {'H':>6} {'A':>6} {'S':>6} "
                f"{'Y9':>6} {'P':>6} {'Y10':>6} {'N_tox':>6} {'M':>6} {'I':>6}"
            ),
        ]
    )
    for label, result in regimes.items():
        lines.append(
            f"  {label:<16} {result['R']:>4.2f} {result['H']:>6.3f} "
            f"{result['A']:>6.3f} {result['S']:>6.3f} {result['Y9']:>6.3f} "
            f"{result['P']:>6.3f} {result['Y10']:>6.3f} {result['N_tox']:>6.3f} "
            f"{result['M_morph']:>6.3f} {result['I_infl']:>6.3f}"
        )

    lines.extend(["", "--- Dynamic R2: LPS driving R through all regimes ---"])
    if dynamic['success']:
        for sample in dynamic['samples']:
            lines.append(
                f"  t={sample['t']:>5.1f}: R={sample['R']:.3f} "
                f"({sample['regime']:<22s}) Φ={sample['Phi']:.3f} "
                f"Ξ={sample['Xi']:.2f} Ω={sample['Omega']:.3f} "
                f"N_tox={sample['N_tox']:.3f}"
            )
        hc5 = dynamic['hc5']
        status5 = '✓' if hc5['passes'] else '✗'
        lines.append("")
        lines.append(f"  {status5} {hc5['constraint']}")
    else:
        lines.append(f"  Solver failed: {dynamic['solver_message']}")

    lines.extend(["", "--- Motor orthogonality (Y9 vs Y10 across regimes) ---"])
    if motor['correlation'] is None:
        lines.append("  Insufficient variance for correlation")
    else:
        lines.append(
            f"  Correlation(Y9, Y10) = {motor['correlation']:.3f} "
            f"({'ORTHOGONAL' if abs(motor['correlation']) < 0.95 else 'COUPLED'})"
        )

    lines.extend(["", "✓ v3 validation complete."])
    return lines


def main() -> int:
    """CLI entrypoint preserving the original validation output."""
    summary = generate_validation_summary()
    for line in format_validation_summary(summary):
        print(line)
    return 0


# ══════════════════════════════════════════════════════════
# MAIN VALIDATION
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    raise SystemExit(main())
