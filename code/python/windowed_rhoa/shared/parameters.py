"""
parameters.py — Canonical parameter set for windowed-RhoA model.

SINGLE SOURCE OF TRUTH. All pipeline modules import from here.
No hardcoded parameters elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from dataclasses import dataclass


ROOT = Path(__file__).resolve().parents[4]
CALCIUM_CROSS_BRANCH_PATH = ROOT / "results" / "exports" / "calcium_hc3_inference" / "calcium_hc3_cross_branch_integration.json"


@dataclass
class ThresholdParams:
    """RhoA regime thresholds. Ordering enforced: R_surv < R_src < R_low < R_high."""
    R_surv: float = 0.05
    R_src: float = 0.15
    R_low: float = 0.30
    R_high: float = 0.70
    w1: float = 0.05          # Window sigmoid width (lower)
    w2: float = 0.05          # Window sigmoid width (upper)
    w_surv: float = 0.02      # Survival step smoothing

    def validate(self):
        assert self.R_surv < self.R_src < self.R_low < self.R_high, \
            f"Ordering violated: {self.R_surv} < {self.R_src} < {self.R_low} < {self.R_high}"
        return True


@dataclass
class Layer1Params:
    """Layer 1: Windowed RhoA + Survival."""
    k_R_GEF: float = 0.4
    alpha_S: float = 0.5       # Src→GEF-H1 feedback (Azoitei 2019)
    k_R_N: float = 0.3         # Narciclasine → RhoA
    k_R_L: float = 0.8         # LPS → RhoA decrease
    k_R_off: float = 0.3       # GAP-mediated inactivation
    k_A: float = 0.8           # Productive reactivity rate
    d_A: float = 0.3           # Reactivity decay
    k_H: float = 0.1           # Homeostatic recovery (slow)
    a_L: float = 0.15          # LPS-mediated H degradation
    a_R: float = 0.5           # Low-RhoA H collapse


@dataclass
class CofilinParams:
    """Dynamic cofilin regulation (Step 1A upgrade)."""
    C_total: float = 1.0       # Total cofilin pool (normalized)
    k_ssh1: float = 0.6        # SSH1-mediated activation rate (LPS-dependent)
    k_LIMK: float = 0.4        # LIMK-mediated inactivation (ROCK-dependent)
    k_ssh1_basal: float = 0.1  # Basal SSH1 activity (LPS-independent)
    C_active_0: float = 0.3    # Initial active cofilin (basal turnover)


@dataclass
class Layer2Params:
    """Layer 2: Motor-resolved outputs."""
    # Src
    k_S: float = 0.5
    d_S: float = 0.2
    # Myh9
    a_9: float = 0.8
    d_9: float = 0.4
    V_K: float = 1.0           # Max ROCK
    K_R_rock: float = 0.3
    # Protrusive support
    a_P: float = 0.5           # mDia → protrusion
    a_C: float = 0.3           # Cofilin → protrusion (STATIC, replaced in Step 1A)
    b_P: float = 0.3           # Y9 antagonism
    d_P: float = 0.2
    V_D: float = 1.0           # Max mDia
    K_R_mdia: float = 0.25
    # Myh10
    a_10: float = 0.4
    w_L_Y10: float = 0.3       # LPS weight
    w_S_Y10: float = 0.5       # Src weight
    w_A_Y10: float = 0.2       # Reactivity weight
    b_10: float = 0.15         # Self-reinforcement
    K_hyst: float = 0.3        # Commitment threshold
    d_10: float = 0.1


@dataclass
class SrcCouplingParams:
    """Optional Aβ/FRET reconciliation branch for Src kinetics."""
    src_mode: str = "threshold_only"
    k_S_thresh: float = 0.5
    k_S_abeta: float = 0.35
    eta_coup: float = 0.35
    K_coup: float = 0.25


@dataclass
class Layer3Params:
    """Layer 3: Disease extension (demoted hypothesis)."""
    k_restore: float = 0.15
    k_desens: float = 0.25
    K_half_desens: float = 0.5
    n_hill: int = 3
    k_casp: float = 0.3        # ⚠ HYPOTHESIS
    d_casp: float = 0.08
    enable_kcasp: bool = True


@dataclass
class GillespieParams:
    """Gillespie SSA configuration (Step 2)."""
    omega: int = 500            # System size
    n_trajectories: int = 1000
    t_max: float = 200.0
    record_dt: float = 0.1
    seed: int = 42


@dataclass
class InferenceParams:
    """Bayesian inference configuration (Step 3)."""
    n_chains: int = 4
    n_warmup: int = 2000
    n_sampling: int = 2000
    adapt_delta: float = 0.95
    max_treedepth: int = 12


@dataclass
class ObservableParams:
    """Observable definitions and noise models."""
    eps: float = 0.01
    # Operational reserve-transition reference used for calibrated duration and
    # first-passage observer definitions. This effective state-space reference
    # is not a molecular, cell-death, or tissue-injury threshold.
    H_crit: float = 0.5
    # Noise std for synthetic data (Step 3A)
    sigma_TNF: float = 0.1
    sigma_pSrc: float = 0.15
    sigma_morph: float = 0.08
    sigma_viab: float = 0.1
    # Non-identifiability thresholds (Step 1C)
    eps_M: float = 0.05        # Morphology similarity threshold
    delta_I: float = 0.15      # Inflammation difference threshold


@dataclass
class AllParams:
    """Master container — single import for all modules."""
    thresh: ThresholdParams = None
    l1: Layer1Params = None
    cof: CofilinParams = None
    l2: Layer2Params = None
    src: SrcCouplingParams = None
    l3: Layer3Params = None
    gill: GillespieParams = None
    inf: InferenceParams = None
    obs: ObservableParams = None

    def __post_init__(self):
        self.thresh = self.thresh or ThresholdParams()
        self.l1 = self.l1 or Layer1Params()
        self.cof = self.cof or CofilinParams()
        self.l2 = self.l2 or Layer2Params()
        self.src = self.src or SrcCouplingParams()
        self.l3 = self.l3 or Layer3Params()
        self.gill = self.gill or GillespieParams()
        self.inf = self.inf or InferenceParams()
        self.obs = self.obs or ObservableParams()

    def validate(self):
        self.thresh.validate()
        return True


# ── Convenience: default parameter set ──
def _load_calcium_cross_branch_updates() -> dict[str, float] | None:
    if not CALCIUM_CROSS_BRANCH_PATH.exists():
        return None
    try:
        payload = json.loads(CALCIUM_CROSS_BRANCH_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not bool(payload.get("propagation_enabled")):
        return None
    updates = payload.get("shared_layer1_updates")
    if not isinstance(updates, dict):
        return None
    keys = ("a_L", "a_R", "k_H")
    if not all(key in updates for key in keys):
        return None
    return {key: float(updates[key]) for key in keys}


def default_params() -> AllParams:
    p = AllParams()
    calcium_updates = _load_calcium_cross_branch_updates()
    if calcium_updates:
        p.l1.a_L = calcium_updates["a_L"]
        p.l1.a_R = calcium_updates["a_R"]
        p.l1.k_H = calcium_updates["k_H"]
    p.validate()
    return p


if __name__ == '__main__':
    p = default_params()
    print(f"Thresholds: R_surv={p.thresh.R_surv}, R_src={p.thresh.R_src}, "
          f"R_low={p.thresh.R_low}, R_high={p.thresh.R_high}")
    print(f"Ordering valid: {p.thresh.validate()}")
    print(f"Cofilin: C_total={p.cof.C_total}, k_ssh1={p.cof.k_ssh1}, k_LIMK={p.cof.k_LIMK}")
    print(f"Gillespie: Ω={p.gill.omega}, n_traj={p.gill.n_trajectories}")
    print("✓ All parameters loaded.")
