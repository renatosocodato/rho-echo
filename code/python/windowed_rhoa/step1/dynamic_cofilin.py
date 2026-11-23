"""
Step 1A — Dynamic cofilin regulation.

Replaces the constant cofilin turnover term `a_C` with an SSH1-dependent
dephosphorylation module, making protrusion extension dynamically regulated
by inflammatory input (LPS → SSH1 → cofilin activation → actin turnover).

Biology: LPS activates SSH1 (slingshot phosphatase 1), which dephosphorylates
cofilin, increasing active cofilin and thus actin turnover / protrusion extension.
ROCK phosphorylates LIMK, which phosphorylates (inactivates) cofilin.
Net cofilin activity = balance of SSH1 (activating) vs LIMK/ROCK (inactivating).

References:
- Melo et al. (2021) — cofilin-mediated actin turnover in microglial protrusion
- Glotfelty et al. (2023) — ROCK inhibitors block cofilin dephosphorylation
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Dict, Optional
from windowed_rhoa.shared.math_utils import W_R, sigmoid, smooth_step
from windowed_rhoa.shared.parameters import AllParams, default_params


# ══════════════════════════════════════════════════════════
# R2 MODEL WITH DYNAMIC COFILIN (8 variables)
# ══════════════════════════════════════════════════════════

def R2_cofilin_rhs(t, y, p: AllParams, L_func, N_func):
    """
    Layer 1+2 model with dynamic cofilin: 8 state variables.

    State vector: [R, H, A, S, Y9, P, Y10, C_active]

    0: R        — RhoA-GTP [0, 1]
    1: H        — Homeostatic reserve [0, 1]
    2: A        — Productive reactivity [0, 1]
    3: S        — Src activity [0, 1]
    4: Y9       — Myh9 (contractile)
    5: P        — Protrusive support (mDia + cofilin)
    6: Y10      — Myh10 (inflammatory motor)
    7: C_active — Active (dephosphorylated) cofilin [0, C_total]
    """
    R, H, A, S, Y9, P, Y10, C_act = y
    R     = np.clip(R, 0, 1)
    H     = np.clip(H, 0, 1)
    A     = np.clip(A, 0, 1)
    S     = np.clip(S, 0, 1)
    Y9    = max(Y9, 0)
    P     = max(P, 0)
    Y10   = max(Y10, 0)
    C_act = np.clip(C_act, 0, p.cof.C_total)

    L = L_func(t)
    N = N_func(t)
    th = p.thresh
    l1 = p.l1
    l2 = p.l2
    cof = p.cof

    W = float(W_R(np.array([R]), th.R_low, th.R_high, th.w1, th.w2)[0])

    # ── Layer 1: RhoA, H, A ──
    T_fixed = 0.5  # MT integrity (fixed in R2; dynamic in full model)
    gef_h1 = (1 - T_fixed) * (1 + l1.alpha_S * S)
    dR = (l1.k_R_GEF * gef_h1 * (1 - R)
          + l1.k_R_N * N * (1 - R)
          - l1.k_R_L * L * R
          - l1.k_R_off * R)

    surv_dmg = l1.a_R * float(smooth_step(np.array([th.R_surv - R]), th.w_surv)[0])
    dH = l1.k_H * (1 - H) - l1.a_L * L * H - surv_dmg * H

    dA = l1.k_A * L * W * H * (1 - A) - l1.d_A * A

    # ── Layer 2: Src ──
    src_gate = float(smooth_step(np.array([th.R_src - R]), th.w_surv)[0])
    dS = l2.k_S * src_gate * (1 - S) - l2.d_S * S

    # ── ROCK and mDia ──
    ROCK = l2.V_K * R / (l2.K_R_rock + R)
    mDia = l2.V_D * R / (l2.K_R_mdia + R)

    # ── Dynamic cofilin (NEW) ──
    # SSH1 activation: basal + LPS-driven
    ssh1_activity = cof.k_ssh1_basal + cof.k_ssh1 * L
    # LIMK inactivation: ROCK-driven
    limk_activity = cof.k_LIMK * ROCK
    # Cofilin dynamics: SSH1 activates, LIMK inactivates
    C_inactive = cof.C_total - C_act
    dC_act = ssh1_activity * C_inactive - limk_activity * C_act

    # ── Myh9 ──
    dY9 = l2.a_9 * ROCK - l2.d_9 * Y9

    # ── Protrusive support (uses dynamic C_active instead of constant a_C) ──
    dP = l2.a_P * mDia + l2.a_C * C_act - l2.b_P * Y9 * P - l2.d_P * P

    # ── Myh10 ──
    F_infl = l2.w_L_Y10 * L + l2.w_S_Y10 * S + l2.w_A_Y10 * A
    h_Y10 = Y10**2 / (l2.K_hyst**2 + Y10**2)
    dY10 = l2.a_10 * F_infl + l2.b_10 * h_Y10 - l2.d_10 * Y10

    return [dR, dH, dA, dS, dY9, dP, dY10, dC_act]


def solve_R2_cofilin(p: AllParams, L_func=None, N_func=None,
                     t_span=(0, 100), y0=None, **kw):
    """Solve the R2 model with dynamic cofilin."""
    L_func = L_func or (lambda t: 0.0)
    N_func = N_func or (lambda t: 0.0)
    if y0 is None:
        y0 = [0.6, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, p.cof.C_active_0]
    t_eval = kw.pop('t_eval', np.linspace(*t_span, 5000))
    return solve_ivp(
        R2_cofilin_rhs, t_span, y0,
        args=(p, L_func, N_func),
        t_eval=t_eval, method='BDF',
        rtol=1e-8, atol=1e-10, max_step=0.1, **kw
    )


def clamped_cofilin_equilibrium(R, L, p: AllParams, C_clamp=None):
    """Stationary free coordinates at fixed R and L, with optional fixed cofilin.

    The upstream coordinates are equilibrated first. Myh10 is then
    equilibrated from zero with its drive held at that equilibrium value.
    The returned roots document any alternative nonnegative fixed points.
    """
    th, l1, l2, cof = p.thresh, p.l1, p.l2, p.cof
    low = float(smooth_step(th.R_surv - R, th.w_surv))
    gate = float(smooth_step(th.R_src - R, th.w_surv))
    window = float(W_R(R, th.R_low, th.R_high, th.w1, th.w2))
    H = l1.k_H / (l1.k_H + l1.a_L * L + l1.a_R * low)
    activation = l1.k_A * L * window * H
    A = activation / (activation + l1.d_A)
    S = l2.k_S * gate / (l2.k_S * gate + l2.d_S)
    rock = l2.V_K * R / (l2.K_R_rock + R)
    mdia = l2.V_D * R / (l2.K_R_mdia + R)
    ssh = cof.k_ssh1_basal + cof.k_ssh1 * L
    C = cof.C_total * ssh / (ssh + cof.k_LIMK * rock) if C_clamp is None else C_clamp
    Y9 = l2.a_9 * rock / l2.d_9
    P = (l2.a_P * mdia + l2.a_C * C) / (l2.d_P + l2.b_P * Y9)
    drive = l2.a_10 * (l2.w_L_Y10 * L + l2.w_S_Y10 * S + l2.w_A_Y10 * A)
    K2 = l2.K_hyst**2
    roots = np.roots([l2.d_10, -(drive + l2.b_10), l2.d_10 * K2, -drive * K2])
    roots = sorted(float(root.real) for root in roots if abs(root.imag) < 1e-9 and root.real >= 0)
    Y10 = roots[0]
    state = np.array([R, H, A, S, Y9, P, Y10, C])
    return state, roots


# ══════════════════════════════════════════════════════════
# STEADY-STATE SOLVER (for grid-based HC-4 and HC-5 tests)
# ══════════════════════════════════════════════════════════

def compute_steady_state(R_fixed: float, L_fixed: float,
                         p: AllParams) -> Dict:
    """
    Legacy algebraic surrogate at fixed R and L for the historical HC-4/HC-5
    and synthetic-observation exports. For stationary R2/cofilin states use
    clamped_cofilin_equilibrium.
    """
    th, l1, l2, cof = p.thresh, p.l1, p.l2, p.cof
    eps = p.obs.eps

    # ROCK, mDia at fixed R
    ROCK = l2.V_K * R_fixed / (l2.K_R_rock + R_fixed)
    mDia = l2.V_D * R_fixed / (l2.K_R_mdia + R_fixed)

    # Src
    src_gate = float(smooth_step(np.array([th.R_src - R_fixed]), th.w_surv)[0])
    S_ss = l2.k_S * src_gate / (l2.k_S * src_gate + l2.d_S) if src_gate > 0.01 else 0.0

    # Window, reactivity
    W = float(W_R(np.array([R_fixed]), th.R_low, th.R_high, th.w1, th.w2)[0])
    denom_A = l1.k_A * L_fixed * W * 1.0 + l1.d_A
    A_ss = l1.k_A * L_fixed * W * 1.0 / denom_A if (L_fixed > 0 and W > 0.01) else 0.0

    # Survival
    surv_dmg = l1.a_R * float(smooth_step(np.array([th.R_surv - R_fixed]), th.w_surv)[0])
    H_ss = l1.k_H / (l1.k_H + l1.a_L * L_fixed + surv_dmg)

    # Dynamic cofilin
    ssh1 = cof.k_ssh1_basal + cof.k_ssh1 * L_fixed
    limk = cof.k_LIMK * ROCK
    C_act_ss = ssh1 * cof.C_total / (ssh1 + limk) if (ssh1 + limk) > 0 else 0.0

    # Y9
    Y9_ss = l2.a_9 * ROCK / l2.d_9

    # P (protrusive support with dynamic cofilin)
    P_num = l2.a_P * mDia + l2.a_C * C_act_ss
    P_den = l2.b_P * Y9_ss + l2.d_P
    P_ss = P_num / P_den if P_den > 0 else 0.0

    # Y10
    F_infl = l2.w_L_Y10 * L_fixed + l2.w_S_Y10 * S_ss + l2.w_A_Y10 * A_ss
    Y10_ss = l2.a_10 * F_infl / l2.d_10

    # Observables
    N_tox = S_ss * (1 - H_ss)
    M_morph = P_ss / (P_ss + Y9_ss + eps)
    I_infl = 0.3 * A_ss + 0.4 * S_ss + 0.3 * Y10_ss

    return {
        'R': R_fixed, 'L': L_fixed, 'H': H_ss, 'A': A_ss, 'S': S_ss,
        'Y9': Y9_ss, 'P': P_ss, 'Y10': Y10_ss, 'C_act': C_act_ss,
        'N_tox': N_tox, 'M_morph': M_morph, 'I_infl': I_infl,
        'ROCK': ROCK, 'mDia': mDia,
    }


# ══════════════════════════════════════════════════════════
# GATE 1A VALIDATION
# ══════════════════════════════════════════════════════════

def validate_gate_1A(p: AllParams) -> Dict:
    """
    Gate 1A: Updated R2 model integrates stably; C_active responds to LPS.
    """
    results = {}

    # Test 1: Integration stability (no LPS)
    sol_0 = solve_R2_cofilin(p, L_func=lambda t: 0.0, t_span=(0, 50))
    results['integrates_no_lps'] = sol_0.success

    # Test 2: Integration stability (with LPS)
    def lps(t):
        return 1.0 if t >= 10 else 0.0
    sol_L = solve_R2_cofilin(p, L_func=lps, t_span=(0, 50))
    results['integrates_with_lps'] = sol_L.success

    # Test 3: C_active responds to LPS
    if sol_L.success:
        C_before = sol_L.y[7, np.searchsorted(sol_L.t, 9)]
        C_after = sol_L.y[7, np.searchsorted(sol_L.t, 30)]
        results['cofilin_responds'] = C_after > C_before * 1.1  # >10% increase
        results['C_before_lps'] = float(C_before)
        results['C_after_lps'] = float(C_after)

    # Test 4: P dynamics change with cofilin
    if sol_0.success and sol_L.success:
        P_no_lps = sol_0.y[5, -1]
        P_with_lps = sol_L.y[5, np.searchsorted(sol_L.t, 30)]
        results['P_responds_to_cofilin'] = abs(P_with_lps - P_no_lps) > 0.01
        results['P_no_lps'] = float(P_no_lps)
        results['P_with_lps'] = float(P_with_lps)

    results['gate_passes'] = all([
        results.get('integrates_no_lps', False),
        results.get('integrates_with_lps', False),
        results.get('cofilin_responds', False),
        results.get('P_responds_to_cofilin', False),
    ])

    return results


if __name__ == '__main__':
    p = default_params()

    print("=" * 60)
    print("STEP 1A — Dynamic Cofilin Validation")
    print("=" * 60)

    # Gate 1A
    g1a = validate_gate_1A(p)
    for k, v in g1a.items():
        print(f"  {k}: {v}")

    # Quick steady-state comparison: static vs dynamic cofilin
    print("\n--- Cofilin effect on steady states ---")
    for L_val in [0.0, 0.5, 1.0]:
        ss = compute_steady_state(0.5, L_val, p)
        print(f"  R=0.5, L={L_val}: C_act={ss['C_act']:.3f}, "
              f"P={ss['P']:.3f}, Y9={ss['Y9']:.3f}, Y10={ss['Y10']:.3f}")

    status = "✓ PASS" if g1a['gate_passes'] else "✗ FAIL"
    print(f"\nGATE 1A: {status}")
