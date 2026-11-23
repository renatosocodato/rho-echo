"""Density-dependent, exact R2/cofilin jump process for piecewise constant L and N=0."""

from fractions import Fraction

import numpy as np
from numba import njit
from scipy.special import expit


def process_parameters(p, omega):
    """Pin the integer lattice and precompute the R-dependent factors."""
    pool = round(omega * p.cof.C_total)
    if not np.isclose(pool, omega * p.cof.C_total, rtol=0, atol=1e-12):
        raise ValueError("C_total must lie on the count lattice")
    r = np.arange(omega + 1) / omega
    th, l1, l2, cof = p.thresh, p.l1, p.l2, p.cof
    factors = np.column_stack((
        expit((th.R_surv - r) / th.w_surv),
        expit((r - th.R_low) / th.w1) * expit((th.R_high - r) / th.w2),
        expit((th.R_src - r) / th.w_surv),
        l2.V_K * r / (l2.K_R_rock + r),
        l2.V_D * r / (l2.K_R_mdia + r),
    ))
    pars = (
        l1.k_R_GEF, l1.alpha_S, l1.k_R_off, l1.k_R_L, l1.k_H, l1.a_L, l1.a_R,
        l1.k_A, l1.d_A, l2.k_S, l2.d_S, l2.a_9, l2.d_9, l2.a_P, l2.a_C,
        l2.b_P, l2.d_P, l2.a_10, l2.w_L_Y10, l2.w_S_Y10, l2.w_A_Y10,
        l2.b_10, l2.K_hyst**2, l2.d_10, cof.k_ssh1_basal, cof.k_ssh1, cof.k_LIMK,
    )
    return pool, factors, pars


@njit
def fill_rates(n, L, omega, pool, factors, pars, rates):
    """Birth and death intensities, adjacent for each count coordinate."""
    R, H, A, S, Y9, P, Y10, C = n
    low, window, src, rock, mdia = factors[R]
    (gef, feedback, off, load_R, repair, load_H, low_H, act, dA, src_on, dS,
     a9, d9, aP, aC, bP, dP, a10, wL, wS, wA, b10, K2, d10, basal, ssh, limk) = pars
    s = S / omega
    y = Y10 / omega
    rates[0] = 0.5 * gef * (1 + feedback * s) * (omega - R)
    rates[1] = (off + load_R * L) * R
    rates[2] = repair * (omega - H)
    rates[3] = (load_H * L + low_H * low) * H
    rates[4] = act * L * window * (H / omega) * (omega - A)
    rates[5] = dA * A
    rates[6] = src_on * src * (omega - S)
    rates[7] = dS * S
    rates[8] = omega * a9 * rock
    rates[9] = d9 * Y9
    rates[10] = omega * aP * mdia + aC * C
    rates[11] = (dP + bP * (Y9 / omega)) * P
    rates[12] = omega * (a10 * (wL * L + wS * s + wA * (A / omega)) + b10 * y*y / (K2 + y*y))
    rates[13] = d10 * Y10
    rates[14] = (basal + ssh * L) * (pool - C)
    rates[15] = limk * rock * C


def morphology_bounds(eps, half_width):
    """Convert the declared decimal cutoffs to rational count coefficients."""
    offset = Fraction(str(eps))
    width = Fraction(str(half_width))
    lower = Fraction(1, 2) - width
    upper = Fraction(1, 2) + width
    return (offset.numerator, offset.denominator, lower.numerator, lower.denominator,
            upper.numerator, upper.denominator)


@njit
def observe(n, t, omega, boundaries, morph_bounds, events):
    """Four marginal first entries for the common initial state inside the window."""
    low, high, src, reserve, _, _ = boundaries
    r, h = n[0] / omega, n[1] / omega
    if np.isnan(events[0]) and (r < low or r > high):
        events[0] = t
    if np.isnan(events[1]) and r < src:
        events[1] = t
    if np.isnan(events[2]) and t >= 5.0:
        offset_n, offset_d, lower_n, lower_d, upper_n, upper_d = morph_bounds
        numerator = n[5] * offset_d
        denominator = (n[5] + n[4]) * offset_d + omega * offset_n
        # M is a ratio of counts. Cross multiply so equality is always excluded.
        inside_band = (lower_n * denominator < lower_d * numerator
                       and upper_d * numerator < upper_n * denominator)
        infl_above = 3 * n[2] + 4 * n[3] + 3 * n[6] > 4 * omega
        if inside_band and infl_above:
            events[2] = t
    if np.isnan(events[3]) and h < reserve:
        events[3] = t


def simulate(initial, omega, pool, factors, pars, boundaries, n_paths, n_saved, seed):
    """Run the exact jump process with strict count-based mismatch boundaries."""
    return _simulate(initial, omega, pool, factors, pars, boundaries,
                     morphology_bounds(*boundaries[4:]), n_paths, n_saved, seed)


@njit(nogil=True)
def _simulate(initial, omega, pool, factors, pars, boundaries, morph_bounds, n_paths, n_saved, seed):
    """Run SSA, restarting the exponential clock at the input step or censoring horizon."""
    np.random.seed(seed)
    horizon = 60.0
    times = np.arange(301) * 0.2
    paths = np.empty((n_saved, len(times), 8), np.float64)
    events = np.full((n_paths, 4), np.nan)
    rates = np.empty(16, np.float64)
    for i in range(n_paths):
        n = initial.copy()
        t = 0.0
        j = 0
        while t < horizon:
            L = 0.0 if t < 5.0 else 1.0
            boundary = 5.0 if L == 0.0 else horizon
            fill_rates(n, L, omega, pool, factors, pars, rates)
            total = rates.sum()
            proposed = t + np.random.exponential(1 / total) if total > 0 else np.inf
            end = min(proposed, boundary)
            if i < n_saved:
                while j < len(times) and times[j] <= end:
                    paths[i, j] = n / omega
                    j += 1
            t = end
            if proposed < boundary:
                target = np.random.random() * total
                cumulative = 0.0
                for channel in range(16):
                    cumulative += rates[channel]
                    if target < cumulative:
                        n[channel // 2] += 1 if channel % 2 == 0 else -1
                        break
            observe(n, t, omega, boundaries, morph_bounds, events[i])
            if i >= n_saved and np.isfinite(events[i]).all():
                break
    return times, paths, events
