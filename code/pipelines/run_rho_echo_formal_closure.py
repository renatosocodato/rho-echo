"""Generate equilibria and exact first passages for the revised Rho-Echo figures."""

import argparse
import csv
import json
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/python"))

from windowed_rhoa.shared.parameters import default_params
from windowed_rhoa.step1.dynamic_cofilin import R2_cofilin_rhs, clamped_cofilin_equilibrium
from windowed_rhoa.step2.matched_jump import fill_rates, process_parameters, simulate

OMEGAS = (200, 500, 2000)
STATES = ("R", "H", "A", "S", "Y9", "P", "Y10", "C_active")
EVENTS = ("t_exit", "t_src", "t_mismatch", "t_surv")
LABELS = ("Window exit", "Src gate entry", "Morphology–inflammation mismatch", "Reserve crossing")
INITIAL = np.array([.43, .95, 0, 0, .5, .5, 0, .3])


def write_csv(out, name, rows):
    with (out / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def equilibrium_grid(p, out):
    groups = {name: [] for name in ("dynamic", "tied", "clamped")}
    maximum = 0.
    condition = 0
    for r0 in np.linspace(.05, .9, 20):
        for load in (0., .3, .6, .8, 1.2):
            for modulator in (0., .5, 1., 1.5, 2.):
                original_k = p.l1.k_A
                for mode, rows in groups.items():
                    r, L = r0, load
                    if mode == "tied":
                        r = float(np.clip(r0 - 1.5*load - .5*modulator, .05, .9))
                        L = .2
                    p.l1.k_A = original_k if mode == "tied" else original_k + .05*modulator
                    state, roots = clamped_cofilin_equilibrium(r, L, p, .5 if mode == "clamped" else None)
                    slope = np.array(R2_cofilin_rhs(0, state, p, lambda _: L, lambda _: 0))
                    free = slope[1:-1] if mode == "clamped" else slope[1:]
                    residual = float(np.max(np.abs(free)))
                    maximum = max(maximum, residual)
                    _, H, A, S, Y9, P, Y10, C = state
                    rows.append(dict(condition_id=condition, mode=mode, R0=r0, L=load, Abeta=modulator,
                                     R_effective=r, L_effective=L, H_ss=H, A_ss=A, S_ss=S,
                                     Y9_ss=Y9, P_ss=P, Y10_ss=Y10, C_active_ss=C,
                                     M_morph=P/(P+Y9+p.obs.eps), I_infl=.3*A+.4*S+.3*Y10,
                                     N_tox=S*(1-H), n_Y10_roots=len(roots),
                                     Y10_roots=";".join(f"{v:.16g}" for v in roots),
                                     max_residual=residual))
                p.l1.k_A = original_k
                condition += 1
    if maximum > 1e-10:
        raise RuntimeError(f"Equilibrium residual {maximum:g}")
    if out is None:
        return {k: float(np.corrcoef([r["M_morph"] for r in rows], [r["I_infl"] for r in rows])[0, 1])
                for k, rows in groups.items()}
    for mode, rows in groups.items():
        write_csv(out, f"equilibrium_{mode}.csv", rows)
    motor_diff = {key: max(abs(a[key]-b[key]) for a, b in zip(groups["dynamic"], groups["clamped"]))
                  for key in ("Y9_ss", "Y10_ss", "P_ss", "M_morph", "I_infl")}
    if any(motor_diff[key] != 0 for key in ("Y9_ss", "Y10_ss", "I_infl")):
        raise RuntimeError("Isolated cofilin clamp changed a motor or the inflammatory readout")
    return dict(max_residual=maximum, conditions_per_branch=condition, cofilin_clamp_max_difference=motor_diff,
                multiroot={k: sum(r["n_Y10_roots"] > 1 for r in rows) for k, rows in groups.items()},
                ranges={k: {col: [min(r[col] for r in rows), max(r[col] for r in rows)]
                            for col in ("C_active_ss", "P_ss", "Y10_ss")} for k, rows in groups.items()})


def verify_generator(p, omega, process):
    """Compare drift to the independent ODE and check the count-domain faces."""
    rng = np.random.default_rng(20261106)
    pool, _, _ = process
    ceilings = np.array([omega, omega, omega, omega, 5*omega, 5*omega, 5*omega, pool])
    points = rng.integers(0, ceilings + 1, size=(256, 8))
    points = np.concatenate((points, np.zeros((1, 8), int), ceilings[None, :]))
    for j in range(8):
        points[2*j, j], points[2*j+1, j] = 0, ceilings[j]
    largest = 0.
    for n in points:
        for L in (0., 1.):
            rates = np.empty(16)
            fill_rates(n, L, omega, *process, rates)
            rhs = R2_cofilin_rhs(0., n/omega, p, lambda _: L, lambda _: 0)
            drift = (rates[::2]-rates[1::2])/omega
            largest = max(largest, float(np.max(np.abs(rhs-drift))))
            assert np.all(rates >= 0)
            assert np.all(rates[1::2][n == 0] == 0)
            bounded = np.array([True, True, True, True, False, False, False, True])
            assert np.all(rates[::2][(n == ceilings) & bounded] == 0)
    if largest > 1e-11:
        raise RuntimeError(f"Generator drift mismatch {largest:g}")
    return largest


def deterministic_reference(p, times):
    initial = INITIAL
    states = []
    for start, end, load in ((0., 5., 0.), (5., 60., 1.)):
        t_eval = times[(times >= start) & (times <= end)]
        sol = solve_ivp(R2_cofilin_rhs, (start, end), initial, t_eval=t_eval,
                        args=(p, lambda _: load, lambda _: 0.), method="BDF",
                        rtol=1e-9, atol=1e-11, max_step=.1)
        if not sol.success:
            raise RuntimeError(sol.message)
        initial = sol.y[:, -1]
        states.extend(sol.y.T if start == 0 else sol.y.T[1:])
    return [dict(t=t, R_det=s[0], H_det=s[1], S_det=s[3], R_src=p.thresh.R_src, H_crit=p.obs.H_crit)
            for t, s in zip(times, states)]


def summarize(events):
    result = {}
    for key, column in zip(EVENTS, events.T):
        finite = column[np.isfinite(column)]
        q = np.quantile(finite, [.25, .5, .75]).tolist() if len(finite) else [None]*3
        result[key] = dict(n_total=len(column), n_events=len(finite), fraction=len(finite)/len(column),
                           p25=q[0], median=q[1], p75=q[2])
    return result


@njit
def pair_counts(weights, axes, indices, delta):
    counts = np.zeros(len(weights), dtype=np.int64)
    for k in range(len(weights)):
        u, v, w = weights[k]
        for i, j in indices:
            diff = u*(axes[i, 0]-axes[j, 0]) + v*(axes[i, 1]-axes[j, 1]) + w*(axes[i, 2]-axes[j, 2])
            if abs(diff) > delta:
                counts[k] += 1
    return counts


def weight_sweep(p, weights_file, out):
    with weights_file.open() as f:
        draws = list(csv.DictReader(f))
    weights = np.array([[float(row[key]) for key in ("w_A", "w_S", "w_Y10")] for row in draws])
    grid = np.array([clamped_cofilin_equilibrium(r, L, p)[0]
                     for r in np.linspace(.05, .9, 50) for L in np.linspace(0, 1.5, 40)])
    _, _, A, S, Y9, P, Y10, _ = grid.T
    eps = p.obs.eps
    maps = {
        "canonical_m_morph": P/(P+Y9+eps),
        "ramified_branch_area_proxy": P/(P+Y9+Y10+eps),
        "solidity_circularity_proxy": Y9/(P+Y9+Y10+eps),
    }
    i, j = np.triu_indices(len(grid), 1)
    rows, ranges = [], {}
    for name, morph in maps.items():
        mask = np.abs(morph[i] - morph[j]) < p.obs.eps_M
        idx = np.column_stack((i[mask], j[mask]))
        counts = pair_counts(weights, np.column_stack((A, S, Y10)), idx, p.obs.delta_I)
        base = int(pair_counts(np.array([[.3, .4, .3]]), np.column_stack((A, S, Y10)), idx, p.obs.delta_I)[0])
        ranges[name] = dict(minimum=int(counts.min()), maximum=int(counts.max()), canonical_weights=base)
        rows.extend(dict(draw_id=row["draw_id"], morphology_definition=name, w_A=w[0],
                         w_S=w[1], w_Y10=w[2], pair_count=int(n))
                    for row, w, n in zip(draws, weights, counts))
    write_csv(out, "readout_weight_sweep.csv", rows)
    return ranges


def related_equilibrium_outputs(p, out, multipliers):
    rows = []
    for r in (.08, .15, .2, .3, .5, .7, .8):
        for L in (0., .3, .5, .8, 1., 1.5):
            _, H, A, S, Y9, P, Y10, C = clamped_cofilin_equilibrium(r, L, p)[0]
            rows.append(dict(R=r, L=L, H=H, A=A, S=S, Y9=Y9, P=P, Y10=Y10, C_act=C,
                             M_morph=P/(P+Y9+p.obs.eps), I_infl=.3*A+.4*S+.3*Y10))
    candidates = [(abs(a["I_infl"]-b["I_infl"]), a, b) for i, a in enumerate(rows)
                  for b in rows[i+1:] if abs(a["M_morph"]-b["M_morph"]) < p.obs.eps_M]
    _, a, b = max(candidates, key=lambda pair: pair[0])
    write_csv(out, "model_condition_matrix.csv", rows)
    write_csv(out, "model_state_pairs.csv", [dict(M1=a["M_morph"], M2=b["M_morph"],
                                                 I1=a["I_infl"], I2=b["I_infl"], R1=a["R"],
                                                 R2=b["R"], L1=a["L"], L2=b["L"])])
    with multipliers.open() as stream:
        draws = list(csv.DictReader(stream))
    estimates = []
    for row in draws:
        perturbed = deepcopy(p)
        for path, factor in row.items():
            if path == "draw_id":
                continue
            group, param = path.split(".")
            target = getattr(perturbed, group)
            setattr(target, param, getattr(target, param) * float(factor))
        estimates.append(dict(draw_id=row["draw_id"], **equilibrium_grid(perturbed, None)))
    write_csv(out, "motor_perturbation_correlations.csv", estimates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"results/rho_echo_formal_closure_20261106")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--weights", type=Path, help="Frozen weight draws, one row per draw")
    parser.add_argument("--readout-only", action="store_true")
    parser.add_argument("--motor-multipliers", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    p = default_params()
    assert abs(p.l1.k_H - 0.1797757843858774) < 1e-12, "Missing frozen calcium calibration"
    if args.readout_only:
        if args.weights:
            print(weight_sweep(p, args.weights, args.out))
        if args.motor_multipliers:
            related_equilibrium_outputs(p, args.out, args.motor_multipliers)
        return
    eq = equilibrium_grid(p, args.out)
    th = p.thresh
    bounds = (th.R_low, th.R_high, th.R_src, p.obs.H_crit, p.obs.eps, p.obs.eps_M)
    ensemble, errors, paired, rows, paths = {}, {}, {}, [], []
    for omega in OMEGAS:
        process = process_parameters(p, omega)
        errors[str(omega)] = verify_generator(p, omega, process)
        n0 = np.rint(INITIAL*omega).astype(np.int64)
        times, trajectories, events = simulate(n0, omega, *process, bounds, args.n,
                                               min(args.n, 5) if omega == 500 else 0, 42 + omega)
        assert np.all((events[np.isfinite(events)] >= 0) & (events[np.isfinite(events)] <= 60))
        ensemble[str(omega)] = summarize(events)
        both = np.isfinite(events[:, 1:3]).all(axis=1)
        paired[str(omega)] = dict(n_both=int(both.sum()),
                                  src_first=int(np.sum(events[both, 1] < events[both, 2])),
                                  mismatch_first=int(np.sum(events[both, 2] < events[both, 1])))
        for index, event in enumerate(events, 1):
            rows.extend(dict(trajectory_id=index, event_key=key, event_label=label, omega=omega,
                             first_passage_time=float(t) if np.isfinite(t) else "",
                             observed=bool(np.isfinite(t)), source_mode="exact_ssa")
                        for key, label, t in zip(EVENTS, LABELS, event))
        for index, trajectory in enumerate(trajectories, 1):
            assert np.all(trajectory[0] == INITIAL)
            paths.extend(dict(trajectory_id=index, omega=omega, t=t, **dict(zip(STATES, x)))
                         for t, x in zip(times, trajectory))
        print(omega, ensemble[str(omega)], flush=True)
    write_csv(args.out, "step2_fpt_events_dense.csv", rows)
    write_csv(args.out, "step2_representative_trajectories.csv", paths)
    write_csv(args.out, "step2_deterministic_reference.csv", deterministic_reference(p, times))
    summary = dict(n_paths=args.n, horizon=60., L_step=5., seeds={str(w):42+w for w in OMEGAS},
                   initial_state=INITIAL.tolist(), thresholds=bounds, parameters=asdict(p),
                   generator_drift_max_abs_error=errors, equilibria=eq,
                   ensemble_stats=ensemble, within_path_src_mismatch=paired)
    if args.weights:
        summary["weight_sweep_pair_count_ranges"] = weight_sweep(p, args.weights, args.out)
    if args.motor_multipliers:
        related_equilibrium_outputs(p, args.out, args.motor_multipliers)
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
