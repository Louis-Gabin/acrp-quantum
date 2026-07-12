#!/usr/bin/env python3
"""Multi-seed, multi-instance solver benchmark for the ACRP QUBO.

Runs every QUBO-based solver on a shared set of instances and seeds, and writes
one tidy row per (instance, solver, depth, seed) to results/benchmark.csv. This
is the data source for Chapter 3 (solver comparison, scaling) and the input to
analysis/stats_tests.py (Wilcoxon / Friedman / Nemenyi).

Solvers covered here:
  - exact brute force   (ground-truth QUBO optimum, deterministic, V <= 22)
  - simulated annealing (classical twin, stochastic over seeds)
  - QAOA numpy          (exact statevector, stochastic over restart seeds)
  - QAOA qiskit         (optional: noiseless and/or depolarising noise)

The Gurobi exact baseline is NOT run here: it uses a different continuous MINLP
model and lives in baseline_gurobi/. Its optimum is cross-checked against the
brute-force QUBO optimum in Section 2.3; on the tested rungs the two agree, so
the brute-force optimum is used as ground truth in this harness.

Run (offline, no qiskit):
    python3 analysis/benchmark.py --no-qiskit

Full run on your Mac (requirements.txt installed):
    python3 analysis/benchmark.py --qiskit --noisy
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pandas as pd

from _common import results_dir  # also sets sys.path to src/ and src/solvers

from instance import Instance
from manoeuvres import build_manoeuvre_set
from qubo import build_qubo
from evaluate import evaluate
from exact_bruteforce import brute_force
from simulated_annealing import simulated_annealing
from qaoa_numpy import solve_qaoa_numpy


def random_instance(n, d, seed, radius=100.0, speed=1.0,
                    density_min=0.4, density_max=0.9, max_tries=20000,
                    name=None):
    """Random instance with a controlled nominal-conflict density.

    Aircraft are placed uniformly in a box and given random headings; the
    instance is accepted when the share of pairs in nominal conflict falls in
    [density_min, density_max]. This complements the maximally dense Circle
    Problem with irregular geometries at a comparable difficulty.
    """
    rng = np.random.default_rng(seed)
    total_pairs = n * (n - 1) // 2
    last = None
    for _ in range(max_tries):
        x0 = rng.uniform(-radius, radius, n)
        y0 = rng.uniform(-radius, radius, n)
        theta0 = rng.uniform(-math.pi, math.pi, n)
        v0 = np.full(n, float(speed))
        inst = Instance(n=n, d=d, x0=x0, y0=y0, theta0=theta0, v0=v0,
                        name=name or f"RND{n}_{seed}")
        last = inst
        dens = len(inst.nominal_conflicts()) / total_pairs if total_pairs else 0.0
        if density_min <= dens <= density_max:
            return inst
    return last


def build_instances(sizes, random_per_size, d, base_seed):
    insts = []
    for n in sizes:
        insts.append(Instance.circle_problem(n=n, radius=100.0, d=d))
        for k in range(random_per_size):
            insts.append(random_instance(
                n, d, seed=base_seed + 1000 * n + k, name=f"RND{n}_{k + 1}"))
    return insts


def _decode(v, state_index):
    return np.array([(state_index >> k) & 1 for k in range(v)], dtype=float)


def feasibility_mass_from_probs(qubo, probs, thresh=1e-6):
    """Probability mass on feasible (one-hot AND conflict-free) states."""
    v = qubo.num_vars
    mass = 0.0
    for s in np.where(probs > thresh)[0]:
        if evaluate(qubo, _decode(v, int(s))).feasible:
            mass += float(probs[s])
    return mass


def metrics_from_counts(qubo, counts, opt_energy):
    """Return (best_x, best_energy, expected_energy, p_opt, feas_rate) from a
    counts/probabilities dictionary (works for noiseless probs and noisy shots).
    """
    total = float(sum(counts.values()))
    exp_e = 0.0
    p_opt = 0.0
    feas = 0.0
    best_e = float("inf")
    best_x = None
    for bitstr, c in counts.items():
        x = np.array([int(b) for b in reversed(bitstr)], dtype=float)
        w = c / total
        e = qubo.energy(x)
        exp_e += w * e
        if math.isclose(e, opt_energy, rel_tol=1e-6, abs_tol=1e-6):
            p_opt += w
        if evaluate(qubo, x).feasible:
            feas += w
        if e < best_e:
            best_e = e
            best_x = x
    return best_x, best_e, exp_e, p_opt, feas


def row(inst, qubo, solver, depth, seed, *, feasible, manoeuvres, energy,
        opt_energy, opt_man, p_opt, feas_rate, exp_energy, seconds, notes=""):
    gap = energy - opt_energy
    approx = exp_energy / opt_energy if abs(opt_energy) > 1e-9 else float("nan")
    hit = bool(feasible and math.isclose(energy, opt_energy, rel_tol=1e-6,
                                         abs_tol=1e-6))
    man_gap = (manoeuvres - opt_man) if (feasible and opt_man is not None) \
        else float("nan")
    return {
        "instance": inst.name, "n": inst.n, "qubits": qubo.num_vars,
        "solver": solver, "depth": depth, "seed": seed,
        "feasible": bool(feasible), "hit_opt": hit,
        "manoeuvres": manoeuvres, "manoeuvre_gap": man_gap,
        "energy": round(float(energy), 4), "opt_energy": round(float(opt_energy), 4),
        "energy_gap": round(float(gap), 4),
        "expected_energy": (round(float(exp_energy), 4)
                            if not math.isnan(exp_energy) else float("nan")),
        "approx_ratio": (round(float(approx), 4)
                         if not math.isnan(approx) else float("nan")),
        "p_opt": round(float(p_opt), 6) if not math.isnan(p_opt) else float("nan"),
        "feas_rate": (round(float(feas_rate), 6)
                      if not math.isnan(feas_rate) else float("nan")),
        "seconds": round(float(seconds), 3), "notes": notes,
    }


def run(args):
    sizes = [int(s) for s in args.sizes]
    seeds = list(range(args.seeds))
    depths = [int(p) for p in args.depths]
    instances = build_instances(sizes, args.random_per_size, args.d, args.base_seed)

    use_qiskit = args.qiskit
    solve_qaoa_qiskit = None
    if use_qiskit:
        try:
            import qiskit  # noqa: F401
            import scipy  # noqa: F401
            from qaoa_qiskit import solve_qaoa_qiskit as _sqq
            solve_qaoa_qiskit = _sqq
        except Exception as exc:
            print(f"[warn] qiskit/scipy unavailable ({exc}); skipping qiskit runs.")
            use_qiskit = False

    rows = []
    for inst in instances:
        man = build_manoeuvre_set(n_heading=args.nh, n_speed=1)
        qubo = build_qubo(inst, man)
        v = qubo.num_vars

        opt_energy = float("nan")
        opt_man = None
        if v <= args.brute_max_vars:
            t = time.time()
            bx, be = brute_force(qubo.Q, qubo.offset)
            bsol = evaluate(qubo, bx)
            opt_energy = be
            opt_man = bsol.num_manoeuvres
            rows.append(row(inst, qubo, "brute_force", float("nan"), float("nan"),
                            feasible=bsol.feasible, manoeuvres=bsol.num_manoeuvres,
                            energy=be, opt_energy=be, opt_man=bsol.num_manoeuvres,
                            p_opt=float("nan"), feas_rate=float("nan"),
                            exp_energy=float("nan"), seconds=time.time() - t,
                            notes="ground truth"))
        else:
            print(f"[warn] {inst.name}: V={v} exceeds brute-force limit; "
                  "gaps will be NaN.")

        gt = opt_energy if not math.isnan(opt_energy) else None

        for seed in seeds:
            t = time.time()
            sa = simulated_annealing(qubo.Q, qubo.offset, num_reads=args.sa_reads,
                                     num_sweeps=args.sa_sweeps, seed=seed)
            ssol = evaluate(qubo, sa.best_x)
            rows.append(row(inst, qubo, "simulated_annealing", float("nan"), seed,
                            feasible=ssol.feasible, manoeuvres=ssol.num_manoeuvres,
                            energy=sa.best_energy,
                            opt_energy=gt if gt is not None else sa.best_energy,
                            opt_man=opt_man, p_opt=float("nan"),
                            feas_rate=float("nan"), exp_energy=float("nan"),
                            seconds=time.time() - t,
                            notes=f"reads={args.sa_reads},sweeps={args.sa_sweeps}"))

            if v <= args.qaoa_numpy_max_vars:
                for p in depths:
                    t = time.time()
                    r = solve_qaoa_numpy(qubo, p=p, restarts=args.restarts, seed=seed)
                    qsol = evaluate(qubo, r.best_x)
                    feas = feasibility_mass_from_probs(qubo, r.probs)
                    rows.append(row(inst, qubo, "qaoa_numpy", p, seed,
                                    feasible=qsol.feasible,
                                    manoeuvres=qsol.num_manoeuvres,
                                    energy=r.best_energy,
                                    opt_energy=gt if gt is not None else r.best_energy,
                                    opt_man=opt_man, p_opt=r.success_probability,
                                    feas_rate=feas, exp_energy=r.expected_energy,
                                    seconds=time.time() - t,
                                    notes=f"restarts={args.restarts}"))

            if use_qiskit and v <= args.qaoa_qiskit_max_vars:
                configs = [("qaoa_qiskit", False)]
                if args.noisy:
                    configs.append(("qaoa_qiskit_noisy", True))
                for name, noisy in configs:
                    for p in depths:
                        t = time.time()
                        r = solve_qaoa_qiskit(qubo, p=p, seed=seed, noisy=noisy,
                                              shots=args.shots,
                                              maxiter=args.qiskit_maxiter)
                        ref = gt if gt is not None else r.best_energy
                        bx, be, exp_e, p_opt, feas = metrics_from_counts(
                            qubo, r.counts, ref)
                        bsol = evaluate(qubo, bx)
                        rows.append(row(inst, qubo, name, p, seed,
                                        feasible=bsol.feasible,
                                        manoeuvres=bsol.num_manoeuvres,
                                        energy=be,
                                        opt_energy=ref, opt_man=opt_man,
                                        p_opt=p_opt, feas_rate=feas,
                                        exp_energy=exp_e, seconds=time.time() - t,
                                        notes=("noisy" if noisy else "noiseless")
                                        + f",shots={args.shots}"))
        print(f"[done] {inst.name} (V={v})")

    df = pd.DataFrame(rows)
    out = args.out or (results_dir() + "/benchmark.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved {out}  ({len(df)} rows)")


def parse_args():
    ap = argparse.ArgumentParser(description="ACRP multi-seed solver benchmark.")
    ap.add_argument("--sizes", nargs="+", default=["3", "4", "5"])
    ap.add_argument("--random-per-size", type=int, default=2, dest="random_per_size")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=1000, dest="base_seed")
    ap.add_argument("--nh", type=int, default=3, help="heading options (3 = reduced set).")
    ap.add_argument("--depths", nargs="+", default=["1", "2", "3", "4"])
    ap.add_argument("--restarts", type=int, default=24)
    ap.add_argument("--d", type=float, default=8.0)
    ap.add_argument("--sa-reads", type=int, default=40, dest="sa_reads")
    ap.add_argument("--sa-sweeps", type=int, default=400, dest="sa_sweeps")
    ap.add_argument("--brute-max-vars", type=int, default=22, dest="brute_max_vars")
    ap.add_argument("--qaoa-numpy-max-vars", type=int, default=16,
                    dest="qaoa_numpy_max_vars")
    ap.add_argument("--qaoa-qiskit-max-vars", type=int, default=18,
                    dest="qaoa_qiskit_max_vars")
    ap.add_argument("--qiskit", action="store_true", help="run the Qiskit QAOA solver.")
    ap.add_argument("--no-qiskit", action="store_false", dest="qiskit")
    ap.add_argument("--noisy", action="store_true", help="also run the noisy Qiskit QAOA.")
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--qiskit-maxiter", type=int, default=200, dest="qiskit_maxiter")
    ap.add_argument("--out", default=None)
    ap.set_defaults(qiskit=False)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
