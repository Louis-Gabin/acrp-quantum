#!/usr/bin/env python3
"""Large-instance classical scaling campaign (n up to 50), default M = 5.

Motivation
----------
David Rey's feedback asked for instances larger than 5 aircraft, on the grounds
that the quantum approach becomes interesting as the instance size grows. There
is, however, a hard ceiling to be honest about: exact QAOA simulation needs
n * M qubits, and 2^(n*M) complex amplitudes of memory. With M = 5 that is 50
qubits at n = 10 and 250 qubits at n = 50, which is impossible to *simulate* on
any machine. Those large instances therefore belong to the CLASSICAL scaling
study, whose whole point is to show how the exact solve cost grows with size.

Two complementary regimes
-------------------------
1. Default ("feasibility wall") run: instances reported as-is. Under the
   simplified control bounds (heading in [-30, +30] deg, speed in [0.94, 1.03])
   some dense/large instances are provably INFEASIBLE: certain crossing pairs
   cannot be separated by ANY discrete manoeuvre. Gurobi reports 'infeasible'
   and that is a genuine result about the model, not a bug.
2. --feasible-only run: each random instance is regenerated (new seed) until the
   exact solver confirms feasibility, giving clean "optimum vs n" curves. The
   random generator is naturally sparse (density ~ 0.04), so feasible instances
   are found in a handful of tries; the feasibility FILTER does the work, not a
   narrow density band. Do NOT set an unreachable density band here: it makes
   the rejection sampler burn all its tries. Keep the band wide.

It runs, for each requested n:
  - Gurobi exact ILP on the discretised counting model -> optimum + solve time,
  - simulated annealing on the same QUBO -> heuristic reference + time.
  (QAOA is intentionally NOT run here: it cannot be simulated at these sizes.
   Use analysis/benchmark.py for the small QAOA-simulable ladder.)

Examples
--------
    # Feasibility-wall run (documents infeasibility):
    python3 analysis/scaling_large.py --sizes 10 12 25 50 --nh 3 --threads 4

    # Clean feasible run (wide band + feasibility filter, fast):
    python3 analysis/scaling_large.py --sizes 10 12 25 50 --nh 3 --threads 4 \
            --feasible-only --out results/scaling_feasible.csv

    # Offline check of the SA + generation path (no Gurobi):
    python3 analysis/scaling_large.py --sizes 4 6 --no-gurobi
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from _common import results_dir  # also puts src/ and src/solvers/ on sys.path

from instance import Instance
from manoeuvres import build_manoeuvre_set
from qubo import build_qubo
from evaluate import evaluate
from simulated_annealing import simulated_annealing
from benchmark import random_instance  # reuse the density-controlled generator


def _sa_budget(v: int) -> tuple[int, int]:
    """Adaptive SA budget (reads, sweeps) that stays reasonable on a laptop."""
    if v <= 30:
        return 40, 1500
    if v <= 60:
        return 20, 800
    if v <= 130:
        return 10, 500
    return 6, 300


def _no_locked_pair(instance, manoeuvres) -> bool:
    """Necessary (not sufficient) feasibility check, usable without Gurobi.

    Returns False as soon as a conflicting pair has ALL manoeuvre combinations
    in conflict (a 'locked' pair), which proves infeasibility. Returning True
    only means 'not provably infeasible this way'.
    """
    M = len(manoeuvres)
    vel = [[instance.velocity(i, m.q, m.dtheta) for m in manoeuvres]
           for i in range(instance.n)]
    for (i, j) in instance.pairs:
        combos_conflict = 0
        for a in range(M):
            for b in range(M):
                if instance.in_conflict(i, vel[i][a], j, vel[j][b]):
                    combos_conflict += 1
        if combos_conflict == M * M:
            return False
    return True


def _feasible(instance, manoeuvres, solve_gurobi_exact, time_limit):
    """True if the instance is solvable. Uses Gurobi when available, else the
    locked-pair necessary condition as a best-effort proxy."""
    if solve_gurobi_exact is not None:
        try:
            g = solve_gurobi_exact(instance, manoeuvres, time_limit=time_limit)
            return bool(g.feasible)
        except Exception:
            return _no_locked_pair(instance, manoeuvres)
    return _no_locked_pair(instance, manoeuvres)


def build_instances(sizes, random_per_size, d, base_seed, density, max_tries):
    """CP_n plus a few density-controlled random instances at each size."""
    dmin, dmax = density
    insts = []
    for n in sizes:
        insts.append(Instance.circle_problem(n=n, radius=100.0, d=d))
        for k in range(random_per_size):
            insts.append(random_instance(
                n, d, seed=base_seed + 1000 * n + k, max_tries=max_tries,
                density_min=dmin, density_max=dmax, name=f"RND{n}_{k + 1}"))
    return insts


def build_instances_feasible(sizes, random_per_size, d, base_seed, density,
                             max_tries, manoeuvres, solve_gurobi_exact,
                             attempts, feasible_time_limit):
    """Like build_instances but each random instance is regenerated (new seed)
    until the solver confirms it is feasible. CP_n is kept as-is (deterministic);
    if CP_n is infeasible it will simply be dropped from the feasible curves by
    the plot, while still appearing in the CSV.

    The random generator is naturally sparse, so keep the density band WIDE
    (e.g. [0.02, 0.9]); the feasibility filter is what selects solvable ones.
    """
    dmin, dmax = density
    insts = []
    for n in sizes:
        insts.append(Instance.circle_problem(n=n, radius=100.0, d=d))
        for k in range(random_per_size):
            chosen = None
            for a in range(attempts):
                seed = base_seed + 1000 * n + k * 100000 + a
                cand = random_instance(n, d, seed=seed, max_tries=max_tries,
                                       density_min=dmin, density_max=dmax,
                                       name=f"RNDF{n}_{k + 1}")
                chosen = cand
                if _feasible(cand, manoeuvres, solve_gurobi_exact,
                             feasible_time_limit):
                    break
            else:
                print(f"[warn] {n=} slot {k + 1}: no feasible instance in "
                      f"{attempts} tries; keeping last candidate.")
            insts.append(chosen)
    return insts


def run(args):
    sizes = [int(s) for s in args.sizes]
    seeds = list(range(args.sa_seeds))
    density = (args.density_min, args.density_max)
    man = build_manoeuvre_set(n_heading=args.nh, n_speed=args.ns)

    solve_gurobi_exact = None
    if args.gurobi:
        try:
            from exact_gurobi import solve_gurobi_exact as _sge
            solve_gurobi_exact = _sge
        except Exception as exc:
            print(f"[warn] could not import the Gurobi solver ({exc}); "
                  "running SA only.")

    if args.feasible_only:
        print(f"[info] feasible-only mode: wide density band "
              f"[{args.density_min}, {args.density_max}], up to "
              f"{args.feasible_attempts} feasibility tries each, "
              f"sample max_tries={args.sample_tries}.")
        instances = build_instances_feasible(
            sizes, args.random_per_size, args.d, args.base_seed, density,
            args.sample_tries, man, solve_gurobi_exact, args.feasible_attempts,
            args.feasible_time_limit)
    else:
        instances = build_instances(sizes, args.random_per_size, args.d,
                                    args.base_seed, density, args.sample_tries)

    rows = []
    for inst in instances:
        qubo = build_qubo(inst, man)
        v = qubo.num_vars
        M = qubo.M

        # --- Gurobi exact optimum (ground truth) -------------------------- #
        g_man = float("nan")
        g_status = "skipped"
        g_gap = float("nan")
        g_time = float("nan")
        g_constraints = float("nan")
        if solve_gurobi_exact is not None:
            try:
                g = solve_gurobi_exact(
                    inst, man, time_limit=args.gurobi_time_limit,
                    threads=args.threads, mip_gap=args.mip_gap)
                g_man = g.num_manoeuvres
                g_status = g.status
                g_gap = round(g.mip_gap, 5)
                g_time = round(g.seconds, 3)
                g_constraints = g.num_conflict_constraints
            except Exception as exc:
                g_status = f"error: {exc}"

        # --- Simulated annealing (best over seeds) ------------------------ #
        reads, sweeps = _sa_budget(v)
        best_man = None
        best_feasible = False
        best_time = float("nan")
        for seed in seeds:
            t = time.time()
            sa = simulated_annealing(qubo.Q, qubo.offset, num_reads=reads,
                                     num_sweeps=sweeps, seed=seed)
            dt = time.time() - t
            sol = evaluate(qubo, sa.best_x)
            key = (0 if sol.feasible else 1, sol.num_manoeuvres)
            if best_man is None or key < (0 if best_feasible else 1, best_man):
                best_man = sol.num_manoeuvres
                best_feasible = sol.feasible
                best_time = round(dt, 3)

        sa_matches = (best_feasible and not np.isnan(g_man)
                      and isinstance(g_man, (int, float))
                      and best_man == g_man)

        row = {
            "instance": inst.name,
            "n": inst.n,
            "M": M,
            "qubits": v,
            "conflict_constraints": g_constraints,
            "gurobi_manoeuvres": g_man,
            "gurobi_status": g_status,
            "gurobi_gap": g_gap,
            "gurobi_time_s": g_time,
            "sa_reads": reads,
            "sa_sweeps": sweeps,
            "sa_manoeuvres": best_man,
            "sa_feasible": best_feasible,
            "sa_time_s": best_time,
            "sa_matches_opt": bool(sa_matches),
        }
        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)
    out_csv = args.out or f"{results_dir()}/scaling_large.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}  ({len(df)} rows)")

    # Feasibility summary (the 'wall' result), grouped by n.
    if "gurobi_status" in df:
        summ = (df.assign(feasible=df["gurobi_status"].eq("optimal"))
                  .groupby("n")["feasible"].mean().round(3))
        print("\nFeasibility rate by n (share solved to optimality):")
        print(summ.to_string())

    _plot(df, args, out_csv)


def _plot(df: pd.DataFrame, args, out_csv: str) -> None:
    feas = df[df["gurobi_status"] == "optimal"].sort_values("n")
    if feas.empty or feas["gurobi_time_s"].isna().all():
        print("[info] no feasible Gurobi rows to plot; skipping figure.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Average over instances at each n for a clean curve.
    by_n = feas.groupby("n").agg(
        man=("gurobi_manoeuvres", "mean"),
        secs=("gurobi_time_s", "mean")).reset_index()

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(by_n["n"], by_n["man"], "o-", color="tab:blue",
             label="Gurobi optimum (manoeuvres)")
    ax1.set_xlabel("number of aircraft n")
    ax1.set_ylabel("optimal number of manoeuvring aircraft")
    ax2 = ax1.twinx()
    ax2.plot(by_n["n"], by_n["secs"], "^-", color="tab:red",
             label="Gurobi solve time (s)")
    ax2.set_yscale("log")
    ax2.set_ylabel("Gurobi solve time (s, log scale)")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")
    tag = "feasible set" if args.feasible_only else "dense set"
    ax1.set_title(f"Classical scaling ({tag}, M={int(feas['M'].iloc[0])})")
    fig.tight_layout()
    out_png = out_csv.rsplit(".", 1)[0] + ".png"
    fig.savefig(out_png, dpi=120)
    print(f"Saved {out_png}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Large-instance classical scaling campaign for the ACRP.")
    ap.add_argument("--sizes", nargs="+", default=["10", "12", "25", "50"],
                    help="numbers of aircraft n (default: 10 12 25 50).")
    ap.add_argument("--nh", type=int, default=5,
                    help="heading options => M (default 5; use 3 for reduced).")
    ap.add_argument("--ns", type=int, default=1,
                    help="speed options (default 1 = heading-only control).")
    ap.add_argument("--random-per-size", type=int, default=2,
                    dest="random_per_size",
                    help="random instances per size, in addition to CP_n.")
    ap.add_argument("--base-seed", type=int, default=2000, dest="base_seed")
    ap.add_argument("--d", type=float, default=8.0, help="separation distance.")
    ap.add_argument("--sa-seeds", type=int, default=5, dest="sa_seeds",
                    help="number of SA seeds (best solution is kept).")
    ap.add_argument("--density-min", type=float, default=0.02, dest="density_min",
                    help="min nominal-conflict density (keep low/reachable).")
    ap.add_argument("--density-max", type=float, default=0.9, dest="density_max",
                    help="max nominal-conflict density (keep band WIDE).")
    ap.add_argument("--sample-tries", type=int, default=400, dest="sample_tries",
                    help="rejection-sampling cap per instance (keep modest so "
                         "an unreachable band cannot stall the run).")
    ap.add_argument("--feasible-only", action="store_true", dest="feasible_only",
                    help="regenerate random instances until solver-feasible.")
    ap.add_argument("--feasible-attempts", type=int, default=40,
                    dest="feasible_attempts",
                    help="max regeneration attempts per feasible instance.")
    ap.add_argument("--feasible-time-limit", type=float, default=30.0,
                    dest="feasible_time_limit",
                    help="Gurobi time cap (s) for the feasibility filter.")
    ap.add_argument("--gurobi", action="store_true", default=True,
                    help="run the Gurobi exact ILP (default on).")
    ap.add_argument("--no-gurobi", action="store_false", dest="gurobi",
                    help="skip Gurobi (SA only; useful offline).")
    ap.add_argument("--gurobi-time-limit", type=float, default=600.0,
                    dest="gurobi_time_limit",
                    help="per-instance Gurobi time cap in seconds (default 600).")
    ap.add_argument("--threads", type=int, default=None,
                    help="cap Gurobi threads (e.g. 4 on an 8 GB Mac).")
    ap.add_argument("--mip-gap", type=float, default=0.0, dest="mip_gap",
                    help="target relative MIP gap (0 = prove optimality).")
    ap.add_argument("--out", default=None,
                    help="output CSV path (default results/scaling_large.csv).")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
