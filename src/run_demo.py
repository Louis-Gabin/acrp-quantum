"""End-to-end demonstration and self-validation of the ACRP QUBO pipeline.

Runs, on Circle Problem instances:
  1. the exact brute-force QUBO optimum (ground truth while Gurobi is pending),
  2. the simulated annealing sampler on the same QUBO,
and checks that both recover a feasible, conflict-free assignment with the
minimum number of manoeuvres. Also emits a trajectory figure for CP_5.

Run:  python3 src/run_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "solvers"))

import numpy as np

from instance import Instance
from manoeuvres import build_manoeuvre_set
from qubo import build_qubo
from evaluate import evaluate
from exact_bruteforce import brute_force
from simulated_annealing import simulated_annealing


def run_instance(n: int, n_heading: int = 5):
    inst = Instance.circle_problem(n=n, radius=100.0, d=8.0, speed=1.0)
    manoeuvres = build_manoeuvre_set(n_heading=n_heading, n_speed=1)
    qubo = build_qubo(inst, manoeuvres)

    print(f"\n=== {inst.name} | aircraft={n} | manoeuvres={len(manoeuvres)} "
          f"| QUBO vars={qubo.num_vars} ===")
    print(f"Nominal conflicts (do-nothing): {len(inst.nominal_conflicts())} "
          f"pairs -> {inst.nominal_conflicts()}")
    print(f"Penalties: A={qubo.A}, B={qubo.B}, C={qubo.C}")

    results = {}

    # Exact ground truth (only for tractable sizes).
    if qubo.num_vars <= 22:
        bx, be = brute_force(qubo.Q, qubo.offset)
        bsol = evaluate(qubo, bx)
        results["exact"] = (be, bsol)
        print(f"[exact]  energy={be:.3f}  feasible={bsol.feasible}  "
              f"manoeuvres={bsol.num_manoeuvres}  assignment={bsol.assignment}")

    # Simulated annealing on the same QUBO (budget scales with problem size).
    num_reads = 40 if qubo.num_vars <= 20 else 120
    num_sweeps = 400 if qubo.num_vars <= 20 else 1500
    sa = simulated_annealing(qubo.Q, qubo.offset, num_reads=num_reads,
                             num_sweeps=num_sweeps, seed=42)
    ssol = evaluate(qubo, sa.best_x)
    results["sa"] = (sa.best_energy, ssol)
    print(f"[SA]     energy={sa.best_energy:.3f}  feasible={ssol.feasible}  "
          f"manoeuvres={ssol.num_manoeuvres}  assignment={ssol.assignment}")

    # Cross-check: SA matches the exact QUBO optimum.
    if "exact" in results:
        gap = sa.best_energy - results["exact"][0]
        print(f"[check]  SA-exact energy gap = {gap:.3f} "
              f"({'MATCH' if abs(gap) < 1e-6 else 'SUBOPTIMAL'})")

    return inst, qubo, results


def plot_solution(inst: Instance, qubo, sol, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    man = qubo.manoeuvres
    fig, ax = plt.subplots(figsize=(6, 6))
    L = inst.x0.max() * 1.4 if inst.x0.max() > 0 else 120
    for i in range(inst.n):
        # Nominal heading (dashed) vs chosen manoeuvre (solid).
        ax.plot(inst.x0[i], inst.y0[i], "o", color="black")
        ax.annotate(str(i), (inst.x0[i], inst.y0[i]), textcoords="offset points",
                    xytext=(6, 6))
        vx0, vy0 = inst.velocity(i, 1.0, 0.0)
        ax.arrow(inst.x0[i], inst.y0[i], vx0 * 30, vy0 * 30, color="grey",
                 alpha=0.4, head_width=3, length_includes_head=True)
        m = sol.assignment[i]
        if m >= 0:
            vx, vy = inst.velocity(i, man[m].q, man[m].dtheta)
            col = "tab:red" if not man[m].is_nominal else "tab:green"
            ax.arrow(inst.x0[i], inst.y0[i], vx * 30, vy * 30, color=col,
                     head_width=3, length_includes_head=True)
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal")
    ax.set_title(f"{inst.name}: green=nominal, red=manoeuvre "
                 f"({sol.num_manoeuvres} manoeuvres, feasible={sol.feasible})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"[plot]   saved {path}")


if __name__ == "__main__":
    for n in (3, 4):
        run_instance(n)
    inst, qubo, results = run_instance(5)
    sol = results.get("exact", results["sa"])[1]
    os.makedirs("results", exist_ok=True)
    plot_solution(inst, qubo, sol, "results/CP_5_solution.png")
