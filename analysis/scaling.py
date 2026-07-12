"""Experiment 3: scaling of solver quality and cost with instance size.

Across the Circle Problem ladder CP_3..CP_6 with a reduced manoeuvre set (so the
qubit count stays in the exact-simulation range 9..18), record for each size:
  - exact QUBO optimum (ground truth),
  - simulated annealing: does it match the exact optimum? wall-clock time,
  - QAOA (numpy, fixed depth): P(opt), one-hot mass, wall-clock time.

This exposes the central scaling story: the classical twin stays exact and cheap,
while the QAOA success probability decays as the qubit count grows, at rising
cost. This is the empirical face of the 'scale gap' from the literature review.

Run:  python3 analysis/scaling.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from _common import one_hot_mass, results_dir
from instance import Instance
from manoeuvres import build_manoeuvre_set
from qubo import build_qubo
from evaluate import evaluate
from exact_bruteforce import brute_force
from simulated_annealing import simulated_annealing
from qaoa_numpy import solve_qaoa_numpy, _energy_spectrum


def main(sizes=(3, 4, 5, 6), n_heading: int = 3, p: int = 2, seed: int = 1) -> None:
    rows = []
    for n in sizes:
        inst = Instance.circle_problem(n=n, radius=100.0, d=8.0)
        man = build_manoeuvre_set(n_heading=n_heading)
        qubo = build_qubo(inst, man)
        V = qubo.num_vars
        c_opt = float(np.min(_energy_spectrum(qubo.Q, qubo.offset)))

        t = time.time()
        bx, be = brute_force(qubo.Q, qubo.offset)
        exact_time = time.time() - t

        t = time.time()
        sa = simulated_annealing(qubo.Q, qubo.offset, num_reads=100,
                                 num_sweeps=2000, seed=42)
        sa_time = time.time() - t
        sa_match = abs(sa.best_energy - be) < 1e-6

        restarts = 100 if V <= 12 else (50 if V <= 15 else 30)
        t = time.time()
        r = solve_qaoa_numpy(qubo, p=p, restarts=restarts, refine_steps=30, seed=seed)
        qaoa_time = time.time() - t

        rows.append({
            "instance": inst.name,
            "aircraft": n,
            "qubits": V,
            "C_opt": round(c_opt, 2),
            "SA_matches_exact": sa_match,
            "SA_time_s": round(sa_time, 2),
            "QAOA_P_opt": round(r.success_probability, 5),
            "QAOA_P_one_hot": round(one_hot_mass(r.probs, inst.n, qubo.M), 5),
            "QAOA_time_s": round(qaoa_time, 2),
            "exact_time_s": round(exact_time, 2),
        })
        print(rows[-1])

    df = pd.DataFrame(rows)
    out_csv = f"{results_dir()}/scaling.csv"
    df.to_csv(out_csv, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(df["qubits"], df["QAOA_P_opt"], "o-", color="tab:blue",
             label="QAOA P(opt)")
    ax1.plot(df["qubits"], df["QAOA_P_one_hot"], "s--", color="tab:green",
             label="QAOA P(one-hot)")
    ax1.set_xlabel("QUBO variables (qubits)")
    ax1.set_ylabel("probability")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(df["qubits"], df["QAOA_time_s"], "^-", color="tab:red",
             label="QAOA time (s)")
    ax2.set_ylabel("QAOA wall-clock time (s)")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center left")
    ax1.set_title(f"Scaling on the Circle Problem (p={p}, reduced manoeuvre set)")
    fig.tight_layout()
    out_png = f"{results_dir()}/scaling.png"
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_csv} and {out_png}")


if __name__ == "__main__":
    main()
