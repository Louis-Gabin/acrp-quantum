"""Experiment 2: sensitivity to the penalty weights A = B.

The QUBO penalty weights must be large enough that the global optimum is a
feasible, conflict-free, one-hot assignment, yet small enough that the energy
landscape is not needlessly rugged for the sampler. This experiment sweeps a
penalty factor `alpha` (A = B = alpha) on a fixed small instance and records, at
fixed QAOA depth:
  - whether the exact QUBO optimum is feasible (validity of the encoding),
  - the QAOA success probability P(opt) and one-hot mass P(one-hot),
  - the approximation ratio.

The theoretical safe threshold is alpha > n (the objective range). The plot
shows the trade-off: too small -> optimum infeasible; too large -> sampler
quality degrades.

Run:  python3 analysis/sweep_penalty.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import one_hot_mass, results_dir
from instance import Instance
from manoeuvres import build_manoeuvre_set
from qubo import build_qubo
from evaluate import evaluate
from exact_bruteforce import brute_force
from qaoa_numpy import solve_qaoa_numpy, _energy_spectrum


def main(n: int = 3, n_heading: int = 3, p: int = 3, seed: int = 1) -> None:
    inst = Instance.circle_problem(n=n, radius=100.0, d=8.0)
    man = build_manoeuvre_set(n_heading=n_heading)
    alphas = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0]

    rows = []
    for alpha in alphas:
        qubo = build_qubo(inst, man, A=alpha, B=alpha)
        bx, be = brute_force(qubo.Q, qubo.offset)
        opt_sol = evaluate(qubo, bx)
        energies = _energy_spectrum(qubo.Q, qubo.offset)
        c_opt = float(np.min(energies))
        r = solve_qaoa_numpy(qubo, p=p, restarts=100, refine_steps=40, seed=seed)
        rows.append({
            "alpha": alpha,
            "opt_feasible": opt_sol.feasible,
            "opt_manoeuvres": opt_sol.num_manoeuvres,
            "P_opt": round(r.success_probability, 5),
            "P_one_hot": round(one_hot_mass(r.probs, inst.n, qubo.M), 5),
            "approx_ratio": round(r.expected_energy / c_opt, 4) if c_opt > 0 else None,
        })
        print(rows[-1])

    df = pd.DataFrame(rows)
    out_csv = f"{results_dir()}/sweep_penalty_{inst.name}.csv"
    df.to_csv(out_csv, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["alpha"], df["P_opt"], "o-", color="tab:blue", label="P(opt)")
    ax.plot(df["alpha"], df["P_one_hot"], "s--", color="tab:green",
            label="P(one-hot)")
    infeasible = df[~df["opt_feasible"]]
    if len(infeasible):
        ax.scatter(infeasible["alpha"], [0] * len(infeasible), marker="x",
                   color="red", s=80, label="optimum infeasible", zorder=5)
    ax.axvline(n, color="grey", linestyle=":", label=f"safe threshold alpha>n={n}")
    ax.set_xlabel("penalty weight alpha (A = B)")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    ax.set_title(f"Penalty-weight sensitivity on {inst.name} (p={p})")
    fig.tight_layout()
    out_png = f"{results_dir()}/sweep_penalty_{inst.name}.png"
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_csv} and {out_png}")


if __name__ == "__main__":
    main()
