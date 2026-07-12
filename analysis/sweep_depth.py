"""Experiment 1: QAOA quality as a function of circuit depth p.

On a fixed small instance (CP_3, reduced manoeuvre set, 9 qubits), run QAOA for
p = 1..P_MAX with the exact numpy statevector simulator and record, for each p:
  - the expected energy <C>,
  - the approximation ratio r = <C> / C_opt (>= 1, closer to 1 is better),
  - the success probability P(opt) = probability of sampling the optimum,
  - the one-hot mass P(one-hot) = probability of a constraint-satisfying state.

This directly tests the theoretical expectation that QAOA improves with depth,
and quantifies how far shallow (NISQ-realistic) QAOA is from the optimum.

Run:  python3 analysis/sweep_depth.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import one_hot_mass, results_dir
from instance import Instance
from manoeuvres import build_manoeuvre_set
from qubo import build_qubo
from qaoa_numpy import solve_qaoa_numpy, _energy_spectrum, expectation


def main(n: int = 3, n_heading: int = 3, p_max: int = 6, seed: int = 1) -> None:
    inst = Instance.circle_problem(n=n, radius=100.0, d=8.0)
    man = build_manoeuvre_set(n_heading=n_heading)
    qubo = build_qubo(inst, man)
    energies = _energy_spectrum(qubo.Q, qubo.offset)
    c_opt = float(np.min(energies))

    rows = []
    for p in range(1, p_max + 1):
        r = solve_qaoa_numpy(qubo, p=p, restarts=120, refine_steps=50, seed=seed)
        onehot = one_hot_mass(r.probs, inst.n, qubo.M)
        rows.append({
            "p": p,
            "expected_energy": round(r.expected_energy, 4),
            "approx_ratio": round(r.expected_energy / c_opt, 4),
            "P_opt": round(r.success_probability, 5),
            "P_one_hot": round(onehot, 5),
        })
        print(rows[-1])

    df = pd.DataFrame(rows)
    out_csv = f"{results_dir()}/sweep_depth_{inst.name}.csv"
    df.to_csv(out_csv, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(df["p"], df["P_opt"], "o-", color="tab:blue", label="P(opt)")
    ax1.plot(df["p"], df["P_one_hot"], "s--", color="tab:green", label="P(one-hot)")
    ax1.set_xlabel("QAOA depth p")
    ax1.set_ylabel("probability")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(df["p"], df["approx_ratio"], "^-", color="tab:red",
             label="approx ratio")
    ax2.set_ylabel("approximation ratio  <C>/C_opt")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper center")
    ax1.set_title(f"QAOA depth sweep on {inst.name} "
                  f"({qubo.num_vars} qubits, C_opt={c_opt:.0f})")
    fig.tight_layout()
    out_png = f"{results_dir()}/sweep_depth_{inst.name}.png"
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_csv} and {out_png}")


if __name__ == "__main__":
    main()
