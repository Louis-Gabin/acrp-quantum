"""Unified benchmark: run every available solver on the same instances.

Runs, for each Circle Problem instance:
  - exact brute force (ground-truth QUBO optimum, V <= 22)
  - simulated annealing (classical twin)
  - QAOA numpy (exact statevector, small V)
  - QAOA qiskit (only if qiskit + scipy are installed)

Writes a tidy results table to results/benchmark.csv and prints it.

Run:  python3 src/run_benchmark.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "solvers"))

import pandas as pd

from instance import Instance
from manoeuvres import build_manoeuvre_set
from qubo import build_qubo
from evaluate import evaluate
from exact_bruteforce import brute_force
from simulated_annealing import simulated_annealing
from qaoa_numpy import solve_qaoa_numpy


def _row(instance, solver, energy, sol, seconds, extra=""):
    return {
        "instance": instance.name,
        "aircraft": instance.n,
        "solver": solver,
        "energy": round(float(energy), 3),
        "feasible": sol.feasible,
        "manoeuvres": sol.num_manoeuvres,
        "seconds": round(seconds, 2),
        "notes": extra,
    }


def _has_qiskit() -> bool:
    try:
        import qiskit  # noqa: F401
        import scipy  # noqa: F401
        return True
    except Exception:
        return False


def main(sizes=(3, 4, 5), qaoa_numpy_max_vars=16, qaoa_p=2) -> None:
    rows = []
    for n in sizes:
        inst = Instance.circle_problem(n=n, radius=100.0, d=8.0)
        man = build_manoeuvre_set(n_heading=5, n_speed=1)
        qubo = build_qubo(inst, man)

        if qubo.num_vars <= 22:
            t = time.time()
            bx, be = brute_force(qubo.Q, qubo.offset)
            rows.append(_row(inst, "exact_bruteforce", be, evaluate(qubo, bx),
                             time.time() - t, "ground truth"))

        t = time.time()
        nr = 40 if qubo.num_vars <= 20 else 120
        ns = 400 if qubo.num_vars <= 20 else 1500
        sa = simulated_annealing(qubo.Q, qubo.offset, num_reads=nr,
                                 num_sweeps=ns, seed=42)
        rows.append(_row(inst, "simulated_annealing", sa.best_energy,
                         evaluate(qubo, sa.best_x), time.time() - t,
                         f"reads={nr},sweeps={ns}"))

        if qubo.num_vars <= qaoa_numpy_max_vars:
            t = time.time()
            r = solve_qaoa_numpy(qubo, p=qaoa_p, seed=1)
            rows.append(_row(inst, f"qaoa_numpy_p{qaoa_p}", r.best_energy,
                             evaluate(qubo, r.best_x), time.time() - t,
                             f"P(opt)={r.success_probability:.4f}"))

        if _has_qiskit() and qubo.num_vars <= 18:
            from qaoa_qiskit import solve_qaoa_qiskit
            t = time.time()
            r = solve_qaoa_qiskit(qubo, p=qaoa_p, seed=1)
            rows.append(_row(inst, f"qaoa_qiskit_p{qaoa_p}", r.best_energy,
                             evaluate(qubo, r.best_x), time.time() - t, "noiseless"))

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out = "results/benchmark.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved {out}")
    if not _has_qiskit():
        print("\n[info] qiskit/scipy not found: the qiskit QAOA row was skipped. "
              "Install requirements.txt to include it.")


if __name__ == "__main__":
    main()
