"""Run QAOA on a Circle Problem instance.

Examples
--------
    python3 src/run_qaoa.py --n 3 --p 1                 # numpy backend (no deps)
    python3 src/run_qaoa.py --n 3 --p 2 --backend qiskit
    python3 src/run_qaoa.py --n 3 --p 2 --backend qiskit --noisy

The numpy backend is exact (statevector) and dependency-free but limited to
small instances (V <= ~16 to stay fast). The qiskit backend runs the thesis
implementation and supports a noisy campaign with --noisy.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "solvers"))

from instance import Instance
from manoeuvres import build_manoeuvre_set
from qubo import build_qubo
from evaluate import evaluate


def main() -> None:
    ap = argparse.ArgumentParser(description="QAOA on the ACRP QUBO.")
    ap.add_argument("--n", type=int, default=3, help="number of aircraft")
    ap.add_argument("--p", type=int, default=1, help="QAOA depth")
    ap.add_argument("--heading", type=int, default=5, help="heading options per aircraft")
    ap.add_argument("--backend", choices=["numpy", "qiskit"], default="numpy")
    ap.add_argument("--noisy", action="store_true", help="qiskit noisy campaign")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    inst = Instance.circle_problem(n=args.n, radius=100.0, d=8.0)
    man = build_manoeuvre_set(n_heading=args.heading, n_speed=1)
    qubo = build_qubo(inst, man)
    print(f"{inst.name}: aircraft={args.n}, manoeuvres={len(man)}, "
          f"QUBO vars={qubo.num_vars}, p={args.p}, backend={args.backend}")

    t = time.time()
    if args.backend == "numpy":
        from qaoa_numpy import solve_qaoa_numpy
        r = solve_qaoa_numpy(qubo, p=args.p, seed=args.seed)
        extra = f"P(opt)={r.success_probability:.4f}"
    else:
        from qaoa_qiskit import solve_qaoa_qiskit
        r = solve_qaoa_qiskit(qubo, p=args.p, seed=args.seed, noisy=args.noisy)
        extra = f"noisy={args.noisy}"

    sol = evaluate(qubo, r.best_x)
    print(f"  best_energy={r.best_energy:.3f}  expected_energy={r.expected_energy:.3f}")
    print(f"  {extra}")
    print(f"  feasible={sol.feasible}  manoeuvres={sol.num_manoeuvres}  "
          f"assignment={sol.assignment}")
    print(f"  elapsed={time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
