"""Compare QAOA on the ACRP QUBO with and without a real device-noise model.

Why this script exists
----------------------
The `--noisy` path of `run_qaoa.py` samples a finite number of shots from an
IDEAL simulator (it builds a default SamplerV2 with no noise model attached),
and it reports `expected_energy` computed on the ideal statevector regardless
of the flag. As a result the noiseless and noisy summaries are identical and
the effect of noise is invisible.

A second problem is the OPTIMISER. `solve_qaoa_qiskit` runs a single COBYLA
start, so at some depths it lands in a poor local optimum (this is why p=2 came
out worse than p=1 and p=3). The dependency-free `solve_qaoa_numpy` uses 200
random restarts, which is why its depth curve is smooth. This script therefore
adds restarts around the Qiskit optimisation: for each depth it optimises from
`--restarts` different seeds and keeps the parameters with the lowest expected
energy before sampling.

What it computes
----------------
  1. Best-of-N optimised (gamma, beta) using the existing solve_qaoa_qiskit.
  2. Samples the SAME circuit twice: once on an ideal simulator and once on an
     Aer simulator carrying an explicit depolarising noise model.
  3. From each distribution: success probability P(opt), feasibility rate
     (share of probability on one-hot, conflict-free states), and the
     sample-based expected energy.

Place this file in the `analysis/` folder and run:
    cd analysis
    python3 qaoa_noise_compare.py               # CP_3, restarts=8
    python3 qaoa_noise_compare.py --n 4         # CP_4
    python3 qaoa_noise_compare.py --restarts 12 # more restarts, steadier curve

Requires: qiskit, qiskit-aer, scipy, numpy. matplotlib is optional (used only
for the figures); if it is missing the CSV is still written.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

# --- make the project's src and src/solvers importable (script lives in analysis/) ---
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "solvers"))

from instance import Instance
from manoeuvres import build_manoeuvre_set
from qubo import build_qubo
from evaluate import evaluate
from qaoa_qiskit import solve_qaoa_qiskit, qubo_to_ising, _build_circuit


def brute_force_optimum(qubo) -> float:
    """Minimum QUBO energy over all 2^V assignments (V is small here)."""
    v = qubo.num_vars
    best = np.inf
    for mask in range(1 << v):
        x = np.array([(mask >> k) & 1 for k in range(v)], dtype=float)
        e = qubo.energy(x)
        if e < best:
            best = e
    return float(best)


def decode_bitstring(bitstring: str, v: int) -> np.ndarray:
    """Same convention as qaoa_qiskit: reverse the bitstring, one bit per variable."""
    bits = bitstring.replace(" ", "")
    return np.array([int(b) for b in reversed(bits)][:v], dtype=float)


def best_of_n_params(qubo, p: int, restarts: int, base_seed: int):
    """Optimise QAOA from several seeds, keep the lowest ideal expected energy."""
    best = None
    for i in range(restarts):
        r = solve_qaoa_qiskit(qubo, p=p, seed=base_seed + i, noisy=False)
        if best is None or r.expected_energy < best.expected_energy:
            best = r
    return best


def distribution_metrics(counts: dict, qubo, opt_energy: float, tol: float = 1e-6):
    """Turn a counts/probabilities dict into P(opt), feasibility rate, <E>."""
    v = qubo.num_vars
    total = float(sum(counts.values()))
    p_opt = 0.0
    p_feasible = 0.0
    exp_energy = 0.0
    for bitstring, weight in counts.items():
        pr = float(weight) / total
        x = decode_bitstring(bitstring, v)
        e = qubo.energy(x)
        exp_energy += pr * e
        if abs(e - opt_energy) <= tol:
            p_opt += pr
        if evaluate(qubo, x).feasible:
            p_feasible += pr
    return p_opt, p_feasible, exp_energy


def make_noise_model(p1: float, p2: float):
    """Simple depolarising noise: p1 on 1-qubit gates, p2 on the 2-qubit gate."""
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return nm


def ideal_distribution(qc):
    from qiskit.quantum_info import Statevector
    return Statevector(qc).probabilities_dict()


def noisy_distribution(qc, noise_model, shots: int):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    sim = AerSimulator(noise_model=noise_model)
    meas = qc.copy()
    meas.measure_all()
    # Transpile to the basis the noise model targets. Pass basis_gates to the
    # transpiler (not the backend) to avoid the coupling_map warning.
    tqc = transpile(meas, basis_gates=["rz", "sx", "x", "cx"],
                    optimization_level=1)
    result = sim.run(tqc, shots=shots).result()
    return result.get_counts()


def main() -> None:
    ap = argparse.ArgumentParser(description="QAOA noiseless vs noisy comparison.")
    ap.add_argument("--n", type=int, default=3, help="number of aircraft")
    ap.add_argument("--heading", type=int, default=3, help="heading options per aircraft")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--restarts", type=int, default=8,
                    help="random restarts of the COBYLA optimisation per depth")
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--p1", type=float, default=0.001, help="1-qubit depolarising rate")
    ap.add_argument("--p2", type=float, default=0.01, help="2-qubit depolarising rate")
    args = ap.parse_args()

    inst = Instance.circle_problem(n=args.n, radius=100.0, d=8.0)
    man = build_manoeuvre_set(n_heading=args.heading, n_speed=1)
    qubo = build_qubo(inst, man)
    v = qubo.num_vars
    opt_energy = brute_force_optimum(qubo)
    h, J, _ = qubo_to_ising(qubo.Q, qubo.offset)
    noise_model = make_noise_model(args.p1, args.p2)

    print(f"{inst.name}: aircraft={args.n}, manoeuvres={len(man)}, "
          f"QUBO vars={v}, exact optimum energy={opt_energy:.3f}")
    print(f"optimiser: best of {args.restarts} restarts per depth")
    print(f"noise model: depolarising p1={args.p1} (1q), p2={args.p2} (2q), "
          f"shots={args.shots}\n")
    header = (f"{'p':>2} | {'mode':>9} | {'P(opt)':>8} | {'feasible':>9} | "
             f"{'<E>':>8}")
    print(header)
    print("-" * len(header))

    rows = []
    for p in args.depths:
        r = best_of_n_params(qubo, p, args.restarts, args.seed)
        gammas, betas = r.params[:p], r.params[p:]
        qc = _build_circuit(h, J, v, gammas, betas)

        ideal = ideal_distribution(qc)
        noisy = noisy_distribution(qc, noise_model, args.shots)

        for mode, dist in (("noiseless", ideal), ("noisy", noisy)):
            p_opt, p_feas, exp_e = distribution_metrics(dist, qubo, opt_energy)
            print(f"{p:>2} | {mode:>9} | {p_opt:>8.4f} | {p_feas:>9.4f} | "
                  f"{exp_e:>8.3f}")
            rows.append({"instance": inst.name, "p": p, "mode": mode,
                         "P_opt": p_opt, "feasible_rate": p_feas,
                         "expected_energy": exp_e})
        print("-" * len(header))

    # Save a CSV next to the other analysis outputs if a results/ folder exists.
    results_dir = os.path.join(HERE, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"qaoa_noise_compare_{inst.name}.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved {out}")

    # Optional plot (skipped silently if matplotlib is unavailable).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        depths = args.depths
        for metric, ylabel in (("P_opt", "P(opt)"), ("feasible_rate", "feasible rate")):
            plt.figure(figsize=(6, 4))
            for mode in ("noiseless", "noisy"):
                ys = [next(r[metric] for r in rows if r["p"] == p and r["mode"] == mode)
                      for p in depths]
                plt.plot(depths, ys, marker="o", label=mode)
            plt.xlabel("QAOA depth p")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} vs depth ({inst.name})")
            plt.legend()
            plt.tight_layout()
            fig_out = os.path.join(results_dir, f"qaoa_noise_{metric}_{inst.name}.png")
            plt.savefig(fig_out, dpi=150)
            print(f"saved {fig_out}")
            plt.close()
    except Exception as exc:  # pragma: no cover
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":
    main()
