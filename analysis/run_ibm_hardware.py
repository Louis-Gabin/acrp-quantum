#!/usr/bin/env python3
"""Submit the CP_3 QAOA(p=1) circuit to real IBM Quantum hardware.

Design: the variational angles are optimised on the LOCAL statevector simulator
(free, no QPU time). Only the final fixed circuit is sampled on the QPU, so the
Open-Plan minutes are spent on execution, never on the outer optimisation loop.

This script reuses the exact same QUBO / Ising / circuit builders as the rest of
the library (src/solvers/qaoa_qiskit.py), so the hardware run is directly
comparable to the noiseless and noisy simulations of Section 3.4.

Prerequisites (see IBM_QUANTUM_RUNBOOK.md):
  - IBM Quantum account on IBM Cloud (Open Plan), API key + instance CRN saved
    once with QiskitRuntimeService.save_account(...).
  - pip install "qiskit>=2.0" "qiskit-ibm-runtime>=0.40" qiskit-aer scipy numpy

Usage:
  python3 analysis/run_ibm_hardware.py --n 3 --heading 3 --p 1 --shots 4096
  python3 analysis/run_ibm_hardware.py --backend ibm_kingston   # force a device
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "solvers"))

from instance import Instance                    # noqa: E402
from manoeuvres import build_manoeuvre_set        # noqa: E402
from qubo import build_qubo                       # noqa: E402
from evaluate import evaluate                     # noqa: E402
from qaoa_qiskit import qubo_to_ising, _build_circuit, solve_qaoa_qiskit  # noqa: E402


def _x_from_bitstring(bitstring: str) -> np.ndarray:
    """Same convention as qaoa_qiskit: qubit k -> x[k] = reversed(bitstring)[k]."""
    return np.array([int(b) for b in reversed(bitstring)], dtype=float)


def optimum_manoeuvres(qubo) -> int | None:
    """Layout-agnostic reference optimum: brute force over all 2^v bitstrings.

    v = 9 for CP_3 at M=3, so this is 512 evaluations (instant). Uses the same
    x-ordering as the measured bitstrings, so it needs no assumption about the
    QUBO variable layout.
    """
    v = qubo.num_vars
    best = None
    for bits in itertools.product((0, 1), repeat=v):
        sol = evaluate(qubo, np.array(bits, dtype=float))
        if sol.feasible and (best is None or sol.num_manoeuvres < best):
            best = sol.num_manoeuvres
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="QAOA on real IBM Quantum hardware.")
    ap.add_argument("--n", type=int, default=3, help="number of aircraft")
    ap.add_argument("--heading", type=int, default=3, help="heading options (3 -> M=3)")
    ap.add_argument("--p", type=int, default=1, help="QAOA depth")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--backend", default=None,
                    help="force a backend name; default = least busy real QPU")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "ibm_hardware_cp3.json"))
    args = ap.parse_args()

    # ---- Build the same QUBO as the simulator campaign -------------------
    inst = Instance.circle_problem(n=args.n, radius=100.0, d=8.0)
    man = build_manoeuvre_set(n_heading=args.heading, n_speed=1)
    qubo = build_qubo(inst, man)
    v, M = qubo.num_vars, len(man)
    print(f"{inst.name}: n={args.n}, M={M}, qubits={v}, p={args.p}")

    # ---- 1) Optimise angles on the LOCAL simulator (free) ----------------
    sim = solve_qaoa_qiskit(qubo, p=args.p, seed=args.seed, noisy=False)
    gammas, betas = sim.params[:args.p], sim.params[args.p:]
    print(f"Optimised angles  gammas={np.round(gammas, 4)}  betas={np.round(betas, 4)}")

    # ---- 2) Reference optimum (for P(opt)) -------------------------------
    opt = optimum_manoeuvres(qubo)
    print(f"Reference optimum manoeuvres: {opt}")

    # ---- 3) Final fixed circuit with measurements ------------------------
    h, J, _ = qubo_to_ising(qubo.Q, qubo.offset)
    qc = _build_circuit(h, J, v, gammas, betas)
    qc.measure_all()

    # ---- 4) Connect and pick a backend -----------------------------------
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    service = QiskitRuntimeService()  # uses the saved default account
    if args.backend:
        backend = service.backend(args.backend)
    else:
        backend = service.least_busy(operational=True, simulator=False,
                                     min_num_qubits=v)
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")

    # ---- 5) Transpile to an ISA circuit for THIS backend -----------------
    from qiskit.transpiler import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa = pm.run(qc)
    print(f"Transpiled depth={isa.depth()}  2q-gates={isa.num_nonlocal_gates()}")

    # ---- 6) Sample on hardware (single job) ------------------------------
    sampler = Sampler(mode=backend)
    t0 = time.time()
    job = sampler.run([isa], shots=args.shots)
    print(f"Job ID: {job.job_id()}  status: {job.status()}")
    result = job.result()
    wall = time.time() - t0
    counts = result[0].data.meas.get_counts()

    # ---- 7) Metrics: feasibility rate and P(opt) -------------------------
    shots = sum(counts.values())
    feas = hit = 0
    for b, c in counts.items():
        sol = evaluate(qubo, _x_from_bitstring(b))
        if sol.feasible:
            feas += c
            if opt is not None and sol.num_manoeuvres == opt:
                hit += c
    feas_rate, p_opt = feas / shots, hit / shots
    print(f"shots={shots}  feasibility_rate={feas_rate:.3f}  P(opt)={p_opt:.3f}")

    # ---- 8) Calibration snapshot (best effort) ---------------------------
    calib = {}
    try:
        props = backend.properties()
        nq = min(v, backend.num_qubits)
        calib = {
            "t1_us": [round(props.t1(q) * 1e6, 1) for q in range(nq)],
            "t2_us": [round(props.t2(q) * 1e6, 1) for q in range(nq)],
            "readout_error": [round(props.readout_error(q), 4) for q in range(nq)],
        }
    except Exception as exc:  # noqa: BLE001
        calib = {"note": f"grab T1/T2 and gate errors from the Platform UI ({exc})"}

    out = {
        "backend": backend.name,
        "num_qubits_device": backend.num_qubits,
        "circuit_qubits": v,
        "depth_transpiled": isa.depth(),
        "two_qubit_gates": isa.num_nonlocal_gates(),
        "shots": shots,
        "feasibility_rate": feas_rate,
        "p_opt": p_opt,
        "optimum_manoeuvres": opt,
        "job_id": job.job_id(),
        "wall_seconds": round(wall, 1),
        "gammas": [float(x) for x in gammas],
        "betas": [float(x) for x in betas],
        "calibration": calib,
        "counts_top10": dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
