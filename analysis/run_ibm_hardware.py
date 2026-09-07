#!/usr/bin/env python3
"""Submit one fixed QAOA circuit to real IBM Quantum hardware.

The variational angles are optimised on the local statevector simulator. Only
the final measured circuit is submitted to the selected QPU, so the outer
optimisation loop never consumes QPU time.

Before running this script, compare the currently accessible backends with:

    python3 analysis/compare_ibm_backends.py --n 3 --heading 3 --p 1 --seed 1

No hardware job is submitted unless the user types exactly ``SUBMIT`` at the
interactive confirmation prompt.

Prerequisites are documented in IBM_QUANTUM_RUNBOOK.md. Typical usage:

    python3 analysis/run_ibm_hardware.py \
        --n 3 --heading 3 --p 1 --shots 4096 --seed 1
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

from instance import Instance  # noqa: E402
from manoeuvres import build_manoeuvre_set  # noqa: E402
from qubo import build_qubo  # noqa: E402
from evaluate import evaluate  # noqa: E402
from qaoa_qiskit import (  # noqa: E402
    _build_circuit,
    qubo_to_ising,
    solve_qaoa_qiskit,
)


def _x_from_bitstring(bitstring: str) -> np.ndarray:
    """Map Qiskit's displayed bit order back to QUBO variable order."""
    return np.array([int(bit) for bit in reversed(bitstring)], dtype=float)


def optimum_manoeuvres(qubo) -> int | None:
    """Return the exact feasible optimum by enumerating all bitstrings."""
    best = None
    for bits in itertools.product((0, 1), repeat=qubo.num_vars):
        solution = evaluate(qubo, np.array(bits, dtype=float))
        if solution.feasible and (
            best is None or solution.num_manoeuvres < best
        ):
            best = solution.num_manoeuvres
    return best


def _backend_name(backend) -> str:
    """Return a backend name across supported Qiskit Runtime versions."""
    name = backend.name
    return name() if callable(name) else str(name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one fixed QAOA circuit on real IBM Quantum hardware."
    )
    parser.add_argument("--n", type=int, default=3, help="number of aircraft")
    parser.add_argument(
        "--heading", type=int, default=3, help="number of heading options"
    )
    parser.add_argument("--p", type=int, default=1, help="QAOA depth")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--backend",
        default="ibm_kingston",
        help=(
            "IBM QPU backend (default: ibm_kingston, selected from the "
            "pre-run backend comparison)"
        ),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "results", "ibm_hardware_cp3.json"),
    )
    args = parser.parse_args()

    if args.n < 1 or args.heading < 1 or args.p < 1 or args.shots < 1:
        parser.error("--n, --heading, --p and --shots must all be positive")
    return args


def main() -> None:
    args = _parse_args()

    # 1) Build the same QUBO as the simulator campaign.
    instance = Instance.circle_problem(n=args.n, radius=100.0, d=8.0)
    manoeuvres = build_manoeuvre_set(n_heading=args.heading, n_speed=1)
    qubo = build_qubo(instance, manoeuvres)
    num_vars, num_manoeuvres = qubo.num_vars, len(manoeuvres)
    print(
        f"{instance.name}: n={args.n}, M={num_manoeuvres}, "
        f"qubits={num_vars}, p={args.p}"
    )

    # 2) Optimise angles locally. This does not use IBM hardware.
    simulation = solve_qaoa_qiskit(
        qubo, p=args.p, seed=args.seed, noisy=False
    )
    gammas = simulation.params[: args.p]
    betas = simulation.params[args.p :]
    print(
        "Optimised angles  "
        f"gammas={np.round(gammas, 6)}  betas={np.round(betas, 6)}"
    )

    # 3) Compute an exact reference optimum for P(opt).
    optimum = optimum_manoeuvres(qubo)
    print(f"Reference optimum manoeuvres: {optimum}")

    # 4) Build the final fixed circuit and add measurements.
    h, J, _ = qubo_to_ising(qubo.Q, qubo.offset)
    circuit = _build_circuit(h, J, num_vars, gammas, betas)
    circuit.measure_all()

    # 5) Connect to the explicitly selected backend and validate it.
    from qiskit_ibm_runtime import (
        QiskitRuntimeService,
        SamplerOptions,
        SamplerV2 as Sampler,
    )

    service = QiskitRuntimeService()
    backend = service.backend(args.backend)
    backend_name = _backend_name(backend)
    status = backend.status()

    if not status.operational:
        raise RuntimeError(f"{backend_name} is currently not operational.")
    if getattr(backend, "simulator", False):
        raise RuntimeError(f"{backend_name} is a simulator, not a real QPU.")
    if backend.num_qubits < num_vars:
        raise RuntimeError(
            f"{backend_name} has {backend.num_qubits} qubits, but the circuit "
            f"requires {num_vars}."
        )

    print(f"Backend: {backend_name} ({backend.num_qubits} qubits, operational)")

    # 6) Reproduce the pre-run transpilation seed for this exact backend.
    from qiskit.transpiler import generate_preset_pass_manager

    pass_manager = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
        seed_transpiler=args.seed,
    )
    isa = pass_manager.run(circuit)
    isa_depth = isa.depth()
    two_qubit_gates = isa.num_nonlocal_gates()
    print(f"Transpiled depth={isa_depth}  2q-gates={two_qubit_gates}")

    # Pin error-suppression options so future default changes cannot alter the
    # experimental protocol silently.
    options = SamplerOptions()
    options.dynamical_decoupling.enable = False
    options.twirling.enable_gates = False
    options.twirling.enable_measure = False

    print("\n" + "=" * 70)
    print("FINAL HARDWARE SUBMISSION")
    print("=" * 70)
    print(f"Backend        : {backend_name}")
    print(f"Logical qubits : {num_vars}")
    print(f"ISA depth      : {isa_depth}")
    print(f"2Q gates       : {two_qubit_gates}")
    print(f"Shots          : {args.shots}")
    print(f"QAOA depth p   : {args.p}")
    print(f"Seed           : {args.seed}")
    print(f"Gammas         : {[float(value) for value in gammas]}")
    print(f"Betas          : {[float(value) for value in betas]}")
    print("Dynamical decoupling: disabled")
    print("Gate/measurement twirling: disabled")

    confirmation = input(
        '\nType exactly "SUBMIT" to send this circuit to the IBM QPU: '
    )
    if confirmation != "SUBMIT":
        print("Submission cancelled. No QPU job was sent.")
        return

    # 7) This is the only line that submits work to IBM hardware.
    sampler = Sampler(mode=backend, options=options)
    started_at = time.time()
    job = sampler.run([isa], shots=args.shots)
    print(f"Job ID: {job.job_id()}  status: {job.status()}")
    result = job.result()
    wall_seconds = time.time() - started_at
    counts = result[0].data.meas.get_counts()

    # 8) Compute feasibility and optimum-hit metrics.
    shots_received = sum(counts.values())
    if shots_received == 0:
        raise RuntimeError("IBM returned no measurement shots.")

    feasible_shots = 0
    optimum_shots = 0
    for bitstring, count in counts.items():
        solution = evaluate(qubo, _x_from_bitstring(bitstring))
        if solution.feasible:
            feasible_shots += count
            if (
                optimum is not None
                and solution.num_manoeuvres == optimum
            ):
                optimum_shots += count

    feasibility_rate = feasible_shots / shots_received
    p_opt = optimum_shots / shots_received
    print(
        f"shots={shots_received}  feasibility_rate={feasibility_rate:.3f}  "
        f"P(opt)={p_opt:.3f}"
    )

    # Calibration values are deliberately not collected here: physical qubits
    # are determined by the transpiler layout. The separate comparison file is
    # the pre-run calibration snapshot and backend-selection record.
    comparison_path = os.path.join(
        ROOT, "results", "ibm_backend_comparison_cp3.json"
    )
    sorted_counts = sorted(
        counts.items(), key=lambda item: item[1], reverse=True
    )
    output = {
        "backend": backend_name,
        "backend_operational_at_submission": bool(status.operational),
        "num_qubits_device": backend.num_qubits,
        "circuit_qubits": num_vars,
        "depth_transpiled": isa_depth,
        "two_qubit_gates": two_qubit_gates,
        "qaoa_depth_p": args.p,
        "seed": args.seed,
        "shots_requested": args.shots,
        "shots": shots_received,
        "feasibility_rate": feasibility_rate,
        "p_opt": p_opt,
        "optimum_manoeuvres": optimum,
        "job_id": job.job_id(),
        "wall_seconds": round(wall_seconds, 1),
        "gammas": [float(value) for value in gammas],
        "betas": [float(value) for value in betas],
        "sampler_options": {
            "dynamical_decoupling": False,
            "twirling_gates": False,
            "twirling_measurements": False,
        },
        "pre_run_comparison_file": os.path.relpath(comparison_path, ROOT),
        "counts_top10": dict(sorted_counts[:10]),
        "counts": dict(sorted_counts),
    }

    output_path = os.path.abspath(os.path.expanduser(args.out))
    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
