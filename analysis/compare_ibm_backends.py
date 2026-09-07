#!/usr/bin/env python3
"""Compare accessible IBM QPUs for the final fixed QAOA circuit.

NO circuit is submitted to hardware. This script only performs local
statevector optimisation and local ISA transpilation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "solvers"))

from instance import Instance
from manoeuvres import build_manoeuvre_set
from qubo import build_qubo
from qaoa_qiskit import qubo_to_ising, _build_circuit, solve_qaoa_qiskit


def safe_instruction_error(backend, name, qargs):
    """Return calibrated instruction error if available."""
    try:
        props = backend.target[name][tuple(qargs)]
        if props is not None and props.error is not None:
            return float(props.error)
    except (KeyError, TypeError):
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--heading", type=int, default=3)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--out",
        default=os.path.join(
            ROOT,
            "results",
            "ibm_backend_comparison_cp3.json",
        ),
    )
    args = ap.parse_args()

    # ------------------------------------------------------------
    # 1. Same problem and QUBO as the hardware experiment
    # ------------------------------------------------------------
    inst = Instance.circle_problem(
        n=args.n,
        radius=100.0,
        d=8.0,
    )

    man = build_manoeuvre_set(
        n_heading=args.heading,
        n_speed=1,
    )

    qubo = build_qubo(inst, man)

    v = qubo.num_vars
    M = len(man)

    print(
        f"{inst.name}: n={args.n}, M={M}, "
        f"qubits={v}, p={args.p}"
    )

    # ------------------------------------------------------------
    # 2. Optimise QAOA angles locally
    # ------------------------------------------------------------
    print("\nOptimising angles locally...")

    sim = solve_qaoa_qiskit(
        qubo,
        p=args.p,
        seed=args.seed,
        noisy=False,
    )

    gammas = sim.params[:args.p]
    betas = sim.params[args.p:]

    print(
        "gammas =",
        np.round(gammas, 6),
    )
    print(
        "betas  =",
        np.round(betas, 6),
    )

    # ------------------------------------------------------------
    # 3. Build final fixed measured circuit
    # ------------------------------------------------------------
    h, J, _ = qubo_to_ising(
        qubo.Q,
        qubo.offset,
    )

    qc = _build_circuit(
        h,
        J,
        v,
        gammas,
        betas,
    )

    qc.measure_all()

    print(
        f"\nLogical circuit: "
        f"depth={qc.depth()}, "
        f"qubits={qc.num_qubits}"
    )

    # ------------------------------------------------------------
    # 4. Connect to IBM
    # ------------------------------------------------------------
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit.transpiler import generate_preset_pass_manager

    service = QiskitRuntimeService()

    backends = service.backends(
        operational=True,
        simulator=False,
        min_num_qubits=v,
    )

    print(
        "\nAccessible QPUs:",
        [b.name for b in backends],
    )

    results = []

    # ------------------------------------------------------------
    # 5. Transpile SAME circuit for every accessible QPU
    # ------------------------------------------------------------
    for backend in backends:

        print("\n" + "=" * 70)
        print(backend.name)
        print("=" * 70)

        pm = generate_preset_pass_manager(
            optimization_level=3,
            backend=backend,
            seed_transpiler=args.seed,
        )

        isa = pm.run(qc)

        depth = isa.depth()
        twoq = isa.num_nonlocal_gates()
        ops = dict(isa.count_ops())

        # Logical-to-physical layouts
        try:
            initial_layout = isa.layout.initial_index_layout(
                filter_ancillas=True
            )
        except Exception:
            initial_layout = None

        try:
            final_layout = isa.layout.final_index_layout(
                filter_ancillas=True
            )
        except Exception:
            final_layout = None

        # Find gates actually present in the ISA circuit
        twoq_errors = []
        measured_qubits = set()

        for instruction in isa.data:

            name = instruction.operation.name

            qindices = tuple(
                isa.find_bit(q).index
                for q in instruction.qubits
            )

            if name == "measure":
                measured_qubits.update(qindices)

            if len(qindices) == 2:
                err = safe_instruction_error(
                    backend,
                    name,
                    qindices,
                )

                if err is not None:
                    twoq_errors.append(err)

        readout_errors = []

        for q in sorted(measured_qubits):
            err = safe_instruction_error(
                backend,
                "measure",
                (q,),
            )

            if err is not None:
                readout_errors.append(err)

        mean_2q_error = (
            float(np.mean(twoq_errors))
            if twoq_errors
            else None
        )

        max_2q_error = (
            float(np.max(twoq_errors))
            if twoq_errors
            else None
        )

        mean_readout_error = (
            float(np.mean(readout_errors))
            if readout_errors
            else None
        )

        print(f"ISA depth              : {depth}")
        print(f"2-qubit gates          : {twoq}")
        print(f"Initial physical layout: {initial_layout}")
        print(f"Final physical layout  : {final_layout}")

        if mean_2q_error is not None:
            print(
                f"Mean 2Q gate error     : "
                f"{mean_2q_error:.6f}"
            )

        if max_2q_error is not None:
            print(
                f"Max 2Q gate error      : "
                f"{max_2q_error:.6f}"
            )

        if mean_readout_error is not None:
            print(
                f"Mean readout error     : "
                f"{mean_readout_error:.6f}"
            )

        print(f"Native operations      : {ops}")

        results.append(
            {
                "backend": backend.name,
                "device_qubits": backend.num_qubits,
                "isa_depth": depth,
                "two_qubit_gates": twoq,
                "initial_physical_layout": initial_layout,
                "final_physical_layout": final_layout,
                "mean_2q_error": mean_2q_error,
                "max_2q_error": max_2q_error,
                "mean_readout_error": mean_readout_error,
                "operations": ops,
            }
        )

    # ------------------------------------------------------------
    # 6. Simple structural ranking
    # ------------------------------------------------------------
    ranked = sorted(
        results,
        key=lambda r: (
            r["two_qubit_gates"],
            r["isa_depth"],
            (
                r["mean_2q_error"]
                if r["mean_2q_error"] is not None
                else float("inf")
            ),
        ),
    )

    print("\n")
    print("#" * 70)
    print("STRUCTURAL RANKING")
    print("#" * 70)

    for i, r in enumerate(ranked, start=1):
        print(
            f"{i}. {r['backend']}: "
            f"2Q={r['two_qubit_gates']}, "
            f"depth={r['isa_depth']}, "
            f"mean_2Q_error={r['mean_2q_error']}"
        )

    # ------------------------------------------------------------
    # 7. Save comparison
    # ------------------------------------------------------------
    output = {
        "problem": inst.name,
        "n": args.n,
        "M": M,
        "qubits": v,
        "p": args.p,
        "seed": args.seed,
        "gammas": [float(x) for x in gammas],
        "betas": [float(x) for x in betas],
        "backends": results,
        "structural_ranking": [
            r["backend"]
            for r in ranked
        ],
    }

    os.makedirs(
        os.path.dirname(args.out),
        exist_ok=True,
    )

    with open(args.out, "w") as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        f"\nSaved comparison to:\n{args.out}"
    )

    print(
        "\nNO HARDWARE JOB WAS SUBMITTED."
    )


if __name__ == "__main__":
    main()
