#!/usr/bin/env python3
"""Prepare and submit a controlled CP_3 QAOA depth series to IBM hardware.

The workflow is deliberately split in two commands:

1. ``--local-only`` optimises and validates the angles, then writes an
   immutable plan. It never creates an IBM service and cannot submit a job.
2. ``--plan PATH`` reloads exactly that plan, validates it again, transpiles
   all circuits with a common initial physical layout, and submits every depth
   in one SamplerV2 job after an explicit ``SUBMIT ALL`` confirmation.

The original ``analysis/run_ibm_hardware.py`` remains the script of record for
the first CP_3, p=1 experiment and is not modified by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "solvers"))

from instance import Instance  # noqa: E402
from manoeuvres import build_manoeuvre_set  # noqa: E402
from qubo import build_qubo  # noqa: E402
from evaluate import evaluate  # noqa: E402
from qaoa_numpy import _energy_spectrum, expectation  # noqa: E402


SCHEMA_VERSION = 4
DEFAULT_DEPTHS = [1, 2, 3, 4, 6]
MAX_ENUMERABLE_QUBITS = 22
ENERGY_TOLERANCE = 1e-8

# Angles used in the original CP_3, p=1 hardware run. They are included as a
# deterministic optimisation candidate. They are compared on expected QUBO
# energy, never on P(opt).
CP3_P1_REFERENCE_GAMMA = 4.805503657095598
CP3_P1_REFERENCE_BETA = 2.730936055686168


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _campaign_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _software_versions(include_qiskit: bool = False) -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": _package_version("scipy"),
    }
    if include_qiskit:
        versions["qiskit"] = _package_version("qiskit")
        versions["qiskit_ibm_runtime"] = _package_version(
            "qiskit-ibm-runtime"
        )
    return versions


def _absolute(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(_absolute(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_json(path: str, payload: Any) -> None:
    """Write JSON atomically enough for this workflow and never overwrite."""
    absolute = _absolute(path)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    try:
        with open(absolute, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
    except FileExistsError as error:
        raise RuntimeError(
            f"Refusing to overwrite an existing result: {absolute}"
        ) from error


def _x_from_bitstring(bitstring: str) -> np.ndarray:
    """Map Qiskit's displayed bit order back to QUBO variable order."""
    return np.array([int(bit) for bit in reversed(bitstring)], dtype=float)


def _backend_name(backend) -> str:
    name = backend.name
    return name() if callable(name) else str(name)


def _is_simulator(backend) -> bool:
    marker = getattr(backend, "simulator", None)
    if marker is not None:
        return bool(marker)
    try:
        return bool(backend.configuration().simulator)
    except (AttributeError, TypeError):
        return False


def _qubo_matrix(qubo) -> np.ndarray:
    matrix = qubo.Q.toarray() if hasattr(qubo.Q, "toarray") else qubo.Q
    return np.ascontiguousarray(np.asarray(matrix, dtype=np.float64))


def _qubo_fingerprint(qubo) -> str:
    matrix = _qubo_matrix(qubo)
    digest = hashlib.sha256()
    digest.update(str(matrix.shape).encode("ascii"))
    digest.update(matrix.tobytes())
    digest.update(np.float64(qubo.offset).tobytes())
    return digest.hexdigest()


def classify_all_states(qubo):
    """Return exhaustive feasibility and optimality masks for small QUBOs."""
    num_vars = qubo.num_vars
    if num_vars > MAX_ENUMERABLE_QUBITS:
        raise RuntimeError(
            f"{num_vars} qubits cannot be exhaustively classified by this "
            f"script (limit: {MAX_ENUMERABLE_QUBITS})."
        )

    num_states = 1 << num_vars
    feasible = np.zeros(num_states, dtype=bool)
    manoeuvre_counts = np.full(num_states, -1, dtype=int)
    for index in range(num_states):
        x = np.array(
            [(index >> qubit) & 1 for qubit in range(num_vars)],
            dtype=float,
        )
        solution = evaluate(qubo, x)
        if solution.feasible:
            feasible[index] = True
            manoeuvre_counts[index] = solution.num_manoeuvres

    if not feasible.any():
        raise RuntimeError("No feasible state exists for this instance.")

    optimum = int(manoeuvre_counts[feasible].min())
    optimal = feasible & (manoeuvre_counts == optimum)
    return feasible, manoeuvre_counts, optimum, optimal


def _expected_qubo(energies, theta: np.ndarray, depth: int) -> float:
    value, _ = expectation(energies, theta[:depth], theta[depth:])
    return float(value)


def _extend_angles(
    theta: np.ndarray, previous_depth: int, target_depth: int
) -> np.ndarray:
    """Embed a shallower solution by appending zero-valued QAOA layers."""
    if target_depth <= previous_depth:
        raise ValueError("target_depth must exceed previous_depth")
    gammas = theta[:previous_depth]
    betas = theta[previous_depth:]
    padding = np.zeros(target_depth - previous_depth, dtype=float)
    return np.concatenate([gammas, padding, betas, padding])


def optimise_angles(
    energies,
    depth: int,
    seed: int,
    random_restarts: int,
    maxiter: int,
    deterministic_starts: list[tuple[str, np.ndarray]],
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Run multi-start COBYLA and retain the lowest expected QUBO energy."""
    try:
        from scipy.optimize import minimize
    except ImportError as error:
        raise RuntimeError(
            "SciPy is required. Activate the project .venv and install scipy "
            "before preparing a hardware plan."
        ) from error

    def objective(theta):
        return _expected_qubo(energies, np.asarray(theta), depth)

    rng = np.random.default_rng(seed)
    starts = [(label, np.asarray(start, dtype=float)) for label, start in deterministic_starts]
    for restart in range(random_restarts):
        start = np.concatenate(
            [
                rng.uniform(0.0, 2.0 * np.pi, depth),
                rng.uniform(0.0, np.pi, depth),
            ]
        )
        starts.append((f"random_{restart + 1}", start))

    best_theta: np.ndarray | None = None
    best_value = math.inf
    best_metadata: dict[str, Any] = {}
    successful_runs = 0

    for label, start in starts:
        if start.shape != (2 * depth,):
            raise RuntimeError(
                f"Invalid deterministic start {label!r} for p={depth}: "
                f"expected {2 * depth} angles, got {start.size}."
            )

        # Retain the unmodified deterministic candidate. In particular, this
        # makes the padded p-1 solution an exact upper bound at depth p.
        start_value = objective(start)
        if np.isfinite(start_value) and start_value < best_value:
            best_theta = start.copy()
            best_value = start_value
            best_metadata = {
                "source": f"{label}_unmodified",
                "optimizer_success": None,
                "optimizer_message": "deterministic candidate",
                "optimizer_nfev": 0,
            }

        outcome = minimize(
            objective,
            start,
            method="COBYLA",
            options={"maxiter": maxiter, "rhobeg": 0.3},
        )
        if bool(outcome.success):
            successful_runs += 1
        value = float(outcome.fun)
        # A run that only exhausted MAXFUN is not a validated optimum.
        # Only converged COBYLA outcomes may replace the deterministic
        # warm-start baseline.
        replaces_unmodified_tie = (
            best_metadata.get("optimizer_success") is not True
            and value <= best_value + ENERGY_TOLERANCE
        )
        if (
            bool(outcome.success)
            and np.isfinite(value)
            and (value < best_value or replaces_unmodified_tie)
        ):
            best_theta = np.asarray(outcome.x, dtype=float)
            best_value = value
            best_metadata = {
                "source": label,
                "optimizer_success": bool(outcome.success),
                "optimizer_message": str(outcome.message),
                "optimizer_nfev": int(getattr(outcome, "nfev", -1)),
            }

    if best_theta is None or not np.isfinite(best_value):
        raise RuntimeError(f"COBYLA produced no finite candidate for p={depth}.")
    if successful_runs == 0:
        raise RuntimeError(
            f"No COBYLA run reported convergence for p={depth}. Increase "
            "--maxiter before using these angles on hardware."
        )
    if best_metadata.get("optimizer_success") is not True:
        raise RuntimeError(
            f"The best validated candidate for p={depth} is still the "
            "unmodified warm start. No converged COBYLA run matched or "
            "improved it. Increase --maxiter; no hardware plan was written."
        )

    metadata = {
        **best_metadata,
        "method": "scipy_COBYLA_multistart",
        "random_restarts": random_restarts,
        "deterministic_starts": [label for label, _ in deterministic_starts],
        "total_minimizations": len(starts),
        "successful_minimizations": successful_runs,
        "maxiter": maxiter,
    }
    return best_theta, best_value, metadata


def wilson_interval(successes: int, trials: int, z: float = 1.959963985):
    if trials == 0:
        return (0.0, 0.0)
    phat = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (phat + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * np.sqrt(
            phat * (1.0 - phat) / trials
            + z * z / (4.0 * trials**2)
        )
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or submit a controlled IBM QAOA depth series."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--local-only",
        action="store_true",
        help="optimise and save an immutable plan; never contact IBM",
    )
    mode.add_argument(
        "--plan",
        help="validated local plan to submit; required for hardware mode",
    )
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--heading", type=int, default=3)
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=None,
        help="local-plan depths (default: 1 2 3 4 6)",
    )
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--restarts",
        type=int,
        default=48,
        help="random COBYLA restarts per depth (default: 48)",
    )
    parser.add_argument("--maxiter", type=int, default=4000)
    parser.add_argument("--backend", default="ibm_kingston")
    parser.add_argument("--plan-out", default=None)
    parser.add_argument(
        "--out-dir", default=os.path.join(ROOT, "results")
    )
    args = parser.parse_args()

    if args.n < 1 or args.heading < 1 or args.shots < 1:
        parser.error("--n, --heading and --shots must be positive")
    if args.restarts < 1 or args.maxiter < 1:
        parser.error("--restarts and --maxiter must be positive")
    if args.depths is not None:
        if any(depth < 1 for depth in args.depths):
            parser.error("every depth must be positive")
        if len(set(args.depths)) != len(args.depths):
            parser.error("--depths must not contain duplicates")
    if args.plan and args.plan_out:
        parser.error("--plan-out is only valid with --local-only")
    if args.plan and args.depths is not None:
        parser.error("--depths is read from --plan in hardware mode")
    return args


def _build_problem(num_aircraft: int, heading_options: int):
    instance = Instance.circle_problem(
        n=num_aircraft, radius=100.0, d=8.0
    )
    manoeuvres = build_manoeuvre_set(
        n_heading=heading_options, n_speed=1
    )
    qubo = build_qubo(instance, manoeuvres)
    return instance, manoeuvres, qubo


def prepare_local_plan(args: argparse.Namespace) -> None:
    submission_depths = sorted(args.depths or DEFAULT_DEPTHS)
    if submission_depths != DEFAULT_DEPTHS:
        raise RuntimeError(
            f"This reviewed campaign requires submission depths "
            f"{DEFAULT_DEPTHS}; received {submission_depths}."
        )
    # Optimise every intermediate depth even when it is not destined for the
    # QPU. In particular, p=5 provides a continuous warm-start bridge from
    # p=4 to p=6 without adding a sixth hardware circuit.
    optimisation_depths = list(
        range(submission_depths[0], submission_depths[-1] + 1)
    )
    instance, manoeuvres, qubo = _build_problem(args.n, args.heading)
    num_vars = qubo.num_vars
    print(
        f"{instance.name}: n={args.n}, M={len(manoeuvres)}, "
        f"qubits={num_vars}, submission depths={submission_depths}"
    )
    bridge_depths = [
        depth for depth in optimisation_depths
        if depth not in submission_depths
    ]
    if bridge_depths:
        print(
            f"Local-only bridge depths: {bridge_depths} "
            "(optimised but not submitted)"
        )

    energies = _energy_spectrum(qubo.Q, qubo.offset)
    feasible, _, optimum, optimal = classify_all_states(qubo)
    num_states = feasible.size
    uniform_feasibility = float(feasible.sum()) / num_states
    uniform_p_opt = float(optimal.sum()) / num_states
    print(
        f"Exact optimum: {optimum} manoeuvring aircraft. "
        f"{int(feasible.sum())}/{num_states} states feasible; "
        f"{int(optimal.sum())} optimal."
    )
    print(
        f"Uniform baseline: feasibility={uniform_feasibility*100:.4f}%  "
        f"P(opt)={uniform_p_opt*100:.4f}%"
    )

    plans: list[dict[str, Any]] = []
    previous_theta: np.ndarray | None = None
    previous_depth: int | None = None
    previous_energy: float | None = None
    p1_reference_energy: float | None = None

    print("\nLocal COBYLA optimisation, selected on expected QUBO energy")
    print(
        f"{'p':>3} {'role':>8} {'E[QUBO]':>13} {'ideal feas':>12} "
        f"{'ideal P(opt)':>13} {'successful':>11}  best source"
    )

    for depth in optimisation_depths:
        deterministic_starts: list[tuple[str, np.ndarray]] = []
        if (
            depth == 1
            and args.n == 3
            and args.heading == 3
        ):
            reference = np.array(
                [CP3_P1_REFERENCE_GAMMA, CP3_P1_REFERENCE_BETA],
                dtype=float,
            )
            deterministic_starts.append(("original_cp3_p1", reference))
            p1_reference_energy = _expected_qubo(energies, reference, 1)

        if previous_theta is not None and previous_depth is not None:
            deterministic_starts.append(
                (
                    f"p{previous_depth}_padded_with_zero_layers",
                    _extend_angles(previous_theta, previous_depth, depth),
                )
            )

        theta, expected_qubo, metadata = optimise_angles(
            energies=energies,
            depth=depth,
            seed=args.seed + depth,
            random_restarts=args.restarts,
            maxiter=args.maxiter,
            deterministic_starts=deterministic_starts,
        )

        if (
            previous_energy is not None
            and expected_qubo > previous_energy + ENERGY_TOLERANCE
        ):
            raise RuntimeError(
                f"Expected QUBO energy increased from p={previous_depth} "
                f"({previous_energy:.12g}) to p={depth} "
                f"({expected_qubo:.12g}). No hardware plan was written."
            )
        if (
            depth == 1
            and p1_reference_energy is not None
            and expected_qubo > p1_reference_energy + ENERGY_TOLERANCE
        ):
            raise RuntimeError(
                "The selected p=1 angles are worse in expected QUBO energy "
                "than the original hardware angles."
            )

        gammas = theta[:depth]
        betas = theta[depth:]
        verified_energy, probabilities = expectation(
            energies, gammas, betas
        )
        ideal_feasibility = float(probabilities[feasible].sum())
        ideal_p_opt = float(probabilities[optimal].sum())
        plan = {
            "depth": depth,
            "submit_to_hardware": depth in submission_depths,
            "gammas": [float(value) for value in gammas],
            "betas": [float(value) for value in betas],
            "expected_qubo_energy": float(verified_energy),
            "ideal_feasibility_same_angles": ideal_feasibility,
            "ideal_p_opt_same_angles": ideal_p_opt,
            "angle_selection_criterion": "expected_qubo_energy",
            "optimizer": metadata,
        }
        plans.append(plan)
        role = "submit" if depth in submission_depths else "bridge"
        print(
            f"{depth:>3} {role:>8} {float(verified_energy):>13.6f} "
            f"{ideal_feasibility*100:>11.2f}% "
            f"{ideal_p_opt*100:>12.2f}% "
            f"{metadata['successful_minimizations']:>4}/"
            f"{metadata['total_minimizations']:<6}  {metadata['source']}"
        )
        previous_theta = theta
        previous_depth = depth
        previous_energy = float(verified_energy)

    output_directory = _absolute(args.out_dir)
    if args.plan_out:
        plan_path = _absolute(args.plan_out)
    else:
        plan_path = os.path.join(
            output_directory,
            f"ibm_depth_series_plan_cp{args.n}_{_campaign_stamp()}.json",
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "mode": "local_plan_no_ibm_contact",
        "instance": instance.name,
        "num_aircraft": args.n,
        "heading_options": args.heading,
        "num_manoeuvres": len(manoeuvres),
        "circuit_qubits": num_vars,
        "qubo_sha256": _qubo_fingerprint(qubo),
        "depths": submission_depths,
        "optimisation_depths": optimisation_depths,
        "optimum_manoeuvres": optimum,
        "uniform_feasibility": uniform_feasibility,
        "uniform_p_opt": uniform_p_opt,
        "seed": args.seed,
        "random_restarts_per_depth": args.restarts,
        "maxiter": args.maxiter,
        "p1_reference_expected_qubo_energy": p1_reference_energy,
        "software_versions": _software_versions(),
        "plans": plans,
    }
    _write_new_json(plan_path, payload)
    print("\nVALIDATION PASSED")
    print("- SciPy COBYLA converged for every retained depth.")
    print("- Expected QUBO energy is non-increasing with depth.")
    if bridge_depths:
        print(f"- Local bridge depths {bridge_depths} will not be submitted.")
    if p1_reference_energy is not None:
        print("- p=1 is no worse in energy than the original p=1 angles.")
    print("- No IBM connection was opened and no QPU job was submitted.")
    print(f"Saved immutable plan: {plan_path}")


def _load_json(path: str) -> dict[str, Any]:
    absolute = _absolute(path)
    with open(absolute, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


def _validate_loaded_plan(
    payload: dict[str, Any], qubo, energies, feasible, optimal
) -> list[dict[str, Any]]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported plan schema: {payload.get('schema_version')!r}."
        )
    if payload.get("qubo_sha256") != _qubo_fingerprint(qubo):
        raise RuntimeError(
            "The plan QUBO fingerprint does not match the reconstructed QUBO."
        )

    depths = payload.get("depths")
    optimisation_depths = payload.get("optimisation_depths")
    plans = payload.get("plans")
    if not isinstance(depths, list) or not isinstance(plans, list):
        raise RuntimeError("Malformed plan: missing depths or plans.")
    if depths != DEFAULT_DEPTHS:
        raise RuntimeError(
            f"This controlled campaign requires depths {DEFAULT_DEPTHS}; "
            f"the plan contains {depths}."
        )
    expected_optimisation_depths = list(
        range(DEFAULT_DEPTHS[0], DEFAULT_DEPTHS[-1] + 1)
    )
    if optimisation_depths != expected_optimisation_depths:
        raise RuntimeError(
            "Malformed plan: missing the local p=5 bridge optimisation."
        )
    if [item.get("depth") for item in plans] != optimisation_depths:
        raise RuntimeError(
            "Malformed plan: entries do not match optimisation depths."
        )

    previous_energy: float | None = None
    for item in plans:
        depth = int(item["depth"])
        gammas = np.asarray(item["gammas"], dtype=float)
        betas = np.asarray(item["betas"], dtype=float)
        if gammas.shape != (depth,) or betas.shape != (depth,):
            raise RuntimeError(f"Malformed angle vector at p={depth}.")
        expected, probabilities = expectation(energies, gammas, betas)
        ideal_feasibility = float(probabilities[feasible].sum())
        ideal_p_opt = float(probabilities[optimal].sum())
        checks = [
            ("expected QUBO energy", float(expected), item["expected_qubo_energy"]),
            (
                "ideal feasibility",
                ideal_feasibility,
                item["ideal_feasibility_same_angles"],
            ),
            ("ideal P(opt)", ideal_p_opt, item["ideal_p_opt_same_angles"]),
        ]
        for label, recomputed, stored in checks:
            if not math.isclose(
                recomputed, float(stored), rel_tol=1e-10, abs_tol=1e-10
            ):
                raise RuntimeError(
                    f"Plan integrity check failed for {label} at p={depth}."
                )
        if (
            previous_energy is not None
            and float(expected) > previous_energy + ENERGY_TOLERANCE
        ):
            raise RuntimeError(
                f"Loaded plan energy increases at p={depth}; submission blocked."
            )
        if item.get("angle_selection_criterion") != "expected_qubo_energy":
            raise RuntimeError(
                f"Invalid angle-selection criterion at p={depth}."
            )
        if item.get("optimizer", {}).get("method") != "scipy_COBYLA_multistart":
            raise RuntimeError(f"Invalid optimizer at p={depth}.")
        if item.get("optimizer", {}).get("optimizer_success") is not True:
            raise RuntimeError(
                f"The selected COBYLA result did not converge at p={depth}."
            )
        previous_energy = float(expected)
    submitted = [
        item for item in plans if item.get("submit_to_hardware") is True
    ]
    if [item["depth"] for item in submitted] != DEFAULT_DEPTHS:
        raise RuntimeError("Malformed plan: invalid hardware-depth selection.")
    return submitted


def _initial_layout_from_isa(isa, num_vars: int) -> list[int]:
    layout = getattr(isa, "layout", None)
    if layout is None or not hasattr(layout, "initial_index_layout"):
        raise RuntimeError("Qiskit did not expose the transpiled initial layout.")
    physical = list(layout.initial_index_layout(filter_ancillas=True))
    if len(physical) != num_vars:
        raise RuntimeError(
            f"Expected a {num_vars}-qubit initial layout, got {physical}."
        )
    return [int(qubit) for qubit in physical]


def _final_layout_from_isa(isa, num_vars: int) -> list[int] | None:
    layout = getattr(isa, "layout", None)
    if layout is None or not hasattr(layout, "final_index_layout"):
        return None
    physical = list(layout.final_index_layout(filter_ancillas=True))
    if len(physical) != num_vars:
        return None
    return [int(qubit) for qubit in physical]


def submit_plan(args: argparse.Namespace) -> None:
    plan_file_sha256 = _file_sha256(args.plan)
    payload = _load_json(args.plan)
    num_aircraft = int(payload.get("num_aircraft", -1))
    heading_options = int(payload.get("heading_options", -1))
    if num_aircraft != 3 or heading_options != 3:
        raise RuntimeError(
            "This reviewed hardware campaign is restricted to CP_3 with "
            "three heading options. CP_10 requires a separate protocol."
        )

    instance, manoeuvres, qubo = _build_problem(
        num_aircraft, heading_options
    )
    num_vars = qubo.num_vars
    energies = _energy_spectrum(qubo.Q, qubo.offset)
    feasible, _, optimum, optimal = classify_all_states(qubo)
    plans = _validate_loaded_plan(
        payload, qubo, energies, feasible, optimal
    )
    if int(payload["optimum_manoeuvres"]) != optimum:
        raise RuntimeError("The exact optimum does not match the saved plan.")

    # Qiskit and IBM imports occur only in hardware mode.
    from qaoa_qiskit import _build_circuit, qubo_to_ising
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import (
        QiskitRuntimeService,
        SamplerOptions,
        SamplerV2 as Sampler,
    )

    h, J, _ = qubo_to_ising(qubo.Q, qubo.offset)
    for plan in plans:
        circuit = _build_circuit(
            h,
            J,
            num_vars,
            plan["gammas"],
            plan["betas"],
        )
        circuit.measure_all()
        plan["circuit"] = circuit

    service = QiskitRuntimeService()
    backend = service.backend(args.backend)
    backend_name = _backend_name(backend)
    status = backend.status()
    if not status.operational:
        raise RuntimeError(f"{backend_name} is currently not operational.")
    if _is_simulator(backend):
        raise RuntimeError(f"{backend_name} is a simulator, not a real QPU.")
    if backend.num_qubits < num_vars:
        raise RuntimeError(
            f"{backend_name} has {backend.num_qubits} qubits; "
            f"the circuit requires {num_vars}."
        )

    seed_transpiler = int(payload["seed"])
    deepest = max(plans, key=lambda item: int(item["depth"]))
    probe_manager = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
        seed_transpiler=seed_transpiler,
    )
    deepest_probe = probe_manager.run(deepest["circuit"])
    common_initial_layout = _initial_layout_from_isa(
        deepest_probe, num_vars
    )

    fixed_manager = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
        seed_transpiler=seed_transpiler,
        initial_layout=common_initial_layout,
    )
    for plan in plans:
        isa = fixed_manager.run(plan["circuit"])
        actual_initial = _initial_layout_from_isa(isa, num_vars)
        if actual_initial != common_initial_layout:
            raise RuntimeError(
                f"Common-layout check failed at p={plan['depth']}."
            )
        plan["isa"] = isa
        plan["depth_transpiled"] = int(isa.depth())
        plan["two_qubit_gates"] = int(isa.num_nonlocal_gates())
        plan["initial_physical_layout"] = actual_initial
        plan["final_physical_layout"] = _final_layout_from_isa(
            isa, num_vars
        )

    options = SamplerOptions()
    options.dynamical_decoupling.enable = False
    options.twirling.enable_gates = False
    options.twirling.enable_measure = False

    print("\n" + "=" * 78)
    print("FINAL CONTROLLED HARDWARE SUBMISSION")
    print("=" * 78)
    print(f"Plan           : {_absolute(args.plan)}")
    print(f"Plan SHA-256   : {plan_file_sha256}")
    print(f"Backend        : {backend_name}")
    print(f"Logical qubits : {num_vars}")
    print(f"Physical layout: {common_initial_layout}")
    print(f"One IBM job    : {len(plans)} circuits")
    print(f"Shots/circuit  : {args.shots}")
    print(f"Total shots    : {args.shots * len(plans)}")
    print("DD and gate/measurement twirling: disabled")
    print(
        f"\n{'p':>3} {'E[QUBO]':>12} {'ideal P(opt)':>13} "
        f"{'ISA depth':>10} {'2q gates':>9}"
    )
    for plan in plans:
        print(
            f"{plan['depth']:>3} "
            f"{plan['expected_qubo_energy']:>12.6f} "
            f"{plan['ideal_p_opt_same_angles']*100:>12.2f}% "
            f"{plan['depth_transpiled']:>10} "
            f"{plan['two_qubit_gates']:>9}"
        )

    confirmation = input(
        f'\nType exactly "SUBMIT ALL" to send all {len(plans)} circuits '
        f"to {backend_name}: "
    )
    if confirmation != "SUBMIT ALL":
        print("Submission cancelled. No QPU job was sent.")
        return

    if _file_sha256(args.plan) != plan_file_sha256:
        raise RuntimeError(
            "The plan file changed after validation; submission blocked."
        )

    output_directory = _absolute(args.out_dir)
    campaign_id = _campaign_stamp()
    sampler = Sampler(mode=backend, options=options)
    started_at = time.time()
    job = sampler.run(
        [plan["isa"] for plan in plans], shots=args.shots
    )

    manifest_path = os.path.join(
        output_directory,
        f"ibm_depth_series_submission_cp{num_aircraft}_{campaign_id}.json",
    )
    _write_new_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "submitted_at_utc": _utc_now(),
            "backend": backend_name,
            "job_id": job.job_id(),
            "plan_file": _absolute(args.plan),
            "plan_file_sha256": plan_file_sha256,
            "depths": [plan["depth"] for plan in plans],
            "shots_per_circuit": args.shots,
            "circuits_in_single_job": len(plans),
        },
    )
    print(f"Job ID: {job.job_id()}")
    print(f"Submission manifest saved immediately: {manifest_path}")

    result = job.result()
    wall_seconds = time.time() - started_at
    if len(result) != len(plans):
        raise RuntimeError(
            f"IBM returned {len(result)} PUB results for {len(plans)} circuits."
        )

    try:
        quantum_seconds = float(job.usage())
    except Exception as error:  # noqa: BLE001
        quantum_seconds = None
        usage_note = f"job.usage() unavailable: {error}"
    else:
        usage_note = None

    try:
        job_metrics = job.metrics()
    except Exception as error:  # noqa: BLE001
        job_metrics = {"note": f"job.metrics() unavailable: {error}"}

    completed: list[dict[str, Any]] = []
    for index, plan in enumerate(plans):
        depth = int(plan["depth"])
        counts = result[index].data.meas.get_counts()
        shots_received = int(sum(counts.values()))
        if shots_received == 0:
            raise RuntimeError(f"IBM returned no shots for p={depth}.")

        feasible_shots = 0
        optimum_shots = 0
        for bitstring, count in counts.items():
            solution = evaluate(qubo, _x_from_bitstring(bitstring))
            if solution.feasible:
                feasible_shots += count
                if solution.num_manoeuvres == optimum:
                    optimum_shots += count

        feasibility_rate = feasible_shots / shots_received
        p_opt = optimum_shots / shots_received
        feasibility_ci = wilson_interval(feasible_shots, shots_received)
        p_opt_ci = wilson_interval(optimum_shots, shots_received)
        sorted_counts = sorted(
            counts.items(), key=lambda item: item[1], reverse=True
        )

        record = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "completed_at_utc": _utc_now(),
            "backend": backend_name,
            "backend_operational_at_submission": bool(status.operational),
            "job_id": job.job_id(),
            "plan_file": _absolute(args.plan),
            "plan_file_sha256": plan_file_sha256,
            "pub_index": index,
            "circuits_in_single_job": len(plans),
            "quantum_seconds_job_total": quantum_seconds,
            "quantum_seconds_note": usage_note,
            "wall_seconds_job_total": round(wall_seconds, 1),
            "instance": instance.name,
            "num_aircraft": num_aircraft,
            "num_manoeuvres": len(manoeuvres),
            "circuit_qubits": num_vars,
            "qaoa_depth_p": depth,
            "depth_transpiled": plan["depth_transpiled"],
            "two_qubit_gates": plan["two_qubit_gates"],
            "initial_physical_layout": plan["initial_physical_layout"],
            "final_physical_layout": plan["final_physical_layout"],
            "optimization_level": 3,
            "seed_transpiler": seed_transpiler,
            "gammas": plan["gammas"],
            "betas": plan["betas"],
            "expected_qubo_energy_ideal": plan["expected_qubo_energy"],
            "angle_selection_criterion": plan[
                "angle_selection_criterion"
            ],
            "optimizer": plan["optimizer"],
            "shots_requested": args.shots,
            "shots": shots_received,
            "feasible_shots": int(feasible_shots),
            "optimum_shots": int(optimum_shots),
            "optimum_manoeuvres": optimum,
            "feasibility_rate": feasibility_rate,
            "p_opt": p_opt,
            "feasibility_ci95_shot_noise": list(feasibility_ci),
            "p_opt_ci95_shot_noise": list(p_opt_ci),
            "p_opt_conditional_on_feasible": (
                optimum_shots / feasible_shots if feasible_shots else None
            ),
            "uniform_feasibility": payload["uniform_feasibility"],
            "uniform_p_opt": payload["uniform_p_opt"],
            "ideal_feasibility_same_angles": plan[
                "ideal_feasibility_same_angles"
            ],
            "ideal_p_opt_same_angles": plan[
                "ideal_p_opt_same_angles"
            ],
            "sampler_options": {
                "dynamical_decoupling": False,
                "twirling_gates": False,
                "twirling_measurements": False,
            },
            "software_versions": _software_versions(include_qiskit=True),
            "counts_top10_ranked": [
                {"bitstring": bitstring, "count": int(count)}
                for bitstring, count in sorted_counts[:10]
            ],
            "counts": {
                bitstring: int(count) for bitstring, count in sorted_counts
            },
        }
        result_path = os.path.join(
            output_directory,
            f"ibm_hardware_cp{num_aircraft}_p{depth}_{campaign_id}.json",
        )
        _write_new_json(result_path, record)
        completed.append(record)
        print(
            f"p={depth}: feasibility={feasibility_rate*100:.3f}% "
            f"P(opt)={p_opt*100:.3f}%  saved={result_path}"
        )

    summary_path = os.path.join(
        output_directory,
        f"ibm_depth_series_cp{num_aircraft}_{campaign_id}.json",
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "completed_at_utc": _utc_now(),
        "instance": instance.name,
        "backend": backend_name,
        "job_id": job.job_id(),
        "plan_file": _absolute(args.plan),
        "plan_file_sha256": plan_file_sha256,
        "circuit_qubits": num_vars,
        "common_initial_physical_layout": common_initial_layout,
        "optimum_manoeuvres": optimum,
        "uniform_feasibility": payload["uniform_feasibility"],
        "uniform_p_opt": payload["uniform_p_opt"],
        "shots_per_circuit": args.shots,
        "circuits_in_single_job": len(plans),
        "quantum_seconds_job_total": quantum_seconds,
        "quantum_seconds_note": usage_note,
        "wall_seconds_job_total": round(wall_seconds, 1),
        "job_metrics": job_metrics,
        "software_versions": _software_versions(include_qiskit=True),
        "runs": [
            {
                key: value
                for key, value in record.items()
                if key not in {"counts", "counts_top10_ranked"}
            }
            for record in completed
        ],
    }
    _write_new_json(summary_path, summary)
    print("\nDEPTH SERIES COMPLETE")
    print(f"IBM-reported QPU usage: {quantum_seconds!r} seconds")
    print(f"Saved combined summary: {summary_path}")


def main() -> None:
    args = _parse_args()
    if args.local_only:
        prepare_local_plan(args)
    else:
        submit_plan(args)


if __name__ == "__main__":
    main()
