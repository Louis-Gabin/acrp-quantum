#!/usr/bin/env python3
"""Controlled CP3 QAOA confirmation campaign on IBM hardware.

Commands:
  prepare   Reconstruct, optimise, validate and freeze a local JSON plan.
  submit    Validate a frozen plan, transpile eight PUBs and submit one job.
  retrieve  Retrieve a submitted job from its manifest and save all results.

``prepare`` never creates an IBM Runtime service. Existing files are never
overwritten. The historical depth-series scripts and JSON results are inputs,
not mutable artefacts of this program.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "solvers"))

from evaluate import evaluate  # noqa: E402
from instance import Instance  # noqa: E402
from manoeuvres import build_manoeuvre_set  # noqa: E402
from qaoa_numpy import _energy_spectrum, expectation  # noqa: E402
from qubo import build_qubo  # noqa: E402

SCHEMA_VERSION = 1
BACKEND = "ibm_kingston"
SHOTS = 4096
OPTIMISER_SEED = 20260830
TRANSPILE_SEED = 20260830
PRIOR_JOB_ID = "da8aqcse74ec73aie27g"
PUB_ORDER = [
    "p2_A", "p6", "p3_true", "p2_B",
    "p4", "p3_null", "p1", "p2_C",
]
REFERENCE = {
    1: (14.488900, 0.0945), 2: (10.359086, 0.2188),
    3: (10.351181, 0.2188), 4: (8.812859, 0.2741),
    6: (5.844989, 0.4804),
}
FALLBACK_ANGLES = {
    1: {"gammas": [0.046778], "betas": [2.686096]},
    2: {"gammas": [-0.040383, 1.419208], "betas": [2.055518, 1.353375]},
    3: {"gammas": [-0.040533, 1.419084, -0.000369], "betas": [2.050531, 1.354938, 0.001241]},
    4: {"gammas": [4.816115, 5.347342, 2.493471, 3.680582], "betas": [1.496994, 2.708949, 1.529883, 1.341728]},
    6: {
        "gammas": [4.815614, 5.347997, 2.504787, 3.686824, 0.297277, 0.319005],
        "betas": [1.477915, 2.711700, 1.558716, 1.288251, 0.065849, -0.156647],
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def software(qiskit: bool = False) -> dict[str, Any]:
    out = {"python": platform.python_version(), "numpy": np.__version__, "scipy": version("scipy")}
    if qiskit:
        out.update({"qiskit": version("qiskit"), "qiskit_ibm_runtime": version("qiskit-ibm-runtime")})
    return out


def canonical_hash(payload: dict[str, Any]) -> str:
    clean = copy.deepcopy(payload)
    clean.pop("plan_sha256", None)
    blob = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, payload: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, default=str)
            handle.write("\n")
    except FileExistsError as error:
        raise RuntimeError(f"Refusing to overwrite existing file: {path}") from error


def load_json(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
        return json.load(handle)


def qubo_fingerprint(qubo) -> str:
    matrix = qubo.Q.toarray() if hasattr(qubo.Q, "toarray") else qubo.Q
    matrix = np.ascontiguousarray(np.asarray(matrix, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(matrix.shape).encode())
    digest.update(matrix.tobytes())
    digest.update(np.float64(qubo.offset).tobytes())
    return digest.hexdigest()


def build_problem():
    instance = Instance.circle_problem(n=3, radius=100.0, d=8.0)
    manoeuvres = build_manoeuvre_set(n_heading=3, n_speed=1)
    qubo = build_qubo(instance, manoeuvres)
    energies = np.asarray(_energy_spectrum(qubo.Q, qubo.offset), dtype=float)
    feasible = np.zeros(len(energies), dtype=bool)
    counts = np.full(len(energies), -1, dtype=int)
    for index in range(len(energies)):
        x = np.array([(index >> bit) & 1 for bit in range(qubo.num_vars)], dtype=float)
        solution = evaluate(qubo, x)
        if solution.feasible:
            feasible[index] = True
            counts[index] = solution.num_manoeuvres
    optimum = int(counts[feasible].min())
    optimal = feasible & (counts == optimum)
    return instance, manoeuvres, qubo, energies, feasible, optimal, optimum


def contains(obj: Any, text: str) -> bool:
    if isinstance(obj, dict):
        return any(contains(k, text) or contains(v, text) for k, v in obj.items())
    if isinstance(obj, list):
        return any(contains(v, text) for v in obj)
    return text in str(obj)


def collect_angles(obj: Any, records: dict[int, dict[str, list[float]]], inherited=None) -> None:
    if isinstance(obj, dict):
        depth = obj.get("qaoa_depth_p", obj.get("depth", obj.get("p", inherited)))
        if "gammas" in obj and "betas" in obj and depth is not None:
            try:
                p = int(depth)
                gammas = [float(v) for v in obj["gammas"]]
                betas = [float(v) for v in obj["betas"]]
                if len(gammas) == len(betas) == p:
                    records[p] = {"gammas": gammas, "betas": betas}
            except (TypeError, ValueError):
                pass
        for value in obj.values():
            collect_angles(value, records, depth)
    elif isinstance(obj, list):
        for value in obj:
            collect_angles(value, records, inherited)


def previous_angles(results_dir: Path) -> tuple[dict[int, Any], str]:
    required = {1, 2, 3, 4, 6}
    for path in sorted(results_dir.rglob("*.json")) if results_dir.exists() else []:
        try:
            data = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not contains(data, PRIOR_JOB_ID):
            continue
        records: dict[int, Any] = {}
        collect_angles(data, records)
        if required <= records.keys():
            return {p: records[p] for p in required}, str(path.resolve())
    return copy.deepcopy(FALLBACK_ANGLES), "handover fallback rounded to six decimals"


def metrics(energies, feasible, optimal, angles) -> dict[str, Any]:
    energy, probs = expectation(energies, np.asarray(angles["gammas"]), np.asarray(angles["betas"]))
    probs = np.asarray(probs, dtype=float)
    return {
        "expected_qubo_energy": float(energy),
        "ideal_feasibility": float(probs[feasible].sum()),
        "ideal_p_opt": float(probs[optimal].sum()),
        "probabilities": probs,
    }


def optimise_p3(energies, restarts: int, maxiter: int) -> dict[str, Any]:
    from scipy.optimize import minimize

    rng = np.random.default_rng(OPTIMISER_SEED)
    best = None
    successful = 0

    def objective(theta):
        value, _ = expectation(energies, theta[:3], theta[3:])
        return float(value)

    for run in range(restarts):
        start = np.r_[rng.uniform(0, 2 * np.pi, 3), rng.uniform(-np.pi, np.pi, 3)]
        result = minimize(objective, start, method="COBYLA", tol=1e-9,
                          options={"maxiter": maxiter, "rhobeg": 0.75, "disp": False})
        if result.success:
            successful += 1
            if best is None or float(result.fun) < float(best.fun):
                best = result
        if run == 0 or (run + 1) % 12 == 0 or run + 1 == restarts:
            shown = math.inf if best is None else float(best.fun)
            print(f"  p=3 restart {run + 1:3d}/{restarts}: best converged E={shown:.9f}")
    if best is None:
        raise RuntimeError("No independent p=3 COBYLA restart converged.")
    theta = np.asarray(best.x, dtype=float)
    return {
        "gammas": theta[:3].tolist(), "betas": theta[3:].tolist(),
        "optimizer": {"method": "scipy_COBYLA_multistart", "seed": OPTIMISER_SEED,
                      "random_independent_restarts": restarts, "successful_restarts": successful,
                      "maxiter": maxiter, "selected_on": "expected_qubo_energy",
                      "message": str(best.message), "nfev": int(best.nfev)},
    }


def circuit_crosscheck(qubo, energies, angles) -> float:
    from qaoa_qiskit import _build_circuit, qubo_to_ising
    from qiskit.quantum_info import Statevector

    h, J, _ = qubo_to_ising(qubo.Q, qubo.offset)
    circuit = _build_circuit(h, J, qubo.num_vars, angles["gammas"], angles["betas"])
    qiskit_probs = np.asarray(Statevector.from_instruction(circuit).probabilities())
    _, numpy_probs = expectation(energies, np.asarray(angles["gammas"]), np.asarray(angles["betas"]))
    return float(np.max(np.abs(qiskit_probs - np.asarray(numpy_probs))))


def validate_plan(payload, qubo, energies, feasible, optimal) -> list[dict[str, Any]]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Unsupported plan schema.")
    if payload.get("plan_sha256") != canonical_hash(payload):
        raise RuntimeError("Plan payload SHA-256 mismatch.")
    if payload.get("qubo_sha256") != qubo_fingerprint(qubo):
        raise RuntimeError("Reconstructed QUBO does not match the frozen plan.")
    if payload.get("backend") != BACKEND or payload.get("shots") != SHOTS:
        raise RuntimeError(f"Protocol requires {BACKEND} and {SHOTS} shots.")
    circuits = payload.get("circuits", [])
    if [item.get("label") for item in circuits] != PUB_ORDER:
        raise RuntimeError("PUB order differs from the reviewed protocol.")
    p2 = [item for item in circuits if item["label"].startswith("p2_")]
    if len(p2) != 3 or any((x["gammas"], x["betas"]) != (p2[0]["gammas"], p2[0]["betas"]) for x in p2[1:]):
        raise RuntimeError("p2_A, p2_B and p2_C are not identical logical circuits.")
    for item in circuits:
        observed = metrics(energies, feasible, optimal, item)
        for key in ("expected_qubo_energy", "ideal_feasibility", "ideal_p_opt"):
            if not math.isclose(observed[key], item[key], abs_tol=1e-10, rel_tol=1e-10):
                raise RuntimeError(f"Frozen metric mismatch for {item['label']}: {key}")
    return circuits


def command_prepare(args) -> None:
    print("LOCAL PREPARATION ONLY: no IBM service will be created.")
    instance, manoeuvres, qubo, energies, feasible, optimal, optimum = build_problem()

    # Validation A: exact bounded CP3 structure.
    if (qubo.num_vars, len(energies), int(feasible.sum()), int(optimal.sum()), optimum) != (9, 512, 8, 6, 2):
        raise RuntimeError("Validation A failed: CP3 structure is not 9/512/8/6/2.")
    print("A PASS  CP3 structure: 9 qubits, 512 states, 8 feasible, 6 optimal, optimum 2")

    old, source = previous_angles(Path(args.results_dir))
    old_metrics = {p: metrics(energies, feasible, optimal, old[p]) for p in (1, 2, 3, 4, 6)}
    # Validation B: protect against recovering angles from the wrong campaign.
    for p, (ref_energy, ref_popt) in REFERENCE.items():
        if abs(old_metrics[p]["expected_qubo_energy"] - ref_energy) > 0.05 or abs(old_metrics[p]["ideal_p_opt"] - ref_popt) > 0.01:
            raise RuntimeError(f"Validation B failed at p={p}; recovered angles do not match the audited campaign.")
    print(f"B PASS  Previous submitted angles recovered from: {source}")

    true_p3 = optimise_p3(energies, args.restarts, args.maxiter)
    true_metrics = metrics(energies, feasible, optimal, true_p3)
    # Validation C: independent p3 must add actual algorithmic value.
    if true_metrics["expected_qubo_energy"] >= old_metrics[3]["expected_qubo_energy"] - 1e-4:
        raise RuntimeError("Validation C failed: independent p=3 did not improve the null-layer p=3 energy.")
    print(f"C PASS  Independent p=3 improves E from {old_metrics[3]['expected_qubo_energy']:.9f} to {true_metrics['expected_qubo_energy']:.9f}")

    max_diffs = {f"p{p}": circuit_crosscheck(qubo, energies, old[p]) for p in (1, 2, 3, 4, 6)}
    max_diffs["p3_true"] = circuit_crosscheck(qubo, energies, true_p3)
    if max(max_diffs.values()) > 1e-8:
        raise RuntimeError(f"Validation D failed: NumPy/Qiskit max probability difference {max(max_diffs.values()):.3e}")
    print(f"D PASS  NumPy/Qiskit statevectors agree (max delta p={max(max_diffs.values()):.3e})")

    if max(abs(old[3]["gammas"][-1]), abs(old[3]["betas"][-1])) > 0.01 or abs(old_metrics[3]["expected_qubo_energy"] - old_metrics[2]["expected_qubo_energy"]) > 0.05:
        raise RuntimeError("Validation E failed: historical p=3 is not a verified near-null-layer control.")
    print("E PASS  Historical p=3 is a near-null-layer control")

    angle_map = {"p1": old[1], "p2": old[2], "p3_null": old[3], "p3_true": true_p3, "p4": old[4], "p6": old[6]}
    circuits = []
    for label in PUB_ORDER:
        key = "p2" if label.startswith("p2_") else label
        angles = angle_map[key]
        observed = metrics(energies, feasible, optimal, angles)
        circuits.append({
            "label": label, "depth": len(angles["gammas"]),
            "role": {"p3_true": "independently_optimised_depth_point", "p3_null": "null_layer_noise_cost_control"}.get(label, "drift_control" if label.startswith("p2_") else "depth_point"),
            "gammas": [float(v) for v in angles["gammas"]], "betas": [float(v) for v in angles["betas"]],
            "expected_qubo_energy": observed["expected_qubo_energy"],
            "ideal_feasibility": observed["ideal_feasibility"], "ideal_p_opt": observed["ideal_p_opt"],
        })
    payload = {
        "schema_version": SCHEMA_VERSION, "created_at_utc": utc_now(),
        "mode": "frozen_local_plan_no_ibm_contact", "instance": instance.name,
        "num_aircraft": 3, "heading_options": 3, "num_manoeuvres": len(manoeuvres),
        "d_min": 8.0, "circuit_qubits": 9, "qubo_sha256": qubo_fingerprint(qubo),
        "backend": BACKEND, "shots": SHOTS, "pub_order": PUB_ORDER,
        "prior_job_id": PRIOR_JOB_ID, "previous_angle_source": source,
        "optimizer_p3_true": true_p3["optimizer"], "transpile_seed": TRANSPILE_SEED,
        "common_initial_physical_layout_required": True,
        "sampler_options": {"dynamical_decoupling": False, "twirling_gates": False, "twirling_measurements": False},
        "validations": {"A_problem": True, "B_prior_angles": True, "C_true_p3": True,
                        "D_statevector": True, "E_null_layer": True, "statevector_max_differences": max_diffs},
        "software_versions": software(qiskit=True), "circuits": circuits,
    }
    payload["plan_sha256"] = canonical_hash(payload)
    # Validation F: round-trip integrity and exact experimental order.
    validate_plan(payload, qubo, energies, feasible, optimal)
    payload["validations"]["F_frozen_plan"] = True
    payload["plan_sha256"] = canonical_hash(payload)
    validate_plan(payload, qubo, energies, feasible, optimal)
    out = Path(args.plan_out) if args.plan_out else Path(args.results_dir) / f"ibm_replication_control_plan_cp3_{stamp()}.json"
    write_new(out, payload)
    print(f"F PASS  Frozen plan is internally consistent; payload SHA-256={payload['plan_sha256']}")
    print(f"Saved new plan: {out.expanduser().resolve()}")


def backend_name(backend) -> str:
    value = backend.name
    return value() if callable(value) else str(value)


def initial_layout(isa, n: int) -> list[int]:
    layout = getattr(isa, "layout", None)
    if layout is None or not hasattr(layout, "initial_index_layout"):
        raise RuntimeError("Qiskit did not expose the initial physical layout.")
    result = [int(v) for v in layout.initial_index_layout(filter_ancillas=True)]
    if len(result) != n:
        raise RuntimeError(f"Expected {n} layout entries, got {result}.")
    return result


def final_layout(isa, n: int) -> list[int] | None:
    layout = getattr(isa, "layout", None)
    if layout is None or not hasattr(layout, "final_index_layout"):
        return None
    result = [int(v) for v in layout.final_index_layout(filter_ancillas=True)]
    return result if len(result) == n else None


def command_submit(args) -> None:
    plan_path = Path(args.plan).expanduser().resolve()
    plan_file_sha = file_hash(plan_path)
    payload = load_json(plan_path)
    _, _, qubo, energies, feasible, optimal, _ = build_problem()
    entries = validate_plan(payload, qubo, energies, feasible, optimal)
    if args.backend != BACKEND or args.shots != SHOTS:
        raise RuntimeError(f"Reviewed protocol is fixed to --backend {BACKEND} --shots {SHOTS}.")

    from qaoa_qiskit import _build_circuit, qubo_to_ising
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerOptions, SamplerV2

    service = QiskitRuntimeService()
    backend = service.backend(BACKEND)
    if not backend.status().operational or getattr(backend, "simulator", False) or backend.num_qubits < 9:
        raise RuntimeError(f"{BACKEND} is not a suitable operational real backend.")
    h, J, _ = qubo_to_ising(qubo.Q, qubo.offset)
    logical = {}
    for item in entries:
        key = "p2_shared" if item["label"].startswith("p2_") else item["label"]
        if key not in logical:
            circuit = _build_circuit(h, J, 9, item["gammas"], item["betas"])
            circuit.measure_all()
            logical[key] = circuit
    probe_pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=TRANSPILE_SEED)
    probe = probe_pm.run(logical["p6"])
    common = initial_layout(probe, 9)
    fixed_pm = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                            seed_transpiler=TRANSPILE_SEED, initial_layout=common)
    isa_by_key = {key: fixed_pm.run(circuit) for key, circuit in logical.items()}
    pubs, transpilation = [], []
    for item in entries:
        key = "p2_shared" if item["label"].startswith("p2_") else item["label"]
        isa = isa_by_key[key]
        if initial_layout(isa, 9) != common:
            raise RuntimeError(f"Common layout validation failed for {item['label']}.")
        pubs.append(isa.copy())
        transpilation.append({"label": item["label"], "isa_depth": int(isa.depth()),
                              "two_qubit_gates": int(isa.num_nonlocal_gates()),
                              "initial_physical_layout": common, "final_physical_layout": final_layout(isa, 9)})
    options = SamplerOptions()
    options.dynamical_decoupling.enable = False
    options.twirling.enable_gates = False
    options.twirling.enable_measure = False
    print(f"Backend {backend_name(backend)}; layout {common}; {len(pubs)} PUBs; {SHOTS} shots each")
    for record in transpilation:
        print(f"  {record['label']:8s} depth={record['isa_depth']:4d}  2q={record['two_qubit_gates']:4d}")
    if input('Type exactly "SUBMIT ALL" to submit this single IBM job: ') != "SUBMIT ALL":
        print("Submission cancelled. No QPU job was sent.")
        return
    if file_hash(plan_path) != plan_file_sha:
        raise RuntimeError("Plan file changed after validation; submission blocked.")
    job = SamplerV2(mode=backend, options=options).run(pubs, shots=SHOTS)
    manifest = {
        "schema_version": SCHEMA_VERSION, "campaign_id": stamp(), "submitted_at_utc": utc_now(),
        "backend": BACKEND, "job_id": job.job_id(), "plan_file": str(plan_path),
        "plan_file_sha256": plan_file_sha, "plan_payload_sha256": payload["plan_sha256"],
        "shots_per_pub": SHOTS, "pub_order": PUB_ORDER, "common_initial_physical_layout": common,
        "transpilation": transpilation, "software_versions": software(qiskit=True),
    }
    out = Path(args.manifest_out) if args.manifest_out else Path(args.results_dir) / f"ibm_replication_control_submission_cp3_{manifest['campaign_id']}.json"
    write_new(out, manifest)
    print(f"Job ID: {job.job_id()}")
    print(f"Submission manifest saved immediately: {out.expanduser().resolve()}")
    print("Use the retrieve command with this manifest; submit does not wait for completion.")


def wilson(successes: int, trials: int) -> list[float]:
    z = 1.959963985
    p = successes / trials
    den = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / den
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / den
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def command_retrieve(args) -> None:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_json(manifest_path)
    plan_path = Path(manifest["plan_file"])
    if file_hash(plan_path) != manifest["plan_file_sha256"]:
        raise RuntimeError("Frozen plan file SHA-256 no longer matches the submission manifest.")
    payload = load_json(plan_path)
    instance, manoeuvres, qubo, energies, feasible, optimal, optimum = build_problem()
    entries = validate_plan(payload, qubo, energies, feasible, optimal)
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    job = service.job(manifest["job_id"])
    result = job.result()
    if len(result) != len(entries):
        raise RuntimeError(f"IBM returned {len(result)} results for {len(entries)} expected PUBs.")
    try:
        usage = float(job.usage())
    except Exception as error:  # noqa: BLE001
        usage = None
        usage_note = str(error)
    else:
        usage_note = None
    runs = []
    for index, item in enumerate(entries):
        counts = {str(k): int(v) for k, v in result[index].data.meas.get_counts().items()}
        shots = sum(counts.values())
        if shots == 0:
            raise RuntimeError(f"No shots returned for {item['label']}.")
        feasible_shots = optimum_shots = 0
        for bitstring, count in counts.items():
            x = np.array([int(bit) for bit in reversed(bitstring)], dtype=float)
            solution = evaluate(qubo, x)
            if solution.feasible:
                feasible_shots += count
                if solution.num_manoeuvres == optimum:
                    optimum_shots += count
        trans = manifest["transpilation"][index]
        run = {**item, **trans, "pub_index": index, "shots": shots,
               "feasible_shots": feasible_shots, "optimum_shots": optimum_shots,
               "feasibility_rate": feasible_shots / shots, "p_opt": optimum_shots / shots,
               "feasibility_ci95_shot_noise": wilson(feasible_shots, shots),
               "p_opt_ci95_shot_noise": wilson(optimum_shots, shots),
               "p_opt_conditional_on_feasible": optimum_shots / feasible_shots if feasible_shots else None,
               "counts": dict(sorted(counts.items(), key=lambda pair: pair[1], reverse=True))}
        runs.append(run)
        print(f"{item['label']:8s} feasibility={100*run['feasibility_rate']:.3f}%  P(opt)={100*run['p_opt']:.3f}%")
    output = {
        "schema_version": SCHEMA_VERSION, "campaign_id": manifest["campaign_id"],
        "completed_at_utc": utc_now(), "backend": manifest["backend"], "job_id": manifest["job_id"],
        "manifest_file": str(manifest_path), "manifest_file_sha256": file_hash(manifest_path),
        "plan_file": str(plan_path), "plan_file_sha256": manifest["plan_file_sha256"],
        "instance": instance.name, "num_manoeuvres": len(manoeuvres), "optimum_manoeuvres": optimum,
        "uniform_feasibility": float(feasible.mean()), "uniform_p_opt": float(optimal.mean()),
        "common_initial_physical_layout": manifest["common_initial_physical_layout"],
        "quantum_seconds_job_total": usage, "quantum_seconds_note": usage_note,
        "software_versions": software(qiskit=True), "runs": runs,
    }
    out = Path(args.out) if args.out else Path(args.results_dir) / f"ibm_replication_control_results_cp3_{manifest['campaign_id']}.json"
    write_new(out, output)
    print(f"Saved combined results: {out.expanduser().resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="local-only optimisation and frozen plan")
    prepare.add_argument("--restarts", type=int, default=96)
    prepare.add_argument("--maxiter", type=int, default=5000)
    prepare.add_argument("--plan-out")
    submit = sub.add_parser("submit", help="submit one frozen eight-PUB job")
    submit.add_argument("--plan", required=True)
    submit.add_argument("--backend", default=BACKEND)
    submit.add_argument("--shots", type=int, default=SHOTS)
    submit.add_argument("--manifest-out")
    retrieve = sub.add_parser("retrieve", help="retrieve by submission manifest")
    retrieve.add_argument("--manifest", required=True)
    retrieve.add_argument("--out")
    args = parser.parse_args()
    if getattr(args, "restarts", 1) < 1 or getattr(args, "maxiter", 1) < 1:
        parser.error("--restarts and --maxiter must be positive")
    return args


def main() -> None:
    args = parse_args()
    {"prepare": command_prepare, "submit": command_submit, "retrieve": command_retrieve}[args.command](args)


if __name__ == "__main__":
    main()
