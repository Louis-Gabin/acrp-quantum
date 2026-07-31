#!/usr/bin/env python3
"""Qubit / memory budget for the two-track (M=3, M=5) study.

This script produces, reproducibly, the two thesis tables that justify the whole
experimental design:

  - how many qubits each (n, M) instance needs (qubits = n * M),
  - the statevector memory it would take to simulate QAOA exactly,
  - whether that simulation is feasible locally (8 GB), only on a large HPC,
    or impossible on any classical machine,
  - a rough order-of-magnitude estimate of the local QAOA run time.

It is pure standard library (no numpy / no Gurobi), so it always runs and the
numbers in the thesis come straight from code rather than from hand computation.

Key facts encoded here
----------------------
  * qubits          = n * M
  * statevector RAM = 2**qubits * 16 bytes  (one complex128 amplitude per basis
                      state). QAOA optimisation needs a few copies, so the true
                      footprint is a small multiple of this.
  * noisy (density-matrix) simulation costs 2**(2*qubits) amplitudes, i.e. it
    squares the memory; that is why the noisy ladder stops far earlier than the
    ideal one (about 13 qubits on 8 GB).

Examples
--------
    python3 analysis/qubit_budget.py                 # default 8 GB, prints tables
    python3 analysis/qubit_budget.py --ram-gb 16
    python3 analysis/qubit_budget.py --sizes 3 4 5 6 7 8 9 10 12 25 50
"""

from __future__ import annotations

import argparse
import os

# Sizes that appear in the study: the small QAOA-simulable ladder plus the four
# large target instances requested (n = 10, 12, 25, 50).
DEFAULT_SIZES = [3, 4, 5, 6, 7, 8, 9, 10, 12, 25, 50]
M_TRACKS = [3, 5]

BYTES_PER_AMPLITUDE = 16  # complex128


def human_bytes(nbytes: float) -> str:
    # Above 1 PB the number is astronomically large; scientific notation in
    # octets is far more readable (and honest) than "562949953421312 YB".
    if nbytes >= 1024.0 ** 5:
        import math
        exp = int(math.log10(nbytes))
        mant = nbytes / (10.0 ** exp)
        return f"~{mant:.1f}e{exp} octets"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    x = float(nbytes)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            if u in ("B", "KB", "MB"):
                return f"{x:.0f} {u}"
            return f"{x:.1f} {u}"
        x /= 1024.0
    return f"{x:.1f} PB"


def statevector_bytes(qubits: int) -> float:
    return (2.0 ** qubits) * BYTES_PER_AMPLITUDE


def feasibility(qubits: int, ram_gb: float) -> str:
    """Classify exact ideal-statevector QAOA simulation feasibility."""
    ram_bytes = ram_gb * (1024 ** 3)
    need = statevector_bytes(qubits)
    # Keep ~half the RAM for the OS, Python, and QAOA working copies.
    if need <= 0.5 * ram_bytes:
        return "Oui, local"
    if qubits <= 27:
        return "Local (limite)"
    if qubits <= 45:
        return "HPC seulement"
    return "Impossible (toute machine)"


def est_local_time(qubits: int) -> str:
    """Order-of-magnitude QAOA (p=2, few restarts) run time on an M3 laptop.

    These are deliberately coarse brackets; the binding constraint near the top
    is memory bandwidth over 2**qubits amplitudes.
    """
    if qubits <= 12:
        return "< 1 min"
    if qubits <= 15:
        return "1-3 min"
    if qubits <= 18:
        return "5-15 min"
    if qubits <= 21:
        return "20-45 min"
    if qubits <= 24:
        return "1-3 h"
    if qubits <= 27:
        return "3-8 h (nuit)"
    return "hors capacite locale"


def noisy_feasible_local(qubits: int, ram_gb: float) -> str:
    """Noisy density-matrix simulation squares the memory (2**(2q))."""
    ram_bytes = ram_gb * (1024 ** 3)
    need = (2.0 ** (2 * qubits)) * BYTES_PER_AMPLITUDE
    return "oui" if need <= 0.5 * ram_bytes else "non"


def build_rows(sizes, m, ram_gb):
    rows = []
    for n in sizes:
        q = n * m
        rows.append({
            "n": n,
            "M": m,
            "qubits": q,
            "statevector_ram": human_bytes(statevector_bytes(q)),
            "qaoa_sim_ideal": feasibility(q, ram_gb),
            "qaoa_sim_noisy_local": noisy_feasible_local(q, ram_gb),
            "est_local_time": est_local_time(q),
        })
    return rows


def markdown_table(rows) -> str:
    headers = ["n", "qubits (n*M)", "RAM statevecteur",
               "Simulation QAOA (ideale)", "Sim. bruitee locale",
               "Temps local estime"]
    keys = ["n", "qubits", "statevector_ram", "qaoa_sim_ideal",
            "qaoa_sim_noisy_local", "est_local_time"]
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r[k]) for k in keys) + " |")
    return "\n".join(out)


def csv_dump(rows, path) -> None:
    keys = ["n", "M", "qubits", "statevector_ram", "qaoa_sim_ideal",
            "qaoa_sim_noisy_local", "est_local_time"]
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")


def results_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(os.path.dirname(here), "results")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser(description="Qubit / memory budget tables.")
    ap.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    ap.add_argument("--ram-gb", type=float, default=8.0, dest="ram_gb")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    md_sections = []
    for m in M_TRACKS:
        rows = build_rows(args.sizes, m, args.ram_gb)
        title = f"### Budget qubits, M = {m} (RAM de reference : {args.ram_gb:.0f} Go)"
        table = markdown_table(rows)
        print("\n" + title + "\n")
        print(table)
        md_sections.append(title + "\n\n" + table + "\n")
        if not args.no_save:
            path = f"{results_dir()}/qubit_budget_m{m}.csv"
            csv_dump(rows, path)
            print(f"\n[saved] {path}")

    if not args.no_save:
        md_path = f"{results_dir()}/qubit_budget.md"
        with open(md_path, "w") as fh:
            fh.write("# Budget qubits et faisabilite de simulation\n\n")
            fh.write("\n".join(md_sections))
        print(f"\n[saved] {md_path}")


if __name__ == "__main__":
    main()
