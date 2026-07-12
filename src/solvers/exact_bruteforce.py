"""Exact brute-force QUBO solver.

Enumerates all 2^V assignments and returns the global minimum-energy state. This
is tractable only for small V (roughly V <= 22). Its purpose in the benchmark is
twofold:

1. Validate the QUBO encoding: the brute-force optimum of a correctly built QUBO
   must be a feasible (one-hot, conflict-free) assignment with the minimum
   number of manoeuvres.
2. Provide a ground-truth QUBO optimum against which the simulated annealing and
   (later) QAOA samplers are scored, until David Rey's Gurobi baseline is run on
   the same instances.
"""

from __future__ import annotations

import itertools

import numpy as np


def brute_force(Q: np.ndarray, offset: float = 0.0):
    """Return (best_x, best_energy) over all binary vectors of length V."""
    v = Q.shape[0]
    if v > 24:
        raise ValueError(f"brute_force is only intended for V<=24 (got {v}).")
    best_x = None
    best_e = float("inf")
    for bits in itertools.product((0, 1), repeat=v):
        x = np.array(bits, dtype=float)
        e = float(x @ Q @ x + offset)
        if e < best_e:
            best_e = e
            best_x = x
    return best_x, best_e
