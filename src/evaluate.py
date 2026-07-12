"""Decoding and evaluation utilities shared by every solver.

Given a raw bit-vector returned by any solver, these helpers decode it into a
per-aircraft manoeuvre assignment and evaluate the two things that matter for
the benchmark: feasibility (one manoeuvre per aircraft AND conflict-free) and
solution quality (number of manoeuvring aircraft).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qubo import QUBO


@dataclass
class Solution:
    assignment: list[int]        # manoeuvre index chosen per aircraft (or -1 if invalid)
    one_hot_ok: bool             # every aircraft selected exactly one manoeuvre
    conflict_free: bool          # no residual pairwise conflict
    feasible: bool               # one_hot_ok AND conflict_free
    num_manoeuvres: int          # number of non-nominal manoeuvres
    energy: float
    residual_conflicts: list[tuple[int, int]]


def evaluate(qubo: QUBO, x: np.ndarray) -> Solution:
    inst = qubo.instance
    man = qubo.manoeuvres
    decoded = qubo.decode(x)

    one_hot_ok = all(len(sel) == 1 for sel in decoded)
    assignment = [int(sel[0]) if len(sel) == 1 else -1 for sel in decoded]

    residual: list[tuple[int, int]] = []
    conflict_free = True
    if one_hot_ok:
        vel = [inst.velocity(i, man[assignment[i]].q, man[assignment[i]].dtheta)
               for i in range(inst.n)]
        for (i, j) in inst.pairs:
            if inst.in_conflict(i, vel[i], j, vel[j]):
                residual.append((i, j))
        conflict_free = len(residual) == 0
    else:
        conflict_free = False

    num_man = sum(1 for i in range(inst.n)
                  if assignment[i] >= 0 and not man[assignment[i]].is_nominal)

    feasible = one_hot_ok and conflict_free
    return Solution(assignment=assignment, one_hot_ok=one_hot_ok,
                    conflict_free=conflict_free, feasible=feasible,
                    num_manoeuvres=num_man, energy=qubo.energy(x),
                    residual_conflicts=residual)
