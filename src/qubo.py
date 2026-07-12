"""QUBO encoding of the tactical 2D ACRP (manoeuvre-selection formulation).

Decision variables
------------------
x[i, m] = 1 if aircraft i selects manoeuvre m, else 0.
The variables are flattened to a single index k = i * M + m, where M is the
number of manoeuvres. The mapping is exposed via `var_index` / `decode`.

Energy (to be minimised)
------------------------
    E(x) = C * sum_{i,m} cost[i,m] * x[i,m]                      (objective)
         + A * sum_i ( sum_m x[i,m] - 1 )^2                      (one-hot)
         + B * sum_{(i,j) in pairs} sum_{m,m' : conflict} x[i,m] x[j,m']
                                                                 (separation)

- cost[i,m] = 0 for the nominal manoeuvre, 1 otherwise, so the objective counts
  the number of manoeuvring aircraft (David Rey's active objective).
- A enforces exactly one manoeuvre per aircraft.
- B penalises every ordered manoeuvre combination that leaves a pair in conflict,
  using the SAME analytic conflict test as the classical baseline.

Penalty weights A and B are set strictly larger than the maximum objective range
(= n) so that the global QUBO optimum is always a feasible, conflict-free, one-
hot assignment whenever one exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from instance import Instance
from manoeuvres import Manoeuvre


@dataclass
class QUBO:
    Q: np.ndarray            # symmetric matrix, shape (V, V)
    offset: float            # constant energy offset
    instance: Instance
    manoeuvres: list[Manoeuvre]
    A: float
    B: float
    C: float

    @property
    def num_vars(self) -> int:
        return self.Q.shape[0]

    @property
    def M(self) -> int:
        return len(self.manoeuvres)

    def var_index(self, i: int, m: int) -> int:
        return i * self.M + m

    def energy(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x + self.offset)

    def decode(self, x: np.ndarray) -> list[list[int]]:
        """Return, per aircraft, the list of selected manoeuvre indices."""
        x = np.asarray(x).reshape(self.instance.n, self.M)
        return [list(np.nonzero(row)[0]) for row in x]


def build_qubo(instance: Instance, manoeuvres: list[Manoeuvre],
               A: float | None = None, B: float | None = None,
               C: float = 1.0) -> QUBO:
    n = instance.n
    M = len(manoeuvres)
    V = n * M
    if A is None:
        A = float(n + 1) * C * 2.0
    if B is None:
        B = float(n + 1) * C * 2.0

    Q = np.zeros((V, V), dtype=float)
    offset = 0.0

    def idx(i: int, m: int) -> int:
        return i * M + m

    # --- Objective: unit cost per non-nominal manoeuvre (diagonal). ---
    for i in range(n):
        for m, man in enumerate(manoeuvres):
            cost = 0.0 if man.is_nominal else 1.0
            Q[idx(i, m), idx(i, m)] += C * cost

    # --- One-hot constraint: A * (sum_m x[i,m] - 1)^2 per aircraft. ---
    # Expanding with x^2 = x for binary variables:
    #   A * ( -x[i,m] summed ) + A * 2 * sum_{m<m'} x[i,m] x[i,m'] + A * 1
    for i in range(n):
        for m in range(M):
            Q[idx(i, m), idx(i, m)] += -A  # linear part: A*(1 - 2) on diagonal
        for m in range(M):
            for mp in range(m + 1, M):
                _add_quadratic(Q, idx(i, m), idx(i, mp), 2.0 * A)
        offset += A

    # --- Separation constraint: penalise conflicting manoeuvre combinations. ---
    # Precompute the velocity of each (aircraft, manoeuvre).
    vel = [[instance.velocity(i, man.q, man.dtheta) for man in manoeuvres]
           for i in range(n)]
    for (i, j) in instance.pairs:
        for m in range(M):
            for mp in range(M):
                if instance.in_conflict(i, vel[i][m], j, vel[j][mp]):
                    _add_quadratic(Q, idx(i, m), idx(j, mp), B)

    return QUBO(Q=Q, offset=offset, instance=instance, manoeuvres=manoeuvres,
               A=A, B=B, C=C)


def _add_quadratic(Q: np.ndarray, a: int, b: int, value: float) -> None:
    """Add a symmetric off-diagonal coupling value split across (a,b) and (b,a)."""
    Q[a, b] += value / 2.0
    Q[b, a] += value / 2.0
