"""ACRP instance model and conflict geometry.

This module defines the tactical 2D Aircraft Conflict Resolution Problem (ACRP)
instance used throughout the thesis benchmark. It is deliberately kept faithful
to the analytic separation geometry implemented in David Rey's baseline solver
(`baseline_gurobi/Instance.py`), so that the QUBO formulation built on top of it
resolves *exactly the same* problem as the classical exact baseline.

Conventions (see GLOSSAIRE.md for the full notation lexicon):
- Aircraft are indexed 0..n-1 internally (David's code uses 1..n; the mapping is
  documented in the glossary).
- Each aircraft i has an initial position (x0[i], y0[i]), an initial heading
  theta0[i] in radians, and a nominal speed v0[i].
- A manoeuvre applied to aircraft i is a pair (q, dtheta): a speed multiplier q
  and a heading deviation dtheta (radians). The resulting velocity vector is
  V_i = q * v0[i] at heading (theta0[i] + dtheta).
- A pair (i, j) is in conflict if their relative trajectory brings them closer
  than the separation distance d at some strictly future time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Control bounds, identical to David Rey's simplified baseline.
Q_MIN = 0.94
Q_MAX = 1.03
H_MIN = -math.pi / 6.0  # -30 degrees
H_MAX = math.pi / 6.0   # +30 degrees


@dataclass
class Instance:
    """A tactical 2D ACRP instance.

    Attributes
    ----------
    n : number of aircraft.
    d : minimum horizontal separation distance.
    x0, y0 : initial positions (numpy arrays of length n).
    theta0 : initial headings in radians (numpy array of length n).
    v0 : nominal speeds (numpy array of length n).
    name : human-readable instance label.
    """

    n: int
    d: float
    x0: np.ndarray
    y0: np.ndarray
    theta0: np.ndarray
    v0: np.ndarray
    name: str = "instance"
    pairs: list[tuple[int, int]] = field(init=False)

    def __post_init__(self) -> None:
        self.x0 = np.asarray(self.x0, dtype=float)
        self.y0 = np.asarray(self.y0, dtype=float)
        self.theta0 = np.asarray(self.theta0, dtype=float)
        self.v0 = np.asarray(self.v0, dtype=float)
        self.pairs = [(i, j) for i in range(self.n) for j in range(self.n) if i < j]

    # ------------------------------------------------------------------ #
    # Instance generators
    # ------------------------------------------------------------------ #
    @classmethod
    def circle_problem(cls, n: int, radius: float = 100.0, d: float = 5.0,
                       speed: float = 1.0) -> "Instance":
        """Generate the canonical Circle Problem (CP_n).

        n aircraft are equally spaced on a circle of the given radius, each
        heading straight towards the centre at a uniform speed. Without any
        manoeuvre every pair converges at the centre simultaneously, so the
        nominal (do-nothing) configuration is maximally infeasible. This is the
        standard stress-test instance of the ACRP literature.
        """
        idx = np.arange(n)
        angle = idx * 2.0 * math.pi / n + math.pi
        x0 = -radius * np.cos(angle)
        y0 = -radius * np.sin(angle)
        # Heading towards the centre (0, 0).
        theta0 = np.arctan2(-y0, -x0)
        # Normalise to (-pi, pi].
        theta0 = np.arctan2(np.sin(theta0), np.cos(theta0))
        v0 = np.full(n, float(speed))
        return cls(n=n, d=d, x0=x0, y0=y0, theta0=theta0, v0=v0, name=f"CP_{n}")

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    def velocity(self, i: int, q: float, dtheta: float) -> tuple[float, float]:
        """Return the velocity vector (vx, vy) of aircraft i under manoeuvre (q, dtheta)."""
        heading = self.theta0[i] + dtheta
        speed = q * self.v0[i]
        return speed * math.cos(heading), speed * math.sin(heading)

    def in_conflict(self, i: int, vi: tuple[float, float],
                    j: int, vj: tuple[float, float]) -> bool:
        """Analytic pairwise conflict test (identical criterion to David Rey's code).

        A conflict exists iff the minimum future separation drops below d while
        the two aircraft are still approaching (closing) each other.
        """
        xr0 = self.x0[i] - self.x0[j]
        yr0 = self.y0[i] - self.y0[j]
        vrx = vi[0] - vj[0]
        vry = vi[1] - vj[1]
        rel_speed_sq = vrx * vrx + vry * vry
        if rel_speed_sq < 1e-12:
            # No relative motion: distance never decreases.
            return False
        sep0 = ((yr0 ** 2 - self.d ** 2) * vrx ** 2
                + (xr0 ** 2 - self.d ** 2) * vry ** 2
                - 2.0 * xr0 * yr0 * vrx * vry)
        tm0 = -(xr0 * vrx + yr0 * vry) / rel_speed_sq
        return sep0 < 0.0 and tm0 > 0.0

    def nominal_conflicts(self) -> list[tuple[int, int]]:
        """Return the pairs in conflict when no aircraft manoeuvres."""
        conflicts = []
        for (i, j) in self.pairs:
            vi = self.velocity(i, 1.0, 0.0)
            vj = self.velocity(j, 1.0, 0.0)
            if self.in_conflict(i, vi, j, vj):
                conflicts.append((i, j))
        return conflicts
