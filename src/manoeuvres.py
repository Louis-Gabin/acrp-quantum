"""Discrete manoeuvre sets for the ACRP QUBO encoding.

The continuous control variables of the tactical 2D ACRP (speed multiplier q and
heading deviation dtheta) are discretised into a finite set of candidate
manoeuvres. Each aircraft must select exactly one manoeuvre; this one-per-
aircraft rule becomes a one-hot constraint in the QUBO (see qubo.py).

Manoeuvre index 0 is always the nominal (do-nothing) manoeuvre (q=1, dtheta=0),
which carries zero cost in the objective. Every other manoeuvre carries unit
cost, so minimising total cost minimises the number of manoeuvring aircraft,
matching the active objective of David Rey's baseline solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from instance import Q_MIN, Q_MAX, H_MIN, H_MAX


@dataclass(frozen=True)
class Manoeuvre:
    q: float        # speed multiplier
    dtheta: float   # heading deviation (radians)
    label: str

    @property
    def is_nominal(self) -> bool:
        return abs(self.q - 1.0) < 1e-9 and abs(self.dtheta) < 1e-9


def build_manoeuvre_set(n_heading: int = 5, n_speed: int = 1) -> list[Manoeuvre]:
    """Return an ordered manoeuvre set with the nominal manoeuvre first.

    Parameters
    ----------
    n_heading : number of heading options spread over [H_MIN, H_MAX] (odd number
        keeps a 0 deviation option; default 5 gives -30, -15, 0, +15, +30 deg).
    n_speed : number of speed options spread over [Q_MIN, Q_MAX] (default 1 gives
        heading-only control at nominal speed q=1).
    """
    headings = _symmetric_grid(H_MIN, H_MAX, n_heading)
    if n_speed <= 1:
        speeds = [1.0]
    else:
        speeds = _linspace(Q_MIN, Q_MAX, n_speed)
        if 1.0 not in speeds:
            speeds.append(1.0)
        speeds = sorted(set(speeds))

    manoeuvres: list[Manoeuvre] = []
    seen = set()
    # Nominal first.
    nominal = Manoeuvre(1.0, 0.0, "nominal")
    manoeuvres.append(nominal)
    seen.add((round(1.0, 6), round(0.0, 6)))
    for q in speeds:
        for h in headings:
            key = (round(q, 6), round(h, 6))
            if key in seen:
                continue
            seen.add(key)
            label = f"q={q:.2f},dh={math.degrees(h):+.0f}deg"
            manoeuvres.append(Manoeuvre(q, h, label))
    return manoeuvres


def _symmetric_grid(lo: float, hi: float, count: int) -> list[float]:
    if count <= 1:
        return [0.0]
    return _linspace(lo, hi, count)


def _linspace(lo: float, hi: float, count: int) -> list[float]:
    if count == 1:
        return [(lo + hi) / 2.0]
    step = (hi - lo) / (count - 1)
    return [lo + step * k for k in range(count)]
