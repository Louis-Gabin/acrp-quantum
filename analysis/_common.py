"""Shared helpers for the analysis scripts (path setup + basis utilities)."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "solvers")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402


def basis_bits(v: int) -> np.ndarray:
    """Return the (2^v, v) matrix of bits, bit k of state n = (n >> k) & 1."""
    N = 1 << v
    return ((np.arange(N)[:, None] >> np.arange(v)[None, :]) & 1).astype(int)


def one_hot_mass(probs: np.ndarray, n: int, M: int) -> float:
    """Probability mass on states that satisfy the one-hot constraint exactly."""
    v = n * M
    bits = basis_bits(v).reshape(-1, n, M)
    ok = np.all(bits.sum(axis=2) == 1, axis=1)
    return float(np.sum(probs[ok]))


def results_dir() -> str:
    d = os.path.join(_ROOT, "results")
    os.makedirs(d, exist_ok=True)
    return d
