"""Simulated annealing solver on the QUBO.

This is the *fair classical twin* of the quantum sampler: it operates on exactly
the same QUBO matrix as QAOA, so any performance difference isolates the effect
of the algorithm rather than of the problem formulation. The implementation is a
standard single-spin-flip Metropolis annealer with a geometric temperature
schedule, written in numpy only (no external dependency).

The delta-energy of a single-bit flip is computed incrementally in O(V) using
the local field, which keeps many reads over many sweeps fast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SAResult:
    best_x: np.ndarray
    best_energy: float
    energies: list[float]  # best energy found per read


def simulated_annealing(Q: np.ndarray, offset: float = 0.0, *,
                        num_reads: int = 50, num_sweeps: int = 1000,
                        beta_min: float = 0.1, beta_max: float = 10.0,
                        seed: int | None = None) -> SAResult:
    """Minimise x^T Q x + offset over binary x.

    Parameters
    ----------
    num_reads : number of independent annealing runs (best is returned).
    num_sweeps : temperature steps per read (one candidate flip per variable per
        sweep).
    beta_min, beta_max : inverse-temperature schedule bounds (geometric ramp).
    seed : RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    Qsym = 0.5 * (Q + Q.T)
    v = Qsym.shape[0]
    betas = np.geomspace(beta_min, beta_max, num_sweeps)

    global_best_x = None
    global_best_e = float("inf")
    per_read_best: list[float] = []

    for _ in range(num_reads):
        x = rng.integers(0, 2, size=v).astype(float)
        # Local field h_k = gradient of x^T Q x w.r.t. flipping bit k.
        # Energy contribution change when flipping x_k from s to 1-s:
        #   delta = (1 - 2 x_k) * (2 * (Q x)_k - Q_kk)
        Qx = Qsym @ x
        energy = float(x @ Qx)
        best_e = energy
        best_x = x.copy()
        for beta in betas:
            order = rng.permutation(v)
            for k in order:
                xk = x[k]
                delta = (1.0 - 2.0 * xk) * (2.0 * Qx[k] - Qsym[k, k])
                if delta <= 0.0 or rng.random() < np.exp(-beta * delta):
                    # Accept flip.
                    x[k] = 1.0 - xk
                    Qx += Qsym[:, k] * (x[k] - xk)
                    energy += delta
                    if energy < best_e:
                        best_e = energy
                        best_x = x.copy()
        per_read_best.append(best_e + offset)
        if best_e < global_best_e:
            global_best_e = best_e
            global_best_x = best_x

    return SAResult(best_x=global_best_x, best_energy=global_best_e + offset,
                    energies=per_read_best)
