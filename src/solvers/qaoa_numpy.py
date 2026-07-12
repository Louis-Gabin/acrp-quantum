"""Dependency-free QAOA statevector simulator for the ACRP QUBO.

This is a self-contained reference implementation of QAOA written in numpy only.
It requires NO qiskit, NO scipy, and runs immediately. It serves three purposes:

1. Let the quantum algorithm be executed and validated even without a quantum
   SDK installed (useful offline and as a teaching/debug reference).
2. Provide a ground-truth cross-check for the Qiskit implementation
   (`qaoa_qiskit.py`): both must agree on the expected energy and on the sampled
   optimum for the same (gamma, beta), which strengthens reproducibility.
3. Report the QAOA success probability (probability of sampling the optimum),
   the metric highlighted in the literature review.

It is exact (full statevector), so it is limited to small variable counts
(V <= ~18 comfortably). Beyond that, use the Qiskit sampler or restrict the
manoeuvre set.

The cost operator is diagonal (the QUBO energy), and the transverse-field mixer
applies the same single-qubit rotation to every qubit, so the implementation is
independent of qubit/bit ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QAOAResult:
    best_x: np.ndarray          # most probable bitstring (as a 0/1 float vector)
    best_energy: float          # QUBO energy of best_x
    expected_energy: float      # <psi|C|psi> at the optimised parameters
    success_probability: float  # probability mass on the true optimum
    params: np.ndarray          # optimised [gammas..., betas...]
    probs: np.ndarray           # full probability distribution over 2^V states


def _energy_spectrum(Q: np.ndarray, offset: float) -> np.ndarray:
    """Return the length-2^V vector of QUBO energies over all bitstrings.

    Bit k of basis index n is (n >> k) & 1, matching the flattened variable
    index used by the QUBO (variable k = aircraft i, manoeuvre m with k=i*M+m).
    """
    v = Q.shape[0]
    N = 1 << v
    bits = ((np.arange(N)[:, None] >> np.arange(v)[None, :]) & 1).astype(float)
    Qs = 0.5 * (Q + Q.T)
    energies = np.einsum("ni,ij,nj->n", bits, Qs, bits) + offset
    return energies


def _apply_transverse_mixer(psi: np.ndarray, beta: float, v: int) -> np.ndarray:
    """Apply exp(-i beta sum_k X_k) to the statevector (reshaped to (2,)*v)."""
    c = np.cos(beta)
    s = -1j * np.sin(beta)
    gate = np.array([[c, s], [s, c]], dtype=complex)
    tensor = psi.reshape((2,) * v)
    for k in range(v):
        tensor = np.tensordot(gate, tensor, axes=([1], [k]))
        tensor = np.moveaxis(tensor, 0, k)
    return tensor.reshape(-1)


def expectation(energies: np.ndarray, gammas, betas):
    """Return (expected_energy, probability_distribution) for given parameters."""
    N = energies.shape[0]
    v = N.bit_length() - 1
    psi = np.full(N, 1.0 / np.sqrt(N), dtype=complex)
    phase = np.exp(-1j * np.outer(np.asarray(gammas), energies))  # per-layer phases
    for layer, beta in enumerate(betas):
        psi = psi * phase[layer]
        psi = _apply_transverse_mixer(psi, beta, v)
    probs = np.abs(psi) ** 2
    return float(np.sum(probs * energies)), probs


def solve_qaoa_numpy(qubo, *, p: int = 1, restarts: int = 200,
                     refine_steps: int = 60, seed: int = 0) -> QAOAResult:
    """Optimise a p-layer QAOA on the QUBO and sample the optimised circuit.

    Uses a dependency-free optimiser: random restarts followed by coordinate
    descent with a shrinking step. Adequate for the small p (1-3) relevant here.
    """
    energies = _energy_spectrum(qubo.Q, qubo.offset)
    true_opt = float(np.min(energies))
    rng = np.random.default_rng(seed)

    def objective(params: np.ndarray) -> float:
        gammas = params[:p]
        betas = params[p:]
        return expectation(energies, gammas, betas)[0]

    best_params = None
    best_val = float("inf")
    for _ in range(restarts):
        gammas = rng.uniform(0.0, 2.0 * np.pi, size=p)
        betas = rng.uniform(0.0, np.pi, size=p)
        params = np.concatenate([gammas, betas])
        val = objective(params)
        if val < best_val:
            best_val = val
            best_params = params

    step = 0.3
    for _ in range(refine_steps):
        improved = False
        for i in range(2 * p):
            for delta in (step, -step):
                trial = best_params.copy()
                trial[i] += delta
                val = objective(trial)
                if val < best_val:
                    best_val = val
                    best_params = trial
                    improved = True
        if not improved:
            step *= 0.5

    exp_energy, probs = expectation(energies, best_params[:p], best_params[p:])
    n_best = int(np.argmax(probs))
    v = qubo.num_vars
    best_x = np.array([(n_best >> k) & 1 for k in range(v)], dtype=float)
    success_prob = float(np.sum(probs[np.isclose(energies, true_opt)]))
    return QAOAResult(best_x=best_x, best_energy=float(energies[n_best]),
                      expected_energy=exp_energy, success_probability=success_prob,
                      params=best_params, probs=probs)
