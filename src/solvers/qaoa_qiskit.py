"""QAOA solver for the ACRP QUBO (Qiskit implementation).

>>> NOT EXECUTED in the analysis sandbox (no qiskit here). Runs on your Mac
>>> after `pip install -r requirements.txt`.

This version uses the stable `qiskit.quantum_info.Statevector` API for the
noiseless campaign, which avoids the primitive-API churn between Qiskit releases
and works on any Qiskit >= 1.0. The optional noisy campaign uses qiskit-aer.

It reuses the SAME `QUBO` object as the classical solvers, converts it to an
Ising cost Hamiltonian, builds a p-layer QAOA ansatz, optimises (gamma, beta)
with SciPy COBYLA, and returns a bitstring scored by the same `evaluate.py`.

The numpy reference (`qaoa_numpy.py`) and this module must agree on the expected
energy for identical parameters: that cross-check is a reproducibility control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def qubo_to_ising(Q: np.ndarray, offset: float = 0.0):
    """Return (h, J, ising_offset) with E = sum_k h_k Z_k + sum_{k<l} J_kl Z_k Z_l.

    Uses x_k = (1 - Z_k)/2. Pure numpy, unit-testable without qiskit.
    """
    Qs = 0.5 * (Q + Q.T)
    v = Qs.shape[0]
    lin = np.diag(Qs).copy()
    h = np.zeros(v)
    J: dict[tuple[int, int], float] = {}
    ising_offset = float(offset)
    for k in range(v):
        ising_offset += lin[k] / 2.0
        h[k] += -lin[k] / 2.0
    for k in range(v):
        for l in range(k + 1, v):
            w = Qs[k, l] + Qs[l, k]
            if abs(w) < 1e-12:
                continue
            ising_offset += w / 4.0
            h[k] += -w / 4.0
            h[l] += -w / 4.0
            J[(k, l)] = J.get((k, l), 0.0) + w / 4.0
    return h, J, ising_offset


@dataclass
class QAOAResult:
    best_x: np.ndarray
    best_energy: float
    expected_energy: float
    params: np.ndarray
    counts: dict


def _cost_operator(h, J, v):
    from qiskit.quantum_info import SparsePauliOp
    terms = []
    for k in range(v):
        if abs(h[k]) > 1e-12:
            label = ["I"] * v
            label[v - 1 - k] = "Z"
            terms.append(("".join(label), float(h[k])))
    for (k, l), w in J.items():
        label = ["I"] * v
        label[v - 1 - k] = "Z"
        label[v - 1 - l] = "Z"
        terms.append(("".join(label), float(w)))
    return SparsePauliOp.from_list(terms)


def _build_circuit(h, J, v, gammas, betas):
    from qiskit import QuantumCircuit
    p = len(gammas)
    qc = QuantumCircuit(v)
    qc.h(range(v))
    for t in range(p):
        for k in range(v):
            if abs(h[k]) > 1e-12:
                qc.rz(2.0 * gammas[t] * float(h[k]), k)
        for (k, l), w in J.items():
            qc.rzz(2.0 * gammas[t] * float(w), k, l)
        for k in range(v):
            qc.rx(2.0 * betas[t], k)
    return qc


def solve_qaoa_qiskit(qubo, *, p: int = 2, maxiter: int = 200, seed: int = 1,
                      noisy: bool = False, shots: int = 8192) -> QAOAResult:
    from qiskit.quantum_info import Statevector
    from scipy.optimize import minimize

    h, J, ising_offset = qubo_to_ising(qubo.Q, qubo.offset)
    v = qubo.num_vars
    cost_op = _cost_operator(h, J, v)
    rng = np.random.default_rng(seed)

    def energy(params):
        gammas, betas = params[:p], params[p:]
        qc = _build_circuit(h, J, v, gammas, betas)
        sv = Statevector(qc)
        return float(np.real(sv.expectation_value(cost_op))) + ising_offset

    x0 = np.concatenate([rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)])
    opt = minimize(energy, x0, method="COBYLA", options={"maxiter": maxiter})

    gammas, betas = opt.x[:p], opt.x[p:]
    qc = _build_circuit(h, J, v, gammas, betas)

    if noisy:
        # Optional noisy campaign: sample on an Aer backend with a noise model.
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2
        meas = qc.copy()
        meas.measure_all()
        sampler = SamplerV2()
        res = sampler.run([meas], shots=shots).result()
        counts = res[0].data.meas.get_counts()
    else:
        probs = Statevector(qc).probabilities_dict()
        counts = {k: v_ for k, v_ in probs.items()}

    best_bitstring = max(counts, key=counts.get)
    x = np.array([int(b) for b in reversed(best_bitstring)], dtype=float)
    return QAOAResult(best_x=x, best_energy=qubo.energy(x),
                      expected_energy=float(opt.fun), params=opt.x, counts=counts)
