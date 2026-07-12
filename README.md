# ACRP-Quantum — Benchmarking quantum and classical solvers for the tactical 2D Aircraft Conflict Resolution Problem

Master thesis codebase (SKEMA Business School, MSc AI). Supervisor: Pr. David Rey.

This repository implements the **methodology chapter** of the thesis: a controlled,
reproducible benchmark that compares a structured classical baseline, a classical
metaheuristic, and a quantum variational algorithm **on exactly the same problem
instances and the same QUBO formulation** of the tactical two-dimensional Aircraft
Conflict Resolution Problem (ACRP).

The design directly targets three of the four research gaps identified in the
literature review (Chapter 1):

| Gap | How this codebase addresses it |
| --- | --- |
| Problem-fidelity | The QUBO encodes the tactical 2D heading/speed ACRP, the same problem solved by the classical baseline, not a delay-only or UAV variant. |
| Benchmarking-rigour | Every solver runs on the same instances; the classical exact baseline (Gurobi) and an exact brute-force QUBO solver provide ground truth. |
| Reproducibility | Fixed seeds, documented penalty weights, versioned instances, and a single command to reproduce every reported number. |

The scale gap is deliberately accepted: instances are kept small enough for
faithful statevector simulation and exact validation.

## Repository layout

```
acrp-quantum/
  README.md                 # this file
  GLOSSAIRE.md              # notation lexicon (symbols, variables, penalties)
  requirements.txt          # Python dependencies for the full pipeline
  src/
    instance.py             # ACRP instance model + analytic conflict geometry
    manoeuvres.py           # discrete manoeuvre sets (speed x heading grid)
    qubo.py                 # QUBO encoding (objective + one-hot + separation)
    evaluate.py             # decode a bitstring -> feasibility + quality
    run_demo.py             # end-to-end validation + figure (RUNS OFFLINE)
    solvers/
      exact_bruteforce.py   # exact QUBO optimum (small V) = ground truth
      simulated_annealing.py# fair classical twin, same QUBO as QAOA
      qaoa_qiskit.py        # QAOA on Qiskit Aer (run on your machine)
  baseline_gurobi/          # David Rey's simplified exact MINLP baseline
  results/                  # generated figures / CSVs
```

## Formulation in one paragraph

Each aircraft `i` selects exactly one manoeuvre `m` (a speed multiplier `q` and a
heading deviation `dtheta`) via a binary variable `x[i,m]`. The QUBO energy sums
(1) a unit cost per non-nominal manoeuvre (so minimising energy minimises the
number of manoeuvring aircraft, David Rey's active objective), (2) a one-hot
penalty `A` enforcing one manoeuvre per aircraft, and (3) a separation penalty
`B` on every manoeuvre pair that leaves two aircraft in conflict, using the
same analytic conflict criterion as the classical baseline. See `GLOSSAIRE.md`.

## How to run

### Offline core (no quantum dependencies) — validated

```bash
python3 src/run_demo.py
```

Builds the Circle Problem instances CP_3..CP_5, solves each QUBO with the exact
brute-force solver (ground truth) and with simulated annealing, checks that both
recover a feasible, conflict-free, minimum-manoeuvre assignment, and saves a
trajectory figure to `results/`.

Expected result: on CP_3 and CP_4 the simulated annealing energy matches the
exact QUBO optimum (gap 0); CP_5 returns a feasible solution.

### Quantum campaign (requires qiskit)

```bash
pip install -r requirements.txt
# then call solvers.qaoa_qiskit.solve_qaoa(qubo, p=2) on a built QUBO
```

`qaoa_qiskit.py` reuses the exact same `QUBO` object, converts it to an Ising
cost Hamiltonian, builds a p-layer QAOA ansatz, optimises the parameters on the
Aer statevector simulator (noiseless first, then a noisy backend), and returns a
bitstring scored by the same `evaluate.py` used for every other solver.

### Classical exact baseline (Gurobi)

See `baseline_gurobi/README.md`. Requires Gurobi >= 12 (academic licence) because
the model uses `nlfunc.cos/sin`.

## Reproducibility notes

- All stochastic solvers take an explicit `seed`.
- Penalty weights `A`, `B` default to `2*(n+1)` (strictly above the objective
  range `n`), which guarantees the QUBO optimum is feasible when a conflict-free
  assignment exists. Any override must be reported.
- Instances are generated deterministically from `(n, radius, d, speed)`.
