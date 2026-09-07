# ACRP-Quantum: Benchmarking quantum and classical solvers for the tactical 2D Aircraft Conflict Resolution Problem

Master thesis research codebase, SKEMA Business School, MSc Artificial
Intelligence for Business Transformation. Supervisor: Professor David Rey.

This repository contains the modelling, solver, analysis and IBM Quantum
experiments used to benchmark classical and quantum approaches to the tactical
two-dimensional Aircraft Conflict Resolution Problem (ACRP). The thesis
manuscript and its LaTeX source are maintained separately and are not part of
this repository.

## Research objective

The project compares:

1. an exact classical baseline implemented with Gurobi;
2. an exact brute-force QUBO solver for small instances;
3. simulated annealing on the same QUBO;
4. the Quantum Approximate Optimisation Algorithm (QAOA), evaluated with exact
   statevector simulation and IBM quantum hardware.

All solvers are evaluated on the same underlying ACRP instances. The QUBO
solver, simulated annealing and QAOA share the same binary formulation and the
same decoding and evaluation functions.

The repository addresses three methodological requirements:

| Requirement | Implementation |
| --- | --- |
| Problem fidelity | The QUBO represents tactical 2D heading and speed decisions rather than a delay-only or generic routing surrogate. |
| Benchmarking rigour | Exact Gurobi and brute-force results provide reference objective values for the tested instances. |
| Reproducibility | Seeds, manoeuvre sets, penalty weights, frozen QAOA parameters, job identifiers and raw result files are retained. |

The scale gap is deliberate. Exact statevector simulation and exhaustive QUBO
validation restrict the quantum comparison to small instances, while the
classical baseline is evaluated on larger cases.

## Problem formulation

Each aircraft `i` selects exactly one manoeuvre `m`, defined by a speed
multiplier `q` and a heading deviation `dtheta`, through a binary variable
`x[i,m]`. The QUBO energy contains:

1. a unit objective cost for each non-nominal manoeuvre;
2. a one-hot penalty `A` enforcing one manoeuvre per aircraft;
3. a separation penalty `B` for each pair of manoeuvres that leaves two
   aircraft in conflict.

The conflict test uses the same analytic relative-motion geometry as the
classical baseline. Decoded samples are evaluated independently by
`src/evaluate.py`. See `GLOSSAIRE.md` for the notation.

## Repository layout

```text
acrp-quantum/
  README.md
  GLOSSAIRE.md
  SETUP_MAC.md
  requirements.txt
  run_m3.sh
  run_m5.sh
  run_overnight_m3.sh
  src/
    instance.py
    manoeuvres.py
    qubo.py
    evaluate.py
    run_demo.py
    run_qaoa.py
    run_benchmark.py
    solvers/
      exact_bruteforce.py
      exact_gurobi.py
      qaoa_numpy.py
      qaoa_qiskit.py
      simulated_annealing.py
  baseline_gurobi/
    README.md
    ACRP_gurobi.ipynb
    Instance.py
  analysis/
    _common.py
    benchmark.py
    qubit_budget.py
    scaling.py
    scaling_large.py
    stats_tests.py
    sweep_depth.py
    sweep_penalty.py
    compare_ibm_backends.py
    run_ibm_hardware.py
    run_ibm_depth_series.py
    run_ibm_replication_control.py
    verify_ibm_run.py
    audit_depth_series.py
    DEPTH_SERIES_INSTRUCTIONS.md
    REPLICATION_CONTROL_INSTRUCTIONS.md
  results/
    ibm_backend_comparison_cp3.json
    ibm_hardware_cp3.json
    ibm_hardware_cp3_p*.json
    ibm_depth_series_plan_cp3_*.json
    ibm_depth_series_submission_cp3_*.json
    ibm_depth_series_cp3_*.json
    ibm_replication_control_plan_cp3_*.json
    ibm_replication_control_submission_cp3_*.json
    ibm_replication_control_results_cp3_*.json
```

Generated figures and benchmark tables are also stored in `results/`.

## Installation

Python 3.12 is recommended. Create an isolated environment before installing
the project dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The Gurobi baseline requires Gurobi 12 or later and a valid licence because the
model uses `gurobipy.nlfunc.cos` and `gurobipy.nlfunc.sin`.

IBM hardware execution requires a compatible Qiskit 2.x environment and IBM
Quantum Runtime access. Credentials must remain in local secure storage or
environment variables and must never be committed to this repository.

## Offline validation

The core validation does not require quantum packages or network access.

```bash
python3 src/run_demo.py
```

This command builds small Circle Problem instances, solves their QUBOs exactly
and with simulated annealing, checks feasibility and objective quality, and
saves a trajectory figure in `results/`.

The broader offline analyses are implemented in:

- `analysis/benchmark.py` for solver comparisons;
- `analysis/sweep_depth.py` for ideal QAOA depth experiments;
- `analysis/sweep_penalty.py` for penalty sensitivity;
- `analysis/stats_tests.py` for statistical comparisons;
- `analysis/qubit_budget.py`, `analysis/scaling.py` and
  `analysis/scaling_large.py` for width and scaling estimates.

## IBM Quantum workflow

Hardware execution is intentionally separated into backend selection, frozen
parameter preparation, submission and verification. Do not rerun a hardware
campaign merely to reproduce a table. The committed JSON files are the
authoritative record of the completed jobs.

### 1. Backend comparison

```bash
python3 -u analysis/compare_ibm_backends.py \
  --n 3 \
  --heading 3 \
  --p 1 \
  --seed 1
```

The comparison records routed ISA depth, two-qubit gate count, physical layout
and backend properties for accessible processors. Its output is stored in
`results/ibm_backend_comparison_cp3.json`.

### 2. Initial CP3 hardware run

`analysis/run_ibm_hardware.py` executes the nine-qubit CP3 circuit and stores
the decoded measurement distribution. The completed run is retained in:

- `results/ibm_hardware_cp3.json`;
- `results/ibm_hardware_cp3_p1_20260827T213336346638Z.json`.

For CP3 with three manoeuvres per aircraft, the binary basis contains 512
states, of which eight are feasible and six are optimal. Uniform sampling
therefore gives a feasibility probability of 1.5625 per cent and an optimum
probability of 1.171875 per cent.

The initial hardware experiment produced 13.3057 per cent feasibility and
6.5674 per cent optimum probability. The matched ideal calculation produced
15.4510 per cent and 6.7695 per cent, respectively. This confirms the
end-to-end implementation for one small circuit. It is not evidence of
scalability or quantum advantage.

### 3. QAOA depth series

The depth campaign evaluates `p` in `{1, 2, 3, 4, 6}` on `ibm_kingston`, with
4096 shots per circuit. The completed IBM job is
`da8aqcse74ec73aie27g`.

| `p` | Two-qubit gates | Ideal optimum probability | Hardware optimum probability | Hardware feasibility |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 54 | 9.45% | 7.54% | 10.06% |
| 2 | 139 | 21.88% | 10.25% | 12.74% |
| 3 | 241 | 21.88% | 5.52% | 7.32% |
| 4 | 317 | 27.41% | 8.69% | 14.04% |
| 6 | 513 | 48.04% | 8.69% | 10.35% |

The ideal improvement at greater depth did not transfer to the hardware in
this campaign. Depth two was the best observed depth for optimum probability,
but the experiment does not establish a universal optimal hardware depth.

The submitted depth-three solution contains a numerically negligible third
layer. It is retained as a null-layer control rather than interpreted as the
best attainable depth-three solution. Relative to depth two, it adds 102
two-qubit gates while leaving the ideal optimum probability unchanged and
coincides with a hardware decline from 10.25 to 5.52 per cent.

The frozen plan, submission manifest and decoded result are stored in:

- `results/ibm_depth_series_plan_cp3_20260827T172136847500Z.json`;
- `results/ibm_depth_series_submission_cp3_20260827T213336346638Z.json`;
- `results/ibm_depth_series_cp3_20260827T213336346638Z.json`.

See `analysis/DEPTH_SERIES_INSTRUCTIONS.md` and
`analysis/audit_depth_series.py` for the protocol and verification procedure.

### 4. Interleaved replication and drift control

The control campaign separates circuit depth more carefully from execution
time by using the interleaved order:

```text
p2_A, p6, p3_true, p2_B, p4, p3_null, p1, p2_C
```

Each circuit used 4096 shots. Depth two was repeated at the beginning, middle
and end of the job, while `p3_true` and `p3_null` distinguish a newly optimised
depth-three circuit from the earlier null-layer control. Dynamical decoupling,
gate twirling and measurement twirling were disabled.

The campaign ran on `ibm_kingston` as job `daa99rhqtnsc73d2d1l0` and consumed
11.0 reported quantum-seconds. All six pre-submission validation groups passed,
with maximum statevector differences below `1.2e-15`.

The final depth-two drift control, `p2_C`, produced 14.0625 per cent feasibility
and 11.7432 per cent optimum probability over 4096 shots. These values describe
one circuit position within the interleaved campaign and should not be used
alone to infer the presence or absence of temporal drift. The complete
per-circuit results and confidence intervals are retained in the result file.

The traceability chain is:

- `results/ibm_replication_control_plan_cp3_20260830T201007516547Z.json`;
- `results/ibm_replication_control_submission_cp3_20260830T203911222372Z.json`;
- `results/ibm_replication_control_results_cp3_20260830T203911222372Z.json`.

See `analysis/REPLICATION_CONTROL_INSTRUCTIONS.md` for the frozen protocol.

## Reproducibility conventions

- Circle instances used in the reported experiments pass `d=8.0` explicitly.
  The default value in the generator must not be relied upon.
- Variables follow `k = i*M + m`, where `i` is the aircraft index and `m` is
  the manoeuvre index.
- Qiskit bitstrings are reversed before conversion to the binary decision
  vector.
- Stochastic solvers take an explicit seed.
- Penalty weights `A` and `B` default to `2*(n+1)`, strictly above the objective
  range `n`. Any override must be reported.
- Hardware parameters are selected using expected QUBO energy, not optimum-hit
  probability.
- Hardware plans are frozen before submission and identified by SHA-256
  digests.
- Plan, submission and result files are kept separately so that optimisation,
  execution and decoding can be audited.
- Job identifiers and shot counts are reported with the hardware outputs.

## Interpretation limits

- Hardware results concern CP3, one device and specific calibration periods.
- Wilson intervals quantify finite-shot uncertainty but not calibration drift.
- The first depth campaign used a nested warm-start chain, so its circuits do
  not necessarily represent the best attainable parameters at every depth.
- No error mitigation was applied in the reported depth campaign.
- Hardware frequencies are not expected to reproduce identically on a later
  calibration.
- Successful hardware execution is not evidence of quantum advantage.
- The exact classical solver remains the relevant performance baseline for the
  tested ACRP instances.

## Data and credential policy

Raw IBM result JSON files, frozen plans, submission manifests and verification
outputs are versioned because they are part of the scientific evidence chain.
IBM API keys, account files, CRNs and local environment configuration are not
research outputs and must never be committed.

Before committing new hardware artefacts, verify that no credentials are
present and run:

```bash
git diff --check
git status --short
```

## Project status

The repository records the completed classical experiments, ideal QAOA
analyses, initial IBM hardware run, five-depth IBM campaign and interleaved
replication-control campaign. The codebase is intended as a transparent
research artefact rather than an operational air-traffic decision system.
