# Classical exact baseline (David Rey)

This folder holds the **simplified** exact solver provided by Pr. David Rey to
work on, deliberately stripped of the more complex parameters of his full public
solver (`acrp-lib`, https://github.com/Louis-Gabin/acrp-lib, formulation from
DOI 10.1016/j.ejor.2021.03.059). The files are kept faithful to the version he
gave; only this README documents them.

## Files

- `Instance.py` — instance reader (AMPL `.dat`, static Circle Problem and dynamic
  scenarios) and preprocessing: analytic conflict detection plus the big-M /
  disjunctive convex-region coefficients used to linearise the non-convex
  separation constraints.
- `Optimization.py` — the Gurobi model.
  - `solveMINLP()`: non-convex MINLP (`NonConvex=2`). Binary `f[i]` marks whether
    aircraft i manoeuvres; `q[i]`, `theta[i]` are the speed and heading controls;
    `z[i,j]` selects the active side of each pairwise disjunction. **Active
    objective: minimise the number of manoeuvring aircraft** (`min sum w_i f_i`).
  - `oracle(Fmin)`: robust/dynamic variant over T random scenarios.
- `ACRP_gurobi.ipynb` — driver notebook (loads an instance, preprocesses, solves).

## Requirements

- **Gurobi >= 12** with a valid (academic) licence. The trigonometric constraints
  use `gurobipy.nlfunc.cos/sin`, introduced in Gurobi 12; earlier versions will
  fail to import `nlfunc`.

## Role in the benchmark

This solver is the structured classical ground truth. The QUBO in `../src` encodes
the SAME tactical 2D problem, so the quantum and metaheuristic solvers can be
compared like-for-like against the exact optimum this baseline computes.

## Open points to confirm with David

1. Official objective for the thesis: number of manoeuvres (active here) vs total
   deviation (commented alternative, and the wording used on the GitHub repo).
2. The silent `sqrt_disc = 0` fallback when the discriminant is negative in
   `preprocessing()`.
3. Restricting separation constraints to detected conflict pairs (rather than all
   pairs) to speed up larger instances.
