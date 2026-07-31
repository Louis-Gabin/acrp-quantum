"""Exact classical baseline on the DISCRETISED counting model (Gurobi ILP).

This solver optimises exactly the deterministic optimisation model presented in
the thesis (Section 2.2, 'Deterministic optimisation model'), i.e. the binary
linear programme that the QUBO of `qubo.py` encodes as penalties:

    min   sum_{i,m} c[i,m] * x[i,m]                      (count of manoeuvres)
    s.t.  sum_m x[i,m] = 1                     for every aircraft i   (one-hot)
          x[i,m] + x[j,m'] <= 1   for every conflicting pair of choices
          x[i,m] in {0,1}

with c[i,m] = 0 for the nominal manoeuvre and 1 otherwise, so the objective
counts the number of manoeuvring aircraft (David Rey's active objective).

Why this module exists
----------------------
The brute-force solver enumerates 2^V states and is limited to V <= ~24, so it
cannot provide a ground-truth optimum beyond about 5 aircraft. Gurobi solves the
SAME discretised model as an integer programme and scales to the large
instances (n = 10, 12, 25, 50) that cannot be simulated on the quantum side.
It therefore provides both (a) the exact optimum used to score the heuristics
and (b) the classical solve-time that documents how the problem scales.

The conflict set is built with the SAME analytic `in_conflict` test as the
QUBO, so this ILP resolves exactly the same problem as the quantum formulation.

Requires: gurobipy >= 12 with a valid (academic) licence. The import is lazy, so
this module can be imported even when Gurobi is absent; the error is raised only
when `solve_gurobi_exact` is actually called.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from instance import Instance
from manoeuvres import Manoeuvre


@dataclass
class GurobiResult:
    assignment: list[int]          # manoeuvre index chosen per aircraft (-1 if none)
    num_manoeuvres: int            # objective value = number of manoeuvring aircraft
    objective: float               # Gurobi ObjVal (== num_manoeuvres here)
    feasible: bool                 # a solution was found
    status: str                    # 'optimal', 'time_limit', 'infeasible', ...
    mip_gap: float                 # final relative MIP gap (0.0 if proven optimal)
    seconds: float                 # wall-clock solve time (model build + optimise)
    num_binary_vars: int           # n * M
    num_conflict_constraints: int  # number of pairwise no-conflict constraints


def _conflict_choice_pairs(instance: Instance,
                           manoeuvres: list[Manoeuvre]) -> list[tuple[int, int, int, int]]:
    """Return every (i, m, j, m') whose joint choice leaves i and j in conflict."""
    M = len(manoeuvres)
    vel = [[instance.velocity(i, man.q, man.dtheta) for man in manoeuvres]
           for i in range(instance.n)]
    forbidden: list[tuple[int, int, int, int]] = []
    for (i, j) in instance.pairs:
        for m in range(M):
            for mp in range(M):
                if instance.in_conflict(i, vel[i][m], j, vel[j][mp]):
                    forbidden.append((i, m, j, mp))
    return forbidden


def solve_gurobi_exact(instance: Instance, manoeuvres: list[Manoeuvre], *,
                       time_limit: float | None = None,
                       threads: int | None = None,
                       mip_gap: float = 0.0,
                       verbose: bool = False) -> GurobiResult:
    """Solve the discretised counting model to (proven) optimality with Gurobi.

    Parameters
    ----------
    time_limit : optional wall-clock cap in seconds (Gurobi returns the best
        incumbent found so far, with a non-zero mip_gap, if it is hit).
    threads : optional cap on solver threads (leave None to let Gurobi decide;
        set e.g. 4 on an 8 GB laptop to keep memory in check).
    mip_gap : target relative optimality gap (0.0 = prove optimality).
    """
    try:
        from gurobipy import Model, GRB, quicksum
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "gurobipy is not available. Install Gurobi >= 12 with an academic "
            "licence (see SETUP_MAC.md) to run the exact ILP baseline."
        ) from exc

    n = instance.n
    M = len(manoeuvres)
    cost = [0.0 if man.is_nominal else 1.0 for man in manoeuvres]
    forbidden = _conflict_choice_pairs(instance, manoeuvres)

    t0 = time.time()
    model = Model("ACRP_ILP_count")
    model.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if threads is not None:
        model.Params.Threads = int(threads)
    model.Params.MIPGap = float(mip_gap)

    x = model.addVars(n, M, vtype=GRB.BINARY, name="x")
    model.setObjective(
        quicksum(cost[m] * x[i, m] for i in range(n) for m in range(M)),
        GRB.MINIMIZE,
    )
    for i in range(n):
        model.addConstr(quicksum(x[i, m] for m in range(M)) == 1, name=f"onehot_{i}")
    for (i, m, j, mp) in forbidden:
        model.addConstr(x[i, m] + x[j, mp] <= 1, name=f"sep_{i}_{m}_{j}_{mp}")

    model.optimize()
    seconds = time.time() - t0

    status_map = {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.SUBOPTIMAL: "suboptimal",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "inf_or_unbounded",
    }
    status = status_map.get(model.status, f"status_{model.status}")

    if model.SolCount > 0:
        assignment = []
        for i in range(n):
            chosen = 0
            for m in range(M):
                if x[i, m].X > 0.5:
                    chosen = m
                    break
            assignment.append(chosen)
        num_man = sum(1 for i in range(n)
                      if not manoeuvres[assignment[i]].is_nominal)
        objective = float(model.ObjVal)
        try:
            gap = float(model.MIPGap)
        except Exception:
            gap = float("nan")
        feasible = True
    else:
        assignment = [-1] * n
        num_man = -1
        objective = float("nan")
        gap = float("nan")
        feasible = False

    return GurobiResult(
        assignment=assignment, num_manoeuvres=num_man, objective=objective,
        feasible=feasible, status=status, mip_gap=gap, seconds=seconds,
        num_binary_vars=n * M, num_conflict_constraints=len(forbidden),
    )
