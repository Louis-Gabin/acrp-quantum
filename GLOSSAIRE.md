# Notation lexicon / Lexique de notation

This file is the single source of truth for the notation used across the code and
the thesis methodology chapter. Symbols are given in English (thesis language),
with a short French gloss.

## Instance parameters

| Symbol | Code | Meaning | Gloss FR |
| --- | --- | --- | --- |
| n | `instance.n` | Number of aircraft | Nombre d'avions |
| d | `instance.d` | Minimum horizontal separation distance | Distance de separation minimale |
| (x0_i, y0_i) | `instance.x0`, `instance.y0` | Initial position of aircraft i | Position initiale |
| theta0_i | `instance.theta0` | Initial heading of aircraft i (radians) | Cap initial |
| v0_i | `instance.v0` | Nominal speed of aircraft i | Vitesse nominale |
| P | `instance.pairs` | Set of aircraft pairs (i, j), i < j | Ensemble des paires |

Indexing: aircraft are 0..n-1 in this codebase. David Rey's baseline uses 1..n;
the two differ only by a +1 offset.

## Control variables and bounds

| Symbol | Code | Meaning | Value |
| --- | --- | --- | --- |
| q_i | manoeuvre `q` | Speed multiplier applied to aircraft i | in [Q_MIN, Q_MAX] |
| dtheta_i | manoeuvre `dtheta` | Heading deviation applied to aircraft i | in [H_MIN, H_MAX] |
| Q_MIN, Q_MAX | `instance.Q_MIN/Q_MAX` | Speed bounds | 0.94, 1.03 |
| H_MIN, H_MAX | `instance.H_MIN/H_MAX` | Heading bounds | -pi/6, +pi/6 (+/-30 deg) |

The nominal manoeuvre is (q = 1, dtheta = 0): the aircraft keeps its planned
trajectory.

## Conflict geometry (identical to the classical baseline)

For a pair (i, j) with chosen velocities v_i, v_j:
- xr0 = x0_i - x0_j, yr0 = y0_i - y0_j (relative initial position)
- vrx = v_ix - v_jx, vry = v_iy - v_jy (relative velocity)
- sep0 = (yr0^2 - d^2) vrx^2 + (xr0^2 - d^2) vry^2 - 2 xr0 yr0 vrx vry
- tm0 = -(xr0 vrx + yr0 vry) / (vrx^2 + vry^2)  (time of closest approach)
- Conflict iff sep0 < 0 and tm0 > 0 (they close to within d at a future time).

## QUBO variables and terms

| Symbol | Code | Meaning |
| --- | --- | --- |
| x[i,m] | flattened index `i*M + m` | 1 if aircraft i selects manoeuvre m |
| M | `qubo.M` | Number of manoeuvres per aircraft |
| V | `qubo.num_vars` | Total QUBO variables = n * M |
| cost[i,m] | in `build_qubo` | 0 if manoeuvre m is nominal, else 1 |
| C | `qubo.C` | Objective weight (default 1) |
| A | `qubo.A` | One-hot penalty weight (default 2(n+1)) |
| B | `qubo.B` | Separation (conflict) penalty weight (default 2(n+1)) |
| offset | `qubo.offset` | Constant energy term (from the one-hot squares) |

Energy: E(x) = C * sum cost[i,m] x[i,m] + A * sum_i (sum_m x[i,m] - 1)^2
             + B * sum over conflicting (i,m,j,m') x[i,m] x[j,m'].

## Ising mapping (for QAOA)

x_k = (1 - Z_k) / 2, giving E = sum_k h_k Z_k + sum_{k<l} J_kl Z_k Z_l + const.
See `solvers/qaoa_qiskit.qubo_to_ising`.

## Evaluation metrics

| Term | Code | Meaning |
| --- | --- | --- |
| one_hot_ok | `Solution.one_hot_ok` | Every aircraft selected exactly one manoeuvre |
| conflict_free | `Solution.conflict_free` | No residual pairwise conflict |
| feasible | `Solution.feasible` | one_hot_ok AND conflict_free |
| num_manoeuvres | `Solution.num_manoeuvres` | Count of non-nominal manoeuvres (objective) |
| energy | `Solution.energy` | QUBO energy of the assignment |
