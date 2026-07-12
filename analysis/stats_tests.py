#!/usr/bin/env python3
"""Statistical analysis of the ACRP benchmark (Section 2.7 -> Chapter 3).

Reads results/benchmark.csv (produced by analysis/benchmark.py) and produces:
  - a per-method summary with 95% confidence intervals (Wilson for the
    optimum-hit proportion, bootstrap for mean gap);
  - a Friedman omnibus test across methods with Kendall's W effect size;
  - a Nemenyi post-hoc critical difference (scikit-posthocs if available, else a
    built-in critical-value table plus mean ranks);
  - paired Wilcoxon signed-rank tests of each QAOA method against the classical
    twin, with rank-biserial effect sizes and Holm-corrected p-values.

Requires: numpy, pandas, scipy. Optional: scikit-posthocs.

Run:
    python3 analysis/stats_tests.py --metric energy_gap --depth 3
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from _common import results_dir  # sets sys.path and gives the results directory

# Studentized-range-based critical values (alpha=0.05) for the Nemenyi test,
# divided by sqrt(2), indexed by the number of compared methods k.
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
       7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values, reps=10000, alpha=0.05, seed=0):
    v = np.asarray([x for x in values
                    if not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(reps, v.size))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def method_label(solver, depth):
    if isinstance(depth, float) and math.isnan(depth):
        return solver
    return f"{solver}_p{int(depth)}"


def load(csv):
    df = pd.read_csv(csv)
    df["method"] = [method_label(s, d) for s, d in zip(df["solver"], df["depth"])]
    return df


def pivot_blocks(df, methods, metric):
    sub = df[df["method"].isin(methods)]
    agg = sub.groupby(["instance", "method"])[metric].mean().reset_index()
    table = agg.pivot(index="instance", columns="method", values=metric)
    return table.reindex(columns=methods).dropna(axis=0, how="any")


def friedman(table):
    from scipy.stats import friedmanchisquare, rankdata
    data = [table[c].values for c in table.columns]
    stat, p = friedmanchisquare(*data)
    N, k = table.shape
    kendall_w = stat / (N * (k - 1)) if N * (k - 1) > 0 else float("nan")
    ranks = np.vstack([rankdata(r) for r in table.values])
    mean_ranks = dict(zip(table.columns, ranks.mean(axis=0)))
    return stat, p, kendall_w, mean_ranks, N, k


def nemenyi(table):
    N, k = table.shape
    cd = Q05.get(k, float("nan")) * math.sqrt(k * (k + 1) / (6.0 * N))
    try:
        import scikit_posthocs as sp
        m = sp.posthoc_nemenyi_friedman(table.values)
        m.index = table.columns
        m.columns = table.columns
        return cd, m
    except Exception:
        return cd, None


def wilcoxon_vs(df, reference, others, metric):
    from scipy.stats import wilcoxon, rankdata
    rows = []
    raw_p = []
    for m in others:
        table = pivot_blocks(df, [reference, m], metric)
        a = table[reference].values
        b = table[m].values
        diff = b - a
        nz = diff[diff != 0]
        if nz.size == 0:
            rows.append({"method": m, "n": int(len(diff)), "W": float("nan"),
                         "p": float("nan"), "rank_biserial": float("nan"),
                         "median_diff": 0.0})
            raw_p.append(1.0)
            continue
        try:
            stat, p = wilcoxon(a, b)
        except Exception:
            stat, p = float("nan"), 1.0
        r = rankdata(np.abs(nz))
        r_plus = r[nz > 0].sum()
        r_minus = r[nz < 0].sum()
        rrb = (r_plus - r_minus) / (r_plus + r_minus)
        rows.append({"method": m, "n": int(nz.size), "W": float(stat),
                     "p": float(p), "rank_biserial": float(rrb),
                     "median_diff": float(np.median(diff))})
        raw_p.append(float(p))
    for row_, a_ in zip(rows, holm(raw_p)):
        row_["p_holm"] = a_
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="ACRP benchmark statistics.")
    ap.add_argument("--csv", default=results_dir() + "/benchmark.csv")
    ap.add_argument("--metric", default="energy_gap")
    ap.add_argument("--depth", type=int, default=None,
                    help="QAOA depth for cross-solver tests (default: max present).")
    ap.add_argument("--reference", default="simulated_annealing")
    ap.add_argument("--out", default=results_dir() + "/stats_summary.csv")
    args = ap.parse_args()

    df = load(args.csv)

    summ = []
    for method, g in df.groupby("method"):
        gap_vals = g[args.metric].dropna().values
        lo_g, hi_g = bootstrap_ci(gap_vals)
        k_hit = int(g["hit_opt"].sum())
        n_hit = int(g["hit_opt"].count())
        wl, wh = wilson_ci(k_hit, n_hit)
        popt = g["p_opt"].dropna().values
        summ.append({
            "method": method, "runs": len(g),
            "hit_rate": (k_hit / n_hit) if n_hit else float("nan"),
            "hit_ci_lo": wl, "hit_ci_hi": wh,
            "mean_metric": float(np.nanmean(g[args.metric].values)) if gap_vals.size else float("nan"),
            "metric_ci_lo": lo_g, "metric_ci_hi": hi_g,
            "mean_p_opt": float(popt.mean()) if popt.size else float("nan"),
            "mean_seconds": float(g["seconds"].mean()),
        })
    summary = pd.DataFrame(summ).sort_values("method")
    print(f"=== Per-method summary (metric='{args.metric}', 95% CI) ===")
    print(summary.to_string(index=False))
    summary.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")

    depths_present = sorted({int(d) for d in df["depth"].dropna().unique()})
    depth = args.depth if args.depth is not None else (
        depths_present[-1] if depths_present else None)
    all_methods = set(df["method"])
    candidates = [args.reference]
    for base in ["qaoa_numpy", "qaoa_qiskit", "qaoa_qiskit_noisy"]:
        label = f"{base}_p{depth}" if depth is not None else base
        if label in all_methods:
            candidates.append(label)
    methods = [m for m in candidates if m in all_methods]
    print(f"\nComparison methods (QAOA depth p={depth}): {methods}")
    if len(methods) < 2:
        print("Not enough methods present for tests.")
        return

    table = pivot_blocks(df, methods, args.metric)
    print(f"Complete instance blocks used ({table.shape[0]}): {list(table.index)}")
    if table.shape[0] < 2:
        print("Not enough complete instance blocks for tests.")
        return

    if len(methods) >= 3:
        stat, p, w, mean_ranks, N, k = friedman(table)
        print(f"\n=== Friedman across {k} methods on {N} instances ===")
        print(f"chi2={stat:.4f}, p={p:.4g}, Kendall W={w:.4f}")
        print("Mean ranks (lower is better):")
        for m in sorted(mean_ranks, key=mean_ranks.get):
            print(f"  {m}: {mean_ranks[m]:.3f}")
        cd, matrix = nemenyi(table)
        print(f"Nemenyi critical difference (alpha=0.05): {cd:.3f}")
        if matrix is not None:
            print("Nemenyi pairwise p-values:")
            print(matrix.to_string())
        else:
            print("(install scikit-posthocs for pairwise Nemenyi p-values; "
                  "meanwhile compare mean-rank differences to the CD above.)")

    others = [m for m in methods if m != args.reference]
    if others:
        wx = wilcoxon_vs(df, args.reference, others, args.metric)
        print(f"\n=== Wilcoxon signed-rank vs {args.reference} (Holm-corrected) ===")
        print(wx.to_string(index=False))


if __name__ == "__main__":
    main()
