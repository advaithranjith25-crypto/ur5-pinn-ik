"""
Phase 5: Analyze experiment_results.csv -- descriptive stats, paired
significance testing, and the honest full picture: success rate AND
iteration-count-among-converged, since these can tell different stories
(see project notes on the branch-mismatch finding).
"""

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_CSV = "experiment_results.csv"


def summarize(df):
    print("=" * 100)
    print("SUCCESS RATE by category / strategy / solver")
    print("=" * 100)
    success = df.groupby(["category", "solver", "strategy"])["converged"].mean().unstack("strategy")
    success = success[["cold", "pinn", "gated"]] * 100
    print(success.round(1).to_string())

    print()
    print("=" * 100)
    print("ITERATIONS TO CONVERGE (mean, among CONVERGED solves only)")
    print("=" * 100)
    converged_df = df[df["converged"]]
    iters = converged_df.groupby(["category", "solver", "strategy"])["n_iters"].mean().unstack("strategy")
    iters = iters[["cold", "pinn", "gated"]]
    print(iters.round(1).to_string())

    print()
    print("=" * 100)
    print("ITERATIONS TO CONVERGE (median, among CONVERGED solves only)")
    print("=" * 100)
    iters_med = converged_df.groupby(["category", "solver", "strategy"])["n_iters"].median().unstack("strategy")
    iters_med = iters_med[["cold", "pinn", "gated"]]
    print(iters_med.round(1).to_string())

    return success, iters


def paired_significance_tests(df):
    """
    Wilcoxon signed-rank test comparing cold-start vs gated iteration counts
    on the SAME target poses -- paired, not independent samples, since each
    pose was solved under all three strategies.
    """
    print()
    print("=" * 100)
    print("PAIRED SIGNIFICANCE TESTS (Wilcoxon signed-rank, cold vs gated)")
    print("Only over poses where BOTH cold and gated converged, so iteration")
    print("counts are meaningfully comparable (not conflated with success/failure)")
    print("=" * 100)

    for category in df["category"].unique():
        for solver in df["solver"].unique():
            sub = df[(df["category"] == category) & (df["solver"] == solver)]

            cold = sub[sub["strategy"] == "cold"].reset_index(drop=True)
            gated = sub[sub["strategy"] == "gated"].reset_index(drop=True)

            both_converged = cold["converged"] & gated["converged"]
            n_pairs = both_converged.sum()

            if n_pairs < 5:
                print(f"{category:15s} {solver}: only {n_pairs} paired convergent samples -- too few for a reliable test")
                continue

            cold_iters = cold.loc[both_converged, "n_iters"].values
            gated_iters = gated.loc[both_converged, "n_iters"].values

            stat, p = stats.wilcoxon(cold_iters, gated_iters)
            direction = "gated FEWER iters" if gated_iters.mean() < cold_iters.mean() else "gated MORE iters"
            sig = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant"
            print(f"{category:15s} {solver}: n={n_pairs}, cold_mean={cold_iters.mean():.1f}, "
                  f"gated_mean={gated_iters.mean():.1f}, {direction}, p={p:.4f} [{sig}]")


def make_plots(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    converged_df = df[df["converged"]]
    categories = df["category"].unique()
    solvers = df["solver"].unique()
    strategies = ["cold", "pinn", "gated"]
    colors = {"cold": "#888888", "pinn": "#4C72B0", "gated": "#55A868"}

    fig, axes = plt.subplots(len(solvers), len(categories), figsize=(14, 8), sharey="row")
    for si, solver in enumerate(solvers):
        for ci, category in enumerate(categories):
            ax = axes[si, ci] if len(solvers) > 1 else axes[ci]
            sub = converged_df[(converged_df["solver"] == solver) & (converged_df["category"] == category)]

            means = [sub[sub["strategy"] == s]["n_iters"].mean() for s in strategies]
            sems = [sub[sub["strategy"] == s]["n_iters"].sem() for s in strategies]

            ax.bar(strategies, means, yerr=sems, capsize=4,
                   color=[colors[s] for s in strategies])
            ax.set_title(f"{solver} / {category}", fontsize=10)
            if ci == 0:
                ax.set_ylabel("Iterations to converge")

    plt.tight_layout()
    plt.savefig("experiment_iterations_plot.png", dpi=150)
    print("\nSaved plot: experiment_iterations_plot.png")


if __name__ == "__main__":
    df = pd.read_csv(RESULTS_CSV)
    print(f"Loaded {len(df)} solve records from {RESULTS_CSV}\n")

    summarize(df)
    paired_significance_tests(df)
    make_plots(df)

    print()
    print("=" * 100)
    print("INTERPRETATION NOTE")
    print("=" * 100)
    print("Success rate and iteration-count-among-converged can tell different stories.")
    print("If gated has LOWER success rate but FEWER iterations when it does converge,")
    print("that's consistent with the branch-mismatch finding: a PINN prediction on the")
    print("wrong IK branch can trap a local solver (DLS) in a non-solution basin (hurting")
    print("success rate), while correct-branch predictions converge notably faster (helping")
    print("iteration count). Both effects are real and worth reporting together, not just")
    print("whichever one looks better.")
