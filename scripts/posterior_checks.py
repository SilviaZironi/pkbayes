"""Posterior predictive check and parameter correlations."""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pkbayes.inference import load_data
from pkbayes.model import analytic_solution

ROOT = Path(__file__).resolve().parents[1]
N_SUB = 1000   # posterior draws used for the predictive bands
SEED = 7


def predictive_draws(post, idx, t, V_b, V_t, dose, rng):
    """Model curves and noisy predictions for a subset of posterior draws."""
    n = len(idx)
    curves = np.empty((n, len(t), 2))
    preds = np.empty((n, len(t), 2))
    for j, k in enumerate(idx):
        C_b, C_t = analytic_solution(
            t, post["Q"][k], post["CL"][k], post["Kp"][k], V_b, V_t, dose
        )
        curves[j] = np.column_stack([C_b, C_t])
        noise = rng.normal(0.0, post["sigma"][k], size=(len(t), 2))
        preds[j] = curves[j] * np.exp(noise)
    return curves, preds


def main():
    t_obs, obs, truth = load_data(ROOT / "data")
    p = truth["params"]
    V_b, V_t, dose = p["V_b"], p["V_t"], truth["dose"]

    idata = xr.open_datatree(ROOT / "results" / "posterior.nc")
    post = {k: idata["posterior"][k].values.ravel() for k in ["Q", "CL", "Kp", "sigma"]}

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(post["Q"]), size=N_SUB, replace=False)

    # --- 1. Posterior predictive bands on a fine time grid ------------------
    t_fine = np.linspace(0.05, t_obs[-1], 300)
    curves, preds = predictive_draws(post, idx, t_fine, V_b, V_t, dose, rng)
    true_b, true_t = analytic_solution(t_fine, dose=dose, **p)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for c, (ax, name, true_curve) in enumerate(
        zip(axes, ["Blood", "Tissue"], [true_b, true_t])
    ):
        lo_p, hi_p = np.quantile(preds[:, :, c], [0.03, 0.97], axis=0)
        lo_c, med_c, hi_c = np.quantile(curves[:, :, c], [0.03, 0.5, 0.97], axis=0)
        ax.fill_between(t_fine, lo_p, hi_p, color=f"C{c}", alpha=0.15,
                        label="94% predictive interval")
        ax.fill_between(t_fine, lo_c, hi_c, color=f"C{c}", alpha=0.4,
                        label="94% interval of model curve")
        ax.plot(t_fine, med_c, color=f"C{c}", label="Posterior median")
        ax.plot(t_fine, true_curve, "k--", lw=1, label="True curve")
        ax.plot(t_obs, obs[:, c], "ko", ms=4, label="Observed")
        ax.set_yscale("log")
        ax.set_xlabel("Time (h)")
        ax.set_title(name)
    axes[0].set_ylabel("Concentration (mg/L)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "posterior_predictive.png", dpi=150)

    # --- 2. Coverage: how many observations fall in the 94% interval? -------
    _, preds_obs = predictive_draws(post, idx, t_obs, V_b, V_t, dose, rng)
    lo, hi = np.quantile(preds_obs, [0.03, 0.97], axis=0)
    inside = (obs >= lo) & (obs <= hi)
    print(f"Observations inside 94% predictive interval: "
          f"{inside.sum()}/{inside.size} ({100 * inside.mean():.0f}%)")

    # --- 3. Pair plot: correlations between parameters ----------------------
    names = ["Q", "CL", "Kp"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (a, b) in zip(axes, combinations(names, 2)):
        r = np.corrcoef(post[a], post[b])[0, 1]
        ax.scatter(post[a][idx], post[b][idx], s=4, alpha=0.3)
        ax.plot(p[a], p[b], "r*", ms=14, label="True value")
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title(f"corr({a}, {b}) = {r:+.2f}")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "pair_plot.png", dpi=150)
    print("Saved figures/posterior_predictive.png and figures/pair_plot.png")


if __name__ == "__main__":
    main()