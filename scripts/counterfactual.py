"""Counterfactual: how much must Kp or CL change to bring tissue AUC below a
threshold, and how does the answer degrade with poorer experimental designs?"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import xarray as xr

from pkbayes.counterfactual import minimal_change
from pkbayes.inference import load_data, build_model
from pkbayes.sensitivity import tissue_exposure

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
THRESHOLD = 100.0   # hypothetical tissue AUC limit (mg*h/L)
N_SUB = 800         # posterior draws used per design
SEED = 2026
T_GRID = np.linspace(0, 48, 961)


def fit(t_obs, obs, truth, compartments, path):
    """Fit a design (or load it if already fitted) and return posterior samples."""
    if not path.exists():
        p = truth["params"]
        model = build_model(t_obs, obs, truth["dose"], p["V_b"], p["V_t"], compartments)
        with model:
            idata = pm.sample(2000, tune=1000, chains=4, target_accept=0.9,
                              random_seed=SEED, progressbar=False)
        idata.to_netcdf(path)
    idata = xr.open_datatree(path)
    return {k: idata["posterior"][k].values.ravel() for k in ["Q", "CL", "Kp"]}


def main():
    t_obs, obs, truth = load_data(ROOT / "data")
    p = truth["params"]
    dose = truth["dose"]
    early = t_obs < 4

    designs = {
        "Full (blood + tissue)": fit(t_obs, obs, truth, ("blood", "tissue"),
                                     RESULTS / "posterior.nc"),
        "No samples before 4 h": fit(t_obs[~early], obs[~early], truth,
                                     ("blood", "tissue"), RESULTS / "posterior_late.nc"),
        "Blood only": fit(t_obs, obs[:, :1], truth, ("blood",),
                          RESULTS / "posterior_blood_only.nc"),
    }

    true_auc, _ = tissue_exposure(p["Q"], p["CL"], p["Kp"], p["V_b"], p["V_t"], dose, T_GRID)
    true_f = {k: minimal_change(k, p, dose, T_GRID, THRESHOLD) for k in ["Kp", "CL"]}
    print(f"True tissue AUC: {true_auc[0]:.1f} mg*h/L (threshold {THRESHOLD})")
    print(f"True minimal change: Kp x{true_f['Kp']:.3f}, CL x{true_f['CL']:.3f}\n")

    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    header = f"{'design':24s} {'P(AUC>T)':>9s}   {'Kp factor (94%)':>24s}   {'CL factor (94%)':>24s}"
    print(header)
    for c, (label, post) in enumerate(designs.items()):
        idx = rng.choice(len(post["Q"]), size=N_SUB, replace=False)
        draws = [{"Q": post["Q"][k], "CL": post["CL"][k], "Kp": post["Kp"][k],
                  "V_b": p["V_b"], "V_t": p["V_t"]} for k in idx]

        auc, _ = tissue_exposure(post["Q"][idx], post["CL"][idx], post["Kp"][idx],
                                 p["V_b"], p["V_t"], dose, T_GRID)
        f_kp = np.array([minimal_change("Kp", d, dose, T_GRID, THRESHOLD) for d in draws])
        f_cl = np.array([minimal_change("CL", d, dose, T_GRID, THRESHOLD) for d in draws])

        q_kp = np.quantile(f_kp, [0.03, 0.5, 0.97])
        q_cl = np.quantile(f_cl, [0.03, 0.5, 0.97])
        print(f"{label:24s} {np.mean(auc > THRESHOLD):9.2f}   "
              f"{q_kp[1]:6.3f} [{q_kp[0]:.3f}, {q_kp[2]:.3f}]   "
              f"{q_cl[1]:6.3f} [{q_cl[0]:.3f}, {q_cl[2]:.3f}]")

        axes[0].hist(auc, bins=40, histtype="step", density=True, color=f"C{c}", label=label)
        axes[1].hist(f_kp, bins=40, histtype="step", density=True, color=f"C{c}")
        axes[2].hist(f_cl, bins=40, histtype="step", density=True, color=f"C{c}")

    axes[0].axvline(THRESHOLD, color="r", label="Threshold")
    axes[0].axvline(true_auc[0], color="k", ls="--", label="True value")
    axes[1].axvline(true_f["Kp"], color="k", ls="--")
    axes[2].axvline(true_f["CL"], color="k", ls="--")
    axes[0].set_xlabel("Tissue AUC 0-48 h (mg·h/L)")
    axes[1].set_xlabel("Required factor on Kp")
    axes[2].set_xlabel("Required factor on CL")
    axes[0].set_title("Current exposure")
    axes[1].set_title("Fix by lowering partitioning")
    axes[2].set_title("Fix by raising clearance")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "counterfactual.png", dpi=150)
    print("\nSaved figures/counterfactual.png")


if __name__ == "__main__":
    main()