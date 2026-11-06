"""Local (time-resolved) and global (Sobol) sensitivity analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sample

from pkbayes.inference import load_data
from pkbayes.sensitivity import PARAM_NAMES, local_sensitivity, tissue_exposure

ROOT = Path(__file__).resolve().parents[1]
N_BASE = 1024      # Saltelli base sample: N * (2D + 2) model evaluations
RANGE_FACTOR = 3   # each parameter varies log-uniformly within /3 .. x3
SEED = 11


def main():
    _, _, truth = load_data(ROOT / "data")
    p = truth["params"]
    dose = truth["dose"]

    # --- 1. Local, time-resolved sensitivity at the true parameters ---------
    t = np.linspace(0.05, 48, 500)
    S = local_sensitivity(p, dose, t)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for c, (ax, title) in enumerate(zip(axes, ["Blood", "Tissue"])):
        for name in PARAM_NAMES:
            ax.plot(t, S[name][c], label=name)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Time (h)")
        ax.set_title(f"{title}: d ln C / d ln θ")
    axes[0].set_ylabel("Normalized sensitivity")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "local_sensitivity.png", dpi=150)

    # --- 2. Global Sobol indices on tissue AUC and Cmax ---------------------
    problem = {
        "num_vars": len(PARAM_NAMES),
        "names": PARAM_NAMES,
        "bounds": [
            [np.log(p[n] / RANGE_FACTOR), np.log(p[n] * RANGE_FACTOR)]
            for n in PARAM_NAMES
        ],
    }
    X = sobol_sample.sample(problem, N_BASE, seed=SEED)   # samples in log space
    Q, CL, Kp = np.exp(X).T
    t_grid = np.linspace(0, 48, 961)
    auc, cmax = tissue_exposure(Q, CL, Kp, p["V_b"], p["V_t"], dose, t_grid)
    print(f"Model evaluations: {len(X)}")

    outputs = {"Tissue AUC (0-48 h)": auc, "Tissue Cmax": cmax}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    rows = []
    x = np.arange(len(PARAM_NAMES))
    for ax, (label, Y) in zip(axes, outputs.items()):
        Si = sobol.analyze(problem, Y, seed=SEED)
        ax.bar(x - 0.2, Si["S1"], 0.4, yerr=Si["S1_conf"], label="First order S1")
        ax.bar(x + 0.2, Si["ST"], 0.4, yerr=Si["ST_conf"], label="Total ST")
        ax.set_xticks(x, PARAM_NAMES)
        ax.set_title(label)
        for i, name in enumerate(PARAM_NAMES):
            rows.append(f"{label},{name},{Si['S1'][i]:.3f},{Si['ST'][i]:.3f}")
            print(f"{label:22s} {name:3s}  S1 = {Si['S1'][i]:.3f}   ST = {Si['ST'][i]:.3f}")
    axes[0].set_ylabel("Sobol index")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "sobol.png", dpi=150)

    with open(ROOT / "results" / "sobol.csv", "w") as f:
        f.write("output,parameter,S1,ST\n" + "\n".join(rows) + "\n")
    print("Saved figures/local_sensitivity.png, figures/sobol.png, results/sobol.csv")


if __name__ == "__main__":
    main()