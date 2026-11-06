"""Prior predictive check: do the priors produce plausible concentration curves?"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from pkbayes.inference import build_model, load_data
from pkbayes.model import analytic_solution

ROOT = Path(__file__).resolve().parents[1]
N_DRAWS = 200
SEED = 1


def main():
    t_obs, obs, truth = load_data(ROOT / "data")
    V_b, V_t = truth["params"]["V_b"], truth["params"]["V_t"]
    dose = truth["dose"]

    model = build_model(t_obs, obs, dose, V_b, V_t)
    with model:
        prior = pm.sample_prior_predictive(draws=N_DRAWS, random_seed=SEED)

    draws = {name: prior["prior"][name].values.ravel() for name in ["Q", "CL", "Kp"]}

    t_fine = np.linspace(0.05, t_obs[-1], 300)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for k in range(N_DRAWS):
        C_b, C_t = analytic_solution(
            t_fine, draws["Q"][k], draws["CL"][k], draws["Kp"][k], V_b, V_t, dose
        )
        axes[0].plot(t_fine, C_b, color="C0", alpha=0.08)
        axes[1].plot(t_fine, C_t, color="C1", alpha=0.08)

    for ax, col, name in zip(axes, [0, 1], ["Blood", "Tissue"]):
        ax.plot(t_obs, obs[:, col], "ko", ms=4, label="Observed")
        ax.set_yscale("log")
        ax.set_xlabel("Time (h)")
        ax.set_title(f"{name}: {N_DRAWS} prior draws")
        ax.legend()
    axes[0].set_ylabel("Concentration (mg/L)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "prior_predictive.png", dpi=150)
    print("Saved figures/prior_predictive.png")


if __name__ == "__main__":
    main()