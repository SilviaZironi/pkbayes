"""Generate the synthetic dataset used for Bayesian calibration.

Outputs:
    data/observations.csv  -->   noisy blood and tissue concentrations
    data/true_params.json  -->   ground-truth parameters, dose, noise level, seed
    figures/data.png    -->      observations over the true model curves
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pkbayes.data import simulate_observations
from pkbayes.model import simulate

TRUE_PARAMS = {"Q": 5.0, "CL": 1.0, "Kp": 3.0, "V_b": 5.0, "V_t": 10.0}
DOSE = 100.0  # mg
SIGMA = 0.15  # log-scale noise, ~15% relative error
SEED = 42
T_OBS = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12, 24, 36, 48])  # h

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    # Simulate observations 
    rng = np.random.default_rng(SEED)
    obs = simulate_observations(TRUE_PARAMS, DOSE, T_OBS, SIGMA, rng)

    # Save data
    table = np.column_stack([T_OBS, obs])
    np.savetxt(
        DATA_DIR / "observations.csv",
        table,
        delimiter=",",
        header="time_h,blood_mg_per_L,tissue_mg_per_L",
        comments="",
        fmt="%.6g",
    )
    with open(DATA_DIR / "true_params.json", "w") as f:
        json.dump(
            {"params": TRUE_PARAMS, "dose": DOSE, "sigma": SIGMA, "seed": SEED},
            f,
            indent=2,
        )

    # Plot data over the true curves 
    t_fine = np.linspace(0.05, T_OBS[-1], 500)
    true = simulate(TRUE_PARAMS, DOSE, t_fine)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (name, color) in enumerate([("Blood", "C0"), ("Tissue", "C1")]):
        ax.plot(t_fine, true[:, i], color=color, label=f"{name} (true)")
        ax.plot(T_OBS, obs[:, i], "o", color=color, label=f"{name} (observed)")
    ax.set_yscale("log")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Concentration (mg/L)")
    ax.set_title(f"Synthetic data (log-normal noise, sigma = {SIGMA})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "data.png", dpi=150)
    print(f"Saved {len(T_OBS)} observations to {DATA_DIR / 'observations.csv'}")


if __name__ == "__main__":
    main()