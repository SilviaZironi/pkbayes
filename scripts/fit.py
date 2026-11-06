"""Calibrate the model with NUTS and save the posterior."""

from pathlib import Path

import arviz as az
import pymc as pm

from pkbayes.inference import build_model, load_data

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 2026


def main():
    RESULTS.mkdir(exist_ok=True)
    t_obs, obs, truth = load_data(ROOT / "data")
    p = truth["params"]
    model = build_model(t_obs, obs, truth["dose"], p["V_b"], p["V_t"])

    with model:
        idata = pm.sample(
            draws=2000,
            tune=1000,
            chains=4,
            target_accept=0.9,
            random_seed=SEED,
        )

    idata.to_netcdf(RESULTS / "posterior.nc")

    summary = az.summary(idata, var_names=["Q", "CL", "Kp", "sigma"])
    print(summary)
    summary.to_csv(RESULTS / "summary.csv")

    n_div = int(idata["sample_stats"]["diverging"].values.sum())
    print(f"\nDivergent transitions: {n_div}")


if __name__ == "__main__":
    main()