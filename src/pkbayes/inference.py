"""Bayesian calibration of the two-compartment model with PyMC."""

import json
from pathlib import Path

import numpy as np
import pymc as pm

from pkbayes.model import analytic_solution


def load_data(data_dir):
    """Load observations and ground truth produced by scripts/generate_data.py."""
    data_dir = Path(data_dir)
    table = np.loadtxt(data_dir / "observations.csv", delimiter=",", skiprows=1)
    t_obs, obs = table[:, 0], table[:, 1:]
    with open(data_dir / "true_params.json") as f:
        truth = json.load(f)
    return t_obs, obs, truth


def build_model(t_obs, obs, dose, V_b, V_t):
    """PyMC model: log-normal likelihood around the closed-form solution.

    Unknown: Q, CL, Kp (physiology/particle) and sigma (measurement noise).
    Known:   dose and compartment volumes (set by the experimental device).
    """
    coords = {"time": t_obs, "compartment": ["blood", "tissue"]}
    with pm.Model(coords=coords) as model:
        # Weakly informative priors: positive, median at a plausible order of
        # magnitude, ~95% of mass within a factor ~7 of the median.
        Q = pm.LogNormal("Q", mu=np.log(2.0), sigma=1.0)
        CL = pm.LogNormal("CL", mu=np.log(0.5), sigma=1.0)
        Kp = pm.LogNormal("Kp", mu=np.log(1.0), sigma=1.0)
        sigma = pm.HalfNormal("sigma", sigma=0.5)

        C_b, C_t = analytic_solution(t_obs, Q, CL, Kp, V_b, V_t, dose, lib=pm.math)
        log_pred = pm.math.log(pm.math.stack([C_b, C_t], axis=1))

        pm.Normal(
            "log_obs",
            mu=log_pred,
            sigma=sigma,
            observed=np.log(obs),
            dims=("time", "compartment"),
        )
    return model