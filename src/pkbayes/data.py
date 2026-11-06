"""Generate synthetic noisy observations from the PBPK model."""

import numpy as np

from pkbayes.model import simulate


def simulate_observations(params, dose, t_obs, sigma, rng):
    """Simulate noisy blood and tissue concentrations.

    Noise is multiplicative log-normal:
        C_obs = C_true * exp(eps),  eps ~ N(0, sigma)

    Parameters
    ----------
    params : dict
        Model parameters (Q, CL, Kp, V_b, V_t).
    dose : float
        Bolus dose injected into blood at t = 0 (mg).
    t_obs : array_like
        Observation times (h), strictly positive.
    sigma : float
        Standard deviation of the log-scale noise (~ relative error).
    rng : numpy.random.Generator
        Random number generator, e.g. np.random.default_rng(seed).

    Returns
    -------
    numpy.ndarray, shape (len(t_obs), 2)
        Observed concentrations: column 0 = blood, column 1 = tissue (mg/L).
    """
    t_obs = np.asarray(t_obs, dtype=float)
    if np.any(t_obs <= 0):
        raise ValueError("Observation times must be strictly positive.")

    out = simulate(params, dose, t_obs)
    C_true = out[:, :2]

    eps = rng.normal(0.0, sigma, size=C_true.shape)
    return C_true * np.exp(eps)