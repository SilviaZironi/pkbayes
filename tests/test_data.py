import numpy as np
import pytest

from pkbayes.data import simulate_observations
from pkbayes.model import simulate

PARAMS = {"Q": 5.0, "CL": 1.0, "Kp": 3.0, "V_b": 5.0, "V_t": 10.0}
DOSE = 100.0
T_OBS = np.array([0.5, 1, 2, 4, 8, 24, 48])


def test_same_seed_gives_same_data():
    a = simulate_observations(PARAMS, DOSE, T_OBS, 0.15, np.random.default_rng(0))
    b = simulate_observations(PARAMS, DOSE, T_OBS, 0.15, np.random.default_rng(0))
    assert np.array_equal(a, b)


def test_observations_are_positive():
    obs = simulate_observations(PARAMS, DOSE, T_OBS, 0.5, np.random.default_rng(1))
    assert np.all(obs > 0)


def test_rejects_time_zero():
    with pytest.raises(ValueError):
        simulate_observations(PARAMS, DOSE, [0.0, 1.0], 0.15, np.random.default_rng(2))


def test_noise_has_requested_sigma():
    sigma = 0.2
    t = np.linspace(1, 48, 5000)
    obs = simulate_observations(PARAMS, DOSE, t, sigma, np.random.default_rng(3))
    true = simulate(PARAMS, DOSE, t)[:, :2]
    log_resid = np.log(obs / true)
    assert abs(log_resid.mean()) < 0.01
    assert abs(log_resid.std() - sigma) < 0.01