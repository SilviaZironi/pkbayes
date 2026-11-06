import numpy as np

from pkbayes.data import simulate_observations
from pkbayes.inference import build_model


def test_model_logp_is_finite():
    params = {"Q": 5.0, "CL": 1.0, "Kp": 3.0, "V_b": 5.0, "V_t": 10.0}
    t = np.array([0.5, 1, 2, 4, 8, 24, 48])
    obs = simulate_observations(params, 100.0, t, 0.15, np.random.default_rng(0))
    model = build_model(t, obs, 100.0, params["V_b"], params["V_t"])
    logps = model.point_logps()
    assert all(np.isfinite(v) for v in logps.values())