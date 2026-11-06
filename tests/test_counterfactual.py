import numpy as np

from pkbayes.counterfactual import minimal_change
from pkbayes.sensitivity import tissue_exposure

PARAMS = {"Q": 5.0, "CL": 1.0, "Kp": 3.0, "V_b": 5.0, "V_t": 10.0}
DOSE = 100.0
T = np.linspace(0, 48, 961)


def _auc(p):
    return tissue_exposure(p["Q"], p["CL"], p["Kp"], p["V_b"], p["V_t"], DOSE, T)[0][0]


def test_change_hits_threshold_exactly():
    for param in ["Kp", "CL"]:
        f = minimal_change(param, PARAMS, DOSE, T, threshold=100.0)
        fixed = {**PARAMS, param: PARAMS[param] * f}
        assert np.isclose(_auc(fixed), 100.0, rtol=1e-6)


def test_change_direction():
    assert minimal_change("Kp", PARAMS, DOSE, T, 100.0) < 1   # lower partitioning
    assert minimal_change("CL", PARAMS, DOSE, T, 100.0) > 1   # faster clearance


def test_no_change_needed_below_threshold():
    assert minimal_change("Kp", PARAMS, DOSE, T, threshold=1e6) == 1.0