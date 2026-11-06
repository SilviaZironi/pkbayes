import numpy as np

from pkbayes.model import analytic_solution
from pkbayes.sensitivity import local_sensitivity

PARAMS = {"Q": 5.0, "CL": 1.0, "Kp": 3.0, "V_b": 5.0, "V_t": 10.0}
DOSE = 100.0
# Fine near t = 0 (fast phase), long enough for the slow phase to vanish
T_LONG = np.concatenate([[0.0], np.geomspace(1e-4, 5000, 20000)])


def test_blood_auc_equals_dose_over_clearance():
    C_b, _ = analytic_solution(T_LONG, dose=DOSE, **PARAMS)
    auc = np.trapezoid(C_b, T_LONG)
    assert np.isclose(auc, DOSE / PARAMS["CL"], rtol=1e-3)


def test_tissue_auc_independent_of_flow():
    expected = PARAMS["Kp"] * DOSE / PARAMS["CL"]
    for Q in [0.5, 5.0, 50.0]:
        params = {**PARAMS, "Q": Q}
        _, C_t = analytic_solution(T_LONG, dose=DOSE, **params)
        assert np.isclose(np.trapezoid(C_t, T_LONG), expected, rtol=1e-3)


def test_local_sensitivity_blood_initial_value():
    # At t -> 0, C_b -> dose / V_b, which does not depend on Q, CL or Kp
    S = local_sensitivity(PARAMS, DOSE, np.array([1e-6]))
    for name in ["Q", "CL", "Kp"]:
        assert abs(S[name][0][0]) < 1e-3