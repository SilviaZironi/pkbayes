import numpy as np
from pkbayes.model import simulate

def test_mass_balance():
    params = {"Q": 5.0, "CL": 1.0, "Kp": 3.0, "V_b": 5.0, "V_t": 10.0}
    out = simulate (params, dose = 100, t_eval = np.linspace(0,48,100))
    C_b = out[:,0]
    C_t = out[:,1]
    A_el = out[:,2]
    total = C_b*params["V_b"] + C_t*params["V_t"] + A_el
    assert np.allclose(total, 100, rtol = 1e-6)