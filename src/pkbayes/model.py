import numpy as np
from scipy.integrate import solve_ivp

def rhs(t,y,params):
    C_b,C_t,A_el = y
    Q = params["Q"]
    CL = params["CL"]
    Kp = params["Kp"]
    V_b = params["V_b"]
    V_t = params["V_t"]
    dC_b_dt = (Q*(C_t/Kp-C_b)-CL*C_b)/V_b
    dC_t_dt = (Q*(C_b-C_t/Kp))/V_t
    dA_el_dt = CL*C_b
    return [dC_b_dt,dC_t_dt,dA_el_dt]

def simulate(params, dose, t_eval):
    y0 = [dose/params["V_b"], 0, 0]
    t_span = (0, t_eval[-1])
    sol = solve_ivp(rhs, t_span = t_span, y0 = y0, t_eval = t_eval, args = (params,), rtol = 1e-8, atol = 1e-10)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T
