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

def analytic_solution(t, Q, CL, Kp, V_b, V_t, dose, lib=np):
    """Closed-form blood and tissue concentrations for an IV bolus.

    The model is linear, dx/dt = A x with x = [C_b, C_t], so the solution
    is a sum of two exponentials with rates given by the eigenvalues of A.
    Written with basic operations only, so it works both with NumPy
    (lib=np) and inside a PyMC model (lib=pm.math), where it is
    automatically differentiable for NUTS.

    Returns
    -------
    C_b, C_t : concentrations at times t (mg/L)
    """
    # Entries of the system matrix A
    a11 = -(Q + CL) / V_b
    a12 = Q / (Kp * V_b)
    a21 = Q / V_t
    a22 = -Q / (Kp * V_t)

    # Eigenvalues of A (real and distinct: the discriminant is always > 0)
    trace = a11 + a22
    det = a11 * a22 - a12 * a21
    root = lib.sqrt(trace**2 - 4 * det)
    lam1 = (trace + root) / 2   # slow (terminal) phase
    lam2 = (trace - root) / 2   # fast (distribution) phase

    c0 = dose / V_b
    e1 = lib.exp(lam1 * t)
    e2 = lib.exp(lam2 * t)

    C_b = c0 * ((lam1 - a22) * e1 - (lam2 - a22) * e2) / (lam1 - lam2)
    C_t = c0 * a21 * (e1 - e2) / (lam1 - lam2)
    return C_b, C_t