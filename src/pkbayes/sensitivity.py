"""Local and global sensitivity of tissue exposure to model parameters."""

import numpy as np

from pkbayes.model import analytic_solution

PARAM_NAMES = ["Q", "CL", "Kp"]


def tissue_exposure(Q, CL, Kp, V_b, V_t, dose, t):
    """Tissue AUC over t and peak tissue concentration.

    Q, CL, Kp can be arrays of N parameter sets: the closed-form solution
    is evaluated for all of them at once (shape (N, len(t))).
    """
    Q, CL, Kp = (np.atleast_1d(np.asarray(x, dtype=float))[:, None] for x in (Q, CL, Kp))
    _, C_t = analytic_solution(t, Q, CL, Kp, V_b, V_t, dose)
    auc = np.trapezoid(C_t, t, axis=-1)
    cmax = C_t.max(axis=-1)
    return auc, cmax


def local_sensitivity(params, dose, t, rel_step=1e-4):
    """Normalized sensitivity coefficients S(t) = d ln C(t) / d ln theta.

    S = 0.5 means: a +1% change in theta gives a +0.5% change in C(t).
    Computed by central finite differences in log space. t must be > 0.

    Returns
    -------
    dict name -> (S_blood(t), S_tissue(t))
    """
    d_ln = np.log1p(rel_step) - np.log1p(-rel_step)
    result = {}
    for name in PARAM_NAMES:
        up, down = dict(params), dict(params)
        up[name] *= 1 + rel_step
        down[name] *= 1 - rel_step
        Cb_up, Ct_up = analytic_solution(t, dose=dose, **up)
        Cb_dn, Ct_dn = analytic_solution(t, dose=dose, **down)
        result[name] = (
            (np.log(Cb_up) - np.log(Cb_dn)) / d_ln,
            (np.log(Ct_up) - np.log(Ct_dn)) / d_ln,
        )
    return result