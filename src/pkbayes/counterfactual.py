"""Minimal parameter changes that bring tissue exposure below a threshold."""

import numpy as np
from scipy.optimize import brentq

from pkbayes.sensitivity import tissue_exposure

# Direction in which each parameter must move to reduce tissue AUC
# (AUC grows with Kp, decreases with CL), as a bracket on log(factor).
LOG_BRACKET = {"Kp": (-10.0, 0.0), "CL": (0.0, 10.0)}


def minimal_change(param, params, dose, t, threshold):
    """Smallest multiplicative factor on `param` giving tissue AUC = threshold.

    Returns 1.0 if the particle is already below the threshold.
    A factor of 0.4 on Kp means "reduce Kp by 60%"; a factor of 2.5 on CL
    means "increase clearance 2.5-fold".
    """
    def gap(log_f):
        p = dict(params)
        p[param] = params[param] * np.exp(log_f)
        auc, _ = tissue_exposure(p["Q"], p["CL"], p["Kp"], p["V_b"], p["V_t"], dose, t)
        return auc[0] - threshold

    if gap(0.0) <= 0:
        return 1.0
    lo, hi = LOG_BRACKET[param]
    return float(np.exp(brentq(gap, lo, hi)))