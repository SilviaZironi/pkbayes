import numpy as np
import matplotlib.pyplot as plt
from pkbayes.model import simulate

params = {"Q": 5.0, "CL": 1.0, "Kp": 3.0, "V_b": 5.0, "V_t": 10.0}              
dose = 100.0
t_eval = np.linspace(0, 48, 500)    

out = simulate(params, dose, t_eval)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(t_eval, out[:,0], label="Blood")
ax.plot(t_eval, out[:,1], label="Tissue")
ax.set_yscale("log")
ax.set_xlabel("Time(h)")         
ax.set_ylabel("Concentration(mg/L)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/simulation.png", dpi=150)