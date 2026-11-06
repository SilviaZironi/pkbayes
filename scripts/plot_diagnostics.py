"""Trace plots and parameter recovery from the saved posterior."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pkbayes.inference import load_data

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
NAMES = ["Q", "CL", "Kp", "sigma"]


def main():
    _, _, truth = load_data(ROOT / "data")
    true_vals = {k: truth["params"][k] for k in ["Q", "CL", "Kp"]}
    true_vals["sigma"] = truth["sigma"]

    idata = xr.open_datatree(RESULTS / "posterior.nc")
    post = idata["posterior"]

    # --- Trace plot: per-chain posterior (left) and chain history (right) ---
    fig, axes = plt.subplots(len(NAMES), 2, figsize=(11, 2.4 * len(NAMES)))
    for i, name in enumerate(NAMES):
        samples = post[name].values  # shape (chain, draw)
        for c in range(samples.shape[0]):
            axes[i, 0].hist(samples[c], bins=50, density=True,
                            histtype="step", color=f"C{c}")
            axes[i, 1].plot(samples[c], color=f"C{c}", lw=0.3, alpha=0.7)
        axes[i, 0].axvline(true_vals[name], color="k", ls="--", label="true value")
        axes[i, 0].set_ylabel(name)
    axes[0, 0].set_title("Posterior (one line per chain)")
    axes[0, 1].set_title("Trace")
    axes[0, 0].legend()
    axes[-1, 1].set_xlabel("Draw")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "trace.png", dpi=150)

    # --- Parameter recovery: is the true value inside the 94% interval? ---
    print(f"{'param':>6} {'true':>8} {'median':>8} {'3%':>8} {'97%':>8}  inside?")
    for name in NAMES:
        s = post[name].values.ravel()
        lo, med, hi = np.quantile(s, [0.03, 0.5, 0.97])
        inside = lo <= true_vals[name] <= hi
        print(f"{name:>6} {true_vals[name]:8.3f} {med:8.3f} {lo:8.3f} {hi:8.3f}  {inside}")
    print("\nSaved figures/trace.png")


if __name__ == "__main__":
    main()