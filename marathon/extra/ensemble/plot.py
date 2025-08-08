import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import foldnorm

from marathon.emit.plot import fig_and_ax

units = {"energy": "meV", "stress": "meV", "forces": "meV/Å"}


def plot(outfolder, predictions, labels, figsize=(10, 7), keys=None):
    if keys is None:
        all_keys = predictions.keys()

        keys = []
        for key in all_keys:
            if key + "_var" in predictions.keys():
                keys.append(key)

    for key in keys:
        fig, ax = fig_and_ax(figsize=figsize)
        calibration_figure(
            1e3 * np.abs(labels[key].flatten() - predictions[key].flatten()),
            1e3 * np.sqrt(predictions[key + "_var"]).flatten(),
            ax=ax,
            unit=units[key],
        )

        fig.savefig(outfolder / f"calibration_{key}.png")
        plt.close(fig)


def calibration_figure(
    error,
    std,
    ax=None,
    markersize: float = 3.0,
    unit="meV",
):
    if ax is None:
        fig, ax = fig_and_ax()

    x = np.linspace(1e-6, 5e3, 5)
    noise_level_2 = x

    quantiles_lower_01 = [foldnorm.ppf(0.15, 0.0, 0.0, i) for i in noise_level_2]
    quantiles_upper_01 = [foldnorm.ppf(0.85, 0.0, 0.0, i) for i in noise_level_2]
    quantiles_lower_05 = [foldnorm.ppf(0.05, 0.0, 0.0, i) for i in noise_level_2]
    quantiles_upper_05 = [foldnorm.ppf(0.95, 0.0, 0.0, i) for i in noise_level_2]
    quantiles_lower_005 = [foldnorm.ppf(0.005, 0.0, 0.0, i) for i in noise_level_2]
    quantiles_upper_005 = [foldnorm.ppf(0.995, 0.0, 0.0, i) for i in noise_level_2]

    ax.scatter(
        std,
        error,
        s=markersize,
        alpha=0.3,
        color="blue",
        rasterized=True,
        linewidth=0.0,
    )
    ax.loglog()
    ax.plot(x, quantiles_upper_05, color="gray", alpha=0.5)
    ax.plot(x, quantiles_lower_05, color="gray", alpha=0.5)
    ax.plot(x, quantiles_upper_01, color="gray", alpha=0.5)
    ax.plot(x, quantiles_lower_01, color="gray", alpha=0.5)
    ax.plot(x, quantiles_upper_005, color="gray", alpha=0.5)
    ax.plot(x, quantiles_lower_005, color="gray", alpha=0.5)

    ax.plot(np.logspace(-3, 100.0), np.logspace(-3, 100.0), linestyle="--", color="grey")
    ax.set_xlim(np.min(std) / 1.5, np.max(std) * 1.5)
    ax.set_ylim(np.min(error) / 1.5, np.max(error) * 1.5)

    xlabel = r"$\sigma$" + f" ({unit})"
    ylabel = r"$|\Delta|$" + f" ({unit})"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
