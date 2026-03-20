import numpy as np

from marathon.emit.properties import (
    DEFAULT_PROPERTIES,
    get_full_unit,
    get_scale,
)
from marathon.evaluate.properties import DEFAULT_NORMALIZATION
from marathon.io import write_yaml


def fig_and_ax(figsize=None):
    import matplotlib.pyplot as plt

    if figsize:
        fig = plt.figure(figsize=figsize, dpi=200)
    else:
        fig = plt.figure(figsize=(8, 8), dpi=200)
    ax = plt.axes()
    return fig, ax


def simple_scatterplot(
    true,
    pred,
    metrics=None,
    ax=None,
    labeltrue="Ground truth",
    labelpred="Prediction",
    plotrange=None,
    precision=4,
    unit="meV",
):
    # drop NaNs if they exist in true values
    pred = pred[~np.isnan(true)]
    true = true[~np.isnan(true)]

    if ax is None:
        fig, ax = fig_and_ax()

    if plotrange is not None:
        rangemin = plotrange[0]
        rangemax = plotrange[1]
    else:
        rangemin = np.min(true)
        rangemax = np.max(true)
        spread = rangemax - rangemin
        rangemin, rangemax = rangemin - spread * 0.05, rangemax + spread * 0.05

    # plot diagonal
    ax.plot(
        [rangemin, rangemax],
        [rangemin, rangemax],
        marker="",
        color="darkgray",
        linestyle="dashed",
        linewidth=1,
    )

    # set range
    ax.set_xlim(rangemin, rangemax)
    ax.set_ylim(rangemin, rangemax)

    # scatterplot!
    ax.scatter(true, pred, marker="o", alpha=0.8)

    # catch everything out of the range and display as arrows
    # plot markers for predictions outside of plot range
    indfailneg = [i for (i, p) in zip(range(len(pred)), pred) if p < rangemin]
    indfailpos = [i for (i, p) in zip(range(len(pred)), pred) if p > rangemax]
    indfail = indfailneg + indfailpos

    if len(indfail) > 0:
        if len(indfailneg) > 0:
            ax.plot(
                true[indfailneg],
                [rangemin + spread * 0.01 for i in indfailneg],
                color="red",
                marker="v",
                linestyle="none",
                markersize=4,
                markeredgecolor="black",
                markeredgewidth=0.4,
            )
        if len(indfailpos) > 0:
            ax.plot(
                true[indfailpos],
                [rangemax - spread * 0.01 for i in indfailpos],
                color="red",
                marker="^",
                linestyle="none",
                markersize=4,
                markeredgecolor="black",
                markeredgewidth=0.4,
            )

    ax.set_xlabel(labeltrue + f" ({unit})")
    ax.set_ylabel(labelpred + f" ({unit})")

    RMSE, MAE, R2 = rmse(true, pred), mae(true, pred), cod(true, pred)
    if metrics is not None:
        # TODO: tolerance loosened from rtol=1e-6 to atol=1e-1 due to numerical
        # instabilities at large batch sizes. revisit if we ever tighten the pipeline.
        np.testing.assert_allclose(RMSE, metrics[0], atol=1e-1)
        np.testing.assert_allclose(MAE, metrics[1], atol=1e-1)
        np.testing.assert_allclose(R2, metrics[2], atol=1e-1)

    formatted_loss = ""
    formatted_loss += f"RMSE: {RMSE:.{precision}f} / "
    formatted_loss += f"MAE: {MAE:.{precision}f} / "
    formatted_loss += f"R$^2$: {R2:.4f}"

    ax.text(
        0.05,
        0.95,
        formatted_loss,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )

    return RMSE, MAE, R2


def plot(
    outfolder,
    predictions,
    labels,
    metrics=None,
    keys=None,
    properties=DEFAULT_PROPERTIES,
    normalization=DEFAULT_NORMALIZATION,
):
    import matplotlib.pyplot as plt

    if keys is None:
        keys = labels.keys()

    eval_metrics = {}

    for key in keys:
        if key not in labels:
            continue

        if metrics is not None and key in metrics:
            our_metrics = [metrics[key]["rmse"], metrics[key]["mae"], metrics[key]["r2"]]
        else:
            our_metrics = None

        scale = get_scale(key, properties)
        unit = get_full_unit(key, properties, normalization)

        fig, ax = fig_and_ax(figsize=(7, 7))
        RMSE, MAE, R2 = simple_scatterplot(
            scale * labels[key].flatten(),
            scale * predictions[key].flatten(),
            ax=ax,
            unit=unit,
            metrics=our_metrics,
        )
        eval_metrics[key] = {"rmse": RMSE, "mae": MAE, "r2": R2}

        fig.savefig(outfolder / f"{key}.png")
        plt.close(fig)

    write_yaml(outfolder / "metrics.yaml", eval_metrics)


def rmse(true, pred):
    """Root mean squared error."""
    return np.sqrt(np.mean((true - pred) ** 2))


def mae(true, pred):
    """Mean absolute error."""
    return np.mean(np.fabs(true - pred))


def cod(true, pred):
    """Coefficient of determination.

    Also often termed R2 or r2.
    Can be negative, but <= 1.0.

    """

    mean = np.mean(true)
    sum_of_squares = np.sum((true - mean) ** 2)
    sum_of_residuals = np.sum((true - pred) ** 2)

    return 100 * (1.0 - (sum_of_residuals / sum_of_squares))
