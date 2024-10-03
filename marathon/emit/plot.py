import numpy as np

import matplotlib.pyplot as plt


def fig_and_ax(figsize=None):
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

    if metrics is not None:
        formatted_loss = ""
        formatted_loss += f"RMSE: {metrics[0]:.{precision}f} / "
        formatted_loss += f"MAE: {metrics[1]:.{precision}f} / "
        formatted_loss += f"R$^2$: {metrics[2]:.4f}"

        ax.text(
            0.05,
            0.95,
            formatted_loss,
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )

    return ax


def plot(outfolder, predictions, labels, metrics=None, keys=None):
    if keys is None:
        keys = labels.keys()

    for key in keys:
        if metrics is not None:
            our_metrics = [metrics[key]["rmse"], metrics[key]["mae"], metrics[key]["r2"]]
        else:
            our_metrics = None

        if key == "energy":
            fig, ax = fig_and_ax(figsize=(7, 7))
            simple_scatterplot(
                labels["energy"].flatten(),
                predictions["energy"].flatten(),
                ax=ax,
                unit="meV/atom",
                metrics=our_metrics,
            )

            fig.savefig(outfolder / "energy.png")
            plt.close(fig)

        if key == "forces":
            fig, ax = fig_and_ax(figsize=(7, 7))
            simple_scatterplot(
                labels["forces"].flatten(),
                predictions["forces"].flatten(),
                ax=ax,
                unit="meV/Å",
                metrics=our_metrics,
            )

            fig.savefig(outfolder / "forces.png")
            plt.close(fig)

        if key == "stress":
            fig, ax = fig_and_ax(figsize=(7, 7))
            simple_scatterplot(
                labels["stress"].flatten(),
                predictions["stress"].flatten(),
                ax=ax,
                precision=3,
                unit="meV",
                metrics=our_metrics,
            )
            fig.savefig(outfolder / "stress_all.png")
            plt.close(fig)

            fig, ax = fig_and_ax(figsize=(7, 7))
            simple_scatterplot(
                labels["stress"][:, 0, 0],
                predictions["stress"][:, 0, 0],
                ax=ax,
                precision=3,
                unit="meV",
                metrics=our_metrics,
            )
            fig.savefig(outfolder / "stress_0.png")
            plt.close(fig)

            fig, ax = fig_and_ax(figsize=(7, 7))
            simple_scatterplot(
                labels["stress"][:, 1, 1],
                predictions["stress"][:, 1, 1],
                ax=ax,
                precision=3,
                unit="meV",
                metrics=our_metrics,
            )
            fig.savefig(outfolder / "stress_1.png")
            plt.close(fig)

            fig, ax = fig_and_ax(figsize=(7, 7))
            simple_scatterplot(
                labels["stress"][:, 2, 2],
                predictions["stress"][:, 2, 2],
                ax=ax,
                precision=3,
                unit="meV",
                metrics=our_metrics,
            )
            fig.savefig(outfolder / "stress_2.png")
            plt.close(fig)

            fig, ax = fig_and_ax(figsize=(7, 7))
            simple_scatterplot(
                labels["stress"][:, 1, 2],
                predictions["stress"][:, 1, 2],
                ax=ax,
                precision=3,
                unit="meV",
                metrics=our_metrics,
            )
            fig.savefig(outfolder / "stress_3.png")
            plt.close(fig)

            fig, ax = fig_and_ax(figsize=(7, 7))
            simple_scatterplot(
                labels["stress"][:, 0, 2],
                predictions["stress"][:, 0, 2],
                ax=ax,
                precision=3,
                unit="meV",
                metrics=our_metrics,
            )
            fig.savefig(outfolder / "stress_4.png")
            plt.close(fig)

            fig, ax = fig_and_ax(figsize=(7, 7))
            simple_scatterplot(
                labels["stress"][:, 0, 1],
                predictions["stress"][:, 0, 1],
                ax=ax,
                precision=3,
                unit="meV",
                metrics=our_metrics,
            )
            fig.savefig(outfolder / "stress_5.png")
            plt.close(fig)
