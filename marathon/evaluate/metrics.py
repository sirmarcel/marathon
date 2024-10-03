import numpy as np
import jax.numpy as jnp


def get_metrics_fn(samples=None, stats=None, keys=["energy", "forces"]):
    """Get metrics function.

    A metrics function ingests the `aux` output of a loss function,
    with an additional leading dimension for batches. Everything else
    is already aggregated. Here, we just take care of the final summing.

    Metrics require statistics like variance to be available, we can
    either compute them here (`samples != None`) or somewhere else
    (`stats != None`)...

    """

    if stats is None:
        assert samples is not None
        stats = get_stats(samples, keys=keys)
    else:
        for key in keys:
            assert key in stats

    def metrics_fn(auxs):
        metrics = {key: {} for key in keys}
        for key in keys:
            sum_of_abs = auxs[f"{key}_abs"].sum()
            rss = auxs[f"{key}_sq"].sum()  # residual sum of squares
            n = auxs[f"{key}_n"].sum()

            if key == "energy":
                metrics[key]["mae"] = 1000 * sum_of_abs / n
                metrics[key]["rmse"] = 1000 * jnp.sqrt(rss / n)
                metrics[key]["r2"] = 100 * (1.0 - rss / stats[key]["sum_of_squares"])

            elif key == "forces":
                metrics[key]["mae"] = 1000 * sum_of_abs / (n * 3)
                metrics[key]["rmse"] = 1000 * jnp.sqrt(rss / (n * 3))
                metrics[key]["r2"] = 100 * (1.0 - rss / stats[key]["sum_of_squares"])

                sum_of_abs = auxs[f"{key}_abs"].sum(axis=0)
                rss = auxs[f"{key}_sq"].sum(axis=0)

                metrics[key]["mae_per_component"] = 1000 * sum_of_abs / n
                metrics[key]["rmse_per_component"] = 1000 * jnp.sqrt(rss / n)
                metrics[key]["r2_per_component"] = 100 * (
                    1.0 - rss / stats[key]["sum_of_squares_per_component"]
                )

            elif key == "stress":
                metrics[key]["mae"] = 1000 * sum_of_abs / (n * 9)
                metrics[key]["rmse"] = 1000 * jnp.sqrt(rss / (n * 9))
                metrics[key]["r2"] = 100 * (1.0 - rss / stats[key]["sum_of_squares"])

                sum_of_abs = auxs[f"{key}_abs"].sum(axis=0)
                rss = auxs[f"{key}_sq"].sum(axis=0)

                metrics[key]["mae_per_component"] = 1000 * sum_of_abs / n
                metrics[key]["rmse_per_component"] = 1000 * jnp.sqrt(rss / n)
                metrics[key]["r2_per_component"] = 100 * (
                    1.0 - rss / stats[key]["sum_of_squares_per_component"]
                )

        return metrics

    return metrics_fn


def get_stats(samples, keys=["energy", "forces"]):
    tmp = {key: [] for key in keys}

    for sample in samples:
        l = sample.labels
        n = sample.graph.nodes.shape[0]
        for key in keys:
            item = l[key]

            # skip nans
            if np.isnan(item).any():
                continue

            if key == "energy":
                tmp[key].append(item[None, ...] / n)
            elif key == "forces":
                tmp[key].append(item)
            elif key == "stress":
                tmp[key].append(item[None, ...])

    arrays = {key: np.concatenate(val) for key, val in tmp.items()}

    out = {}
    for key in keys:
        stats = {
            "mean": np.mean(arrays[key]),
            "media": np.median(arrays[key]),
            "var": np.var(arrays[key]),
            "std": np.std(arrays[key]),
            "max": np.max(arrays[key]),
            "min": np.min(arrays[key]),
        }
        stats["sum_of_squares"] = np.sum((arrays[key] - stats["mean"]) ** 2)
        if key in ["forces", "stress"]:
            stats["mean_per_component"] = np.mean(arrays[key], axis=0)
            stats["median_per_component"] = np.median(arrays[key], axis=0)
            stats["var_per_component"] = np.var(arrays[key], axis=0)
            stats["std_per_component"] = np.std(arrays[key], axis=0)
            stats["max_per_component"] = np.max(arrays[key], axis=0)
            stats["min_per_component"] = np.min(arrays[key], axis=0)

            stats["sum_of_squares_per_component"] = np.sum(
                (arrays[key] - stats["mean_per_component"]) ** 2, axis=0
            )

        out[key] = stats

    return out


# -- test --

from collections import namedtuple

Sample = namedtuple("Sample", ("graph", "labels"))
Graph = namedtuple("Graph", ("nodes"))


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

    return 1.0 - (sum_of_residuals / sum_of_squares)


N = 100
Na = 125
keys = ["energy", "forces", "stress"]

dummy_energy = np.random.random(N)
dummy_forces = np.random.random((N, Na, 3))
dummy_stress = np.random.random((N, 3, 3))

samples = [
    Sample(
        Graph(np.zeros(Na, dtype=int)),
        {"energy": dummy_energy[i], "forces": dummy_forces[i], "stress": dummy_stress[i]},
    )
    for i in range(N)
]

metrics_fn = get_metrics_fn(samples=samples, keys=keys)

dummy_preds = {
    "energy": np.random.random(N),
    "forces": np.random.random((N, Na, 3)),
    "stress": np.random.random((N, 3, 3)),
}

# okay, now we have to make fake batches for testing

batch_size = 2


tmp = {}
for key in keys:
    tmp[f"{key}_abs"] = []
    tmp[f"{key}_sq"] = []
    tmp[f"{key}_n"] = []

for batch in range(N // batch_size):
    start = batch * batch_size
    stop = (batch + 1) * batch_size

    energy = (dummy_preds["energy"][start:stop] - dummy_energy[start:stop]) / Na
    forces = dummy_preds["forces"][start:stop] - dummy_forces[start:stop]
    stress = dummy_preds["stress"][start:stop] - dummy_stress[start:stop]

    tmp["energy_abs"].append(np.abs(energy).sum())
    tmp["forces_abs"].append(np.abs(forces).sum(axis=(0, 1)))
    tmp["stress_abs"].append(np.abs(stress).sum(axis=0))

    tmp["energy_sq"].append((energy**2).sum())
    tmp["forces_sq"].append((forces**2).sum(axis=(0, 1)))
    tmp["stress_sq"].append((stress**2).sum(axis=0))

    tmp["energy_n"].append(2)
    tmp["forces_n"].append(2 * Na)
    tmp["stress_n"].append(2)

auxs = {key: np.stack(val) for key, val in tmp.items()}

metrics = metrics_fn(auxs)


np.testing.assert_allclose(
    metrics["energy"]["r2"],
    100 * cod(dummy_energy / Na, dummy_preds["energy"] / Na),
    rtol=1e-6,
)

np.testing.assert_allclose(
    metrics["energy"]["mae"],
    1000 * mae(dummy_energy / Na, dummy_preds["energy"] / Na),
    rtol=1e-6,
)

np.testing.assert_allclose(
    metrics["energy"]["rmse"],
    1000 * rmse(dummy_energy / Na, dummy_preds["energy"] / Na),
    rtol=1e-6,
)

np.testing.assert_allclose(
    metrics["forces"]["r2"],
    100 * cod(dummy_forces, dummy_preds["forces"].reshape(N, Na, 3)),
    rtol=1e-6,
)

np.testing.assert_allclose(
    metrics["forces"]["mae"],
    1000 * mae(dummy_forces, dummy_preds["forces"].reshape(N, Na, 3)),
    rtol=1e-6,
)

np.testing.assert_allclose(
    metrics["forces"]["rmse"],
    1000 * rmse(dummy_forces, dummy_preds["forces"].reshape(N, Na, 3)),
    rtol=1e-6,
)


np.testing.assert_allclose(
    metrics["stress"]["r2"],
    100 * cod(dummy_stress, dummy_preds["stress"]),
    rtol=1e-6,
)

np.testing.assert_allclose(
    metrics["stress"]["mae"],
    1000 * mae(dummy_stress, dummy_preds["stress"]),
    rtol=1e-6,
)

np.testing.assert_allclose(
    metrics["stress"]["rmse"],
    1000 * rmse(dummy_stress, dummy_preds["stress"]),
    rtol=1e-6,
)


for i in range(3):
    np.testing.assert_allclose(
        metrics["forces"]["r2_per_component"][i],
        100 * cod(dummy_forces[..., i], dummy_preds["forces"].reshape(N, Na, 3)[..., i]),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        metrics["forces"]["mae_per_component"][i],
        1000 * mae(dummy_forces[..., i], dummy_preds["forces"].reshape(N, Na, 3)[..., i]),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        metrics["forces"]["rmse_per_component"][i],
        1000 * rmse(dummy_forces[..., i], dummy_preds["forces"].reshape(N, Na, 3)[..., i]),
        rtol=1e-6,
    )

for i in range(3):
    for j in range(3):
        np.testing.assert_allclose(
            metrics["stress"]["r2_per_component"][i, j],
            100 * cod(dummy_stress[..., i, j], dummy_preds["stress"][..., i, j]),
            rtol=1e-6,
        )

        np.testing.assert_allclose(
            metrics["stress"]["mae_per_component"][i, j],
            1000 * mae(dummy_stress[..., i, j], dummy_preds["stress"][..., i, j]),
            rtol=1e-6,
        )

        np.testing.assert_allclose(
            metrics["stress"]["rmse_per_component"][i, j],
            1000 * rmse(dummy_stress[..., i, j], dummy_preds["stress"][..., i, j]),
            rtol=1e-6,
        )

stats = get_stats(samples, keys=keys)
np.testing.assert_allclose(
    dummy_energy.mean() / Na,
    stats["energy"]["mean"],
    rtol=1e-6,
)
np.testing.assert_allclose(
    dummy_forces.mean(),
    stats["forces"]["mean"],
    rtol=1e-6,
)
np.testing.assert_allclose(
    dummy_stress.mean(),
    stats["stress"]["mean"],
    rtol=1e-6,
)
