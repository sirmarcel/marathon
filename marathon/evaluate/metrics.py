import numpy as np
import jax.numpy as jnp

from marathon.data.properties import DEFAULT_PROPERTIES, is_per_atom
from marathon.evaluate.properties import DEFAULT_NORMALIZATION


def get_metrics_fn(
    samples=None,
    stats=None,
    keys=["energy", "forces"],
    normalization=DEFAULT_NORMALIZATION,
    properties=DEFAULT_PROPERTIES,
):
    """Get metrics function.

    A metrics function ingests the `aux` output of a loss function,
    with an additional leading dimension for batches. Everything else
    is already aggregated. Here, we just take care of the final summing.

    Metrics require statistics like variance to be available, we can
    either compute them here (`samples != None`) or somewhere else
    (`stats != None`)...

    Shape inference from aux data:
    - aux[f"{key}_abs"].ndim == 1 → scalar (only batch dimension)
    - aux[f"{key}_abs"].ndim > 1 → has components (batch + component dims)

    For component properties, we compute both overall metrics (averaged
    over all components) and per-component metrics.

    """

    if stats is None:
        if samples is not None:
            stats = get_stats(
                samples, keys=keys, normalization=normalization, properties=properties
            )

    else:
        for key in keys:
            assert key in stats

    have_stats = stats is not None

    def metrics_fn(auxs):
        # Auto-extend keys with _per_structure variants found in auxs
        actual_keys = list(keys)
        for key in keys:
            ps_key = f"{key}_per_structure"
            if f"{ps_key}_abs" in auxs and ps_key not in actual_keys:
                actual_keys.append(ps_key)

        metrics = {key: {} for key in actual_keys}
        for key in actual_keys:
            abs_data = auxs[f"{key}_abs"]
            sq_data = auxs[f"{key}_sq"]
            n = auxs[f"{key}_n"].sum()

            # ndim == 1 means scalar (batch,);
            # ndim > 1 means components
            is_scalar = abs_data.ndim == 1

            if is_scalar:
                sum_of_abs = abs_data.sum()
                rss = sq_data.sum()

                metrics[key]["mae"] = sum_of_abs / n
                metrics[key]["rmse"] = jnp.sqrt(rss / n)
                if have_stats:
                    metrics[key]["r2"] = 100 * (1.0 - rss / stats[key]["sum_of_squares"])

            else:
                # Has components: shape is (num_batches, *component_dims)
                num_components = np.prod(abs_data.shape[1:])

                sum_of_abs = abs_data.sum()
                rss = sq_data.sum()

                # Overall metrics (averaged over all components)
                metrics[key]["mae"] = sum_of_abs / (n * num_components)
                metrics[key]["rmse"] = jnp.sqrt(rss / (n * num_components))
                if have_stats:
                    metrics[key]["r2"] = 100 * (1.0 - rss / stats[key]["sum_of_squares"])

                # Per-component metrics
                sum_of_abs_per = abs_data.sum(axis=0)
                rss_per = sq_data.sum(axis=0)

                metrics[key]["mae_per_component"] = sum_of_abs_per / n
                metrics[key]["rmse_per_component"] = jnp.sqrt(rss_per / n)
                if have_stats:
                    metrics[key]["r2_per_component"] = 100 * (
                        1.0 - rss_per / stats[key]["sum_of_squares_per_component"]
                    )

        return metrics

    return metrics_fn


def get_stats(
    samples,
    keys=["energy", "forces"],
    normalization=DEFAULT_NORMALIZATION,
    properties=DEFAULT_PROPERTIES,
):
    """Compute reference statistics from samples for R² calculation.

    Auto-adds _per_structure stats for atom-normalized properties.
    """
    # Auto-extend keys with _per_structure variants for normalized properties
    actual_keys = list(keys)
    for key in keys:
        if normalization.get(key) == "atom":
            ps_key = f"{key}_per_structure"
            if ps_key not in actual_keys:
                actual_keys.append(ps_key)

    tmp = {key: [] for key in actual_keys}

    for sample in samples:
        l = sample.labels
        n = sample.labels["num_atoms"]
        for key in actual_keys:
            # For _per_structure keys, read from base key
            if key.endswith("_per_structure"):
                base_key = key[: -len("_per_structure")]
                item = l[base_key]
            else:
                item = l[key]

            # skip nans
            if np.isnan(item).any():
                continue

            # use the spec to determine per-atom vs per-structure
            base_key = (
                key[: -len("_per_structure")] if key.endswith("_per_structure") else key
            )
            per_atom = is_per_atom(properties[base_key]["shape"])

            if per_atom:
                # Per-atom property: concatenate directly
                tmp[key].append(item)
            else:
                # Per-structure property: add leading dimension
                data = np.atleast_1d(item)[None, ...]
                # Apply normalization if configured (not for _per_structure keys)
                if normalization.get(key) == "atom":
                    data = data / n
                tmp[key].append(data)

    arrays = {key: np.concatenate(val) for key, val in tmp.items()}

    out = {}
    for key in actual_keys:
        arr = arrays[key]
        stats = {
            "mean": np.mean(arr),
            "median": np.median(arr),
            "var": np.var(arr),
            "std": np.std(arr),
            "max": np.max(arr),
            "min": np.min(arr),
        }
        stats["sum_of_squares"] = np.sum((arr - stats["mean"]) ** 2)

        # Compute per-component stats if array has components (ndim > 1)
        if arr.ndim > 1:
            stats["mean_per_component"] = np.mean(arr, axis=0)
            stats["median_per_component"] = np.median(arr, axis=0)
            stats["var_per_component"] = np.var(arr, axis=0)
            stats["std_per_component"] = np.std(arr, axis=0)
            stats["max_per_component"] = np.max(arr, axis=0)
            stats["min_per_component"] = np.min(arr, axis=0)

            stats["sum_of_squares_per_component"] = np.sum(
                (arr - stats["mean_per_component"]) ** 2, axis=0
            )

        out[key] = stats

    return out


# -- test --

from collections import namedtuple

Sample = namedtuple("Sample", ("structure", "labels"))


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
        {"positions": np.zeros((Na, 3))},
        {
            "energy": dummy_energy[i],
            "forces": dummy_forces[i],
            "stress": dummy_stress[i],
            "num_atoms": Na,
        },
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
    stress = (dummy_preds["stress"][start:stop] - dummy_stress[start:stop]) / Na

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


# Test standard properties (energy, forces, stress)
np.testing.assert_allclose(
    metrics["energy"]["r2"],
    100 * cod(dummy_energy / Na, dummy_preds["energy"] / Na),
    rtol=1e-6,
)

np.testing.assert_allclose(
    metrics["energy"]["mae"],
    mae(dummy_energy / Na, dummy_preds["energy"] / Na),
    rtol=1e-6,
)

np.testing.assert_allclose(
    metrics["energy"]["rmse"],
    rmse(dummy_energy / Na, dummy_preds["energy"] / Na),
    rtol=1e-6,
)

np.testing.assert_allclose(
    metrics["forces"]["r2"],
    100 * cod(dummy_forces, dummy_preds["forces"].reshape(N, Na, 3)),
    rtol=1e-6,
)

np.testing.assert_allclose(
    metrics["forces"]["mae"],
    mae(dummy_forces, dummy_preds["forces"].reshape(N, Na, 3)),
    rtol=1e-6,
)

np.testing.assert_allclose(
    metrics["forces"]["rmse"],
    rmse(dummy_forces, dummy_preds["forces"].reshape(N, Na, 3)),
    rtol=1e-6,
)


np.testing.assert_allclose(
    metrics["stress"]["r2"],
    100 * cod(dummy_stress / Na, dummy_preds["stress"] / Na),
    rtol=1e-6,
)

np.testing.assert_allclose(
    metrics["stress"]["mae"],
    mae(dummy_stress / Na, dummy_preds["stress"] / Na),
    rtol=1e-6,
)

np.testing.assert_allclose(
    metrics["stress"]["rmse"],
    rmse(dummy_stress / Na, dummy_preds["stress"] / Na),
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
        mae(dummy_forces[..., i], dummy_preds["forces"].reshape(N, Na, 3)[..., i]),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        metrics["forces"]["rmse_per_component"][i],
        rmse(dummy_forces[..., i], dummy_preds["forces"].reshape(N, Na, 3)[..., i]),
        rtol=1e-6,
    )

for i in range(3):
    for j in range(3):
        np.testing.assert_allclose(
            metrics["stress"]["r2_per_component"][i, j],
            100 * cod(dummy_stress[..., i, j] / Na, dummy_preds["stress"][..., i, j] / Na),
            rtol=1e-6,
        )

        np.testing.assert_allclose(
            metrics["stress"]["mae_per_component"][i, j],
            mae(dummy_stress[..., i, j] / Na, dummy_preds["stress"][..., i, j] / Na),
            rtol=1e-6,
        )

        np.testing.assert_allclose(
            metrics["stress"]["rmse_per_component"][i, j],
            rmse(dummy_stress[..., i, j] / Na, dummy_preds["stress"][..., i, j] / Na),
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
    dummy_stress.mean() / Na,
    stats["stress"]["mean"],
    rtol=1e-6,
)


# -- test --

# Custom scalar (per-structure, no components, no atom-normalization)
dummy_custom_scalar = np.random.random(N)

# Custom per-structure with components, shape (4,) - e.g., quaternion
dummy_quaternion = np.random.random((N, 4))

# Custom per-atom with components, shape ("atom", 2)
dummy_peratom = np.random.random((N, Na, 2))

samples_custom = [
    Sample(
        {"positions": np.zeros((Na, 3))},
        {
            "custom_scalar": dummy_custom_scalar[i],
            "custom_quat": dummy_quaternion[i],
            "custom_peratom": dummy_peratom[i],
            "num_atoms": Na,
        },
    )
    for i in range(N)
]

keys_custom = ["custom_scalar", "custom_quat", "custom_peratom"]

custom_properties = {
    "custom_scalar": {"shape": (1,), "storage": "atoms.info"},
    "custom_quat": {"shape": (4,), "storage": "atoms.info"},
    "custom_peratom": {"shape": ("atom", 2), "storage": "atoms.arrays"},
}

# Custom properties have no default normalization (not atom-normalized)
custom_normalization = {}

metrics_fn_custom = get_metrics_fn(
    samples=samples_custom,
    keys=keys_custom,
    normalization=custom_normalization,
    properties=custom_properties,
)

# Build aux data for custom properties
dummy_preds_custom = {
    "custom_scalar": np.random.random(N),
    "custom_quat": np.random.random((N, 4)),
    "custom_peratom": np.random.random((N, Na, 2)),
}

tmp_custom = {}
for key in keys_custom:
    tmp_custom[f"{key}_abs"] = []
    tmp_custom[f"{key}_sq"] = []
    tmp_custom[f"{key}_n"] = []

for batch in range(N // batch_size):
    start = batch * batch_size
    stop = (batch + 1) * batch_size

    # custom_scalar: no normalization
    scalar_res = (
        dummy_preds_custom["custom_scalar"][start:stop] - dummy_custom_scalar[start:stop]
    )
    tmp_custom["custom_scalar_abs"].append(np.abs(scalar_res).sum())
    tmp_custom["custom_scalar_sq"].append((scalar_res**2).sum())
    tmp_custom["custom_scalar_n"].append(batch_size)

    # custom_quat: per-structure with 4 components, no normalization
    quat_res = dummy_preds_custom["custom_quat"][start:stop] - dummy_quaternion[start:stop]
    tmp_custom["custom_quat_abs"].append(np.abs(quat_res).sum(axis=0))  # shape (4,)
    tmp_custom["custom_quat_sq"].append((quat_res**2).sum(axis=0))
    tmp_custom["custom_quat_n"].append(batch_size)

    # custom_peratom: per-atom with 2 components
    peratom_res = (
        dummy_preds_custom["custom_peratom"][start:stop] - dummy_peratom[start:stop]
    )
    tmp_custom["custom_peratom_abs"].append(
        np.abs(peratom_res).sum(axis=(0, 1))
    )  # shape (2,)
    tmp_custom["custom_peratom_sq"].append((peratom_res**2).sum(axis=(0, 1)))
    tmp_custom["custom_peratom_n"].append(batch_size * Na)

auxs_custom = {key: np.stack(val) for key, val in tmp_custom.items()}

metrics_custom = metrics_fn_custom(auxs_custom)

# Test custom_scalar (no per_component)
np.testing.assert_allclose(
    metrics_custom["custom_scalar"]["mae"],
    mae(dummy_custom_scalar, dummy_preds_custom["custom_scalar"]),
    rtol=1e-6,
)
np.testing.assert_allclose(
    metrics_custom["custom_scalar"]["rmse"],
    rmse(dummy_custom_scalar, dummy_preds_custom["custom_scalar"]),
    rtol=1e-6,
)
np.testing.assert_allclose(
    metrics_custom["custom_scalar"]["r2"],
    100 * cod(dummy_custom_scalar, dummy_preds_custom["custom_scalar"]),
    rtol=1e-6,
)
assert "mae_per_component" not in metrics_custom["custom_scalar"]

# Test custom_quat (has per_component with 4 components)
np.testing.assert_allclose(
    metrics_custom["custom_quat"]["mae"],
    mae(dummy_quaternion, dummy_preds_custom["custom_quat"]),
    rtol=1e-6,
)
np.testing.assert_allclose(
    metrics_custom["custom_quat"]["rmse"],
    rmse(dummy_quaternion, dummy_preds_custom["custom_quat"]),
    rtol=1e-6,
)
np.testing.assert_allclose(
    metrics_custom["custom_quat"]["r2"],
    100 * cod(dummy_quaternion, dummy_preds_custom["custom_quat"]),
    rtol=1e-6,
)

for i in range(4):
    np.testing.assert_allclose(
        metrics_custom["custom_quat"]["mae_per_component"][i],
        mae(dummy_quaternion[..., i], dummy_preds_custom["custom_quat"][..., i]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metrics_custom["custom_quat"]["rmse_per_component"][i],
        rmse(dummy_quaternion[..., i], dummy_preds_custom["custom_quat"][..., i]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metrics_custom["custom_quat"]["r2_per_component"][i],
        100 * cod(dummy_quaternion[..., i], dummy_preds_custom["custom_quat"][..., i]),
        rtol=1e-6,
    )

# Test custom_peratom (has per_component with 2 components)
np.testing.assert_allclose(
    metrics_custom["custom_peratom"]["mae"],
    mae(dummy_peratom, dummy_preds_custom["custom_peratom"]),
    rtol=1e-6,
)
np.testing.assert_allclose(
    metrics_custom["custom_peratom"]["rmse"],
    rmse(dummy_peratom, dummy_preds_custom["custom_peratom"]),
    rtol=1e-6,
)
np.testing.assert_allclose(
    metrics_custom["custom_peratom"]["r2"],
    100 * cod(dummy_peratom, dummy_preds_custom["custom_peratom"]),
    rtol=1e-6,
)

for i in range(2):
    np.testing.assert_allclose(
        metrics_custom["custom_peratom"]["mae_per_component"][i],
        mae(dummy_peratom[..., i], dummy_preds_custom["custom_peratom"][..., i]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metrics_custom["custom_peratom"]["rmse_per_component"][i],
        rmse(dummy_peratom[..., i], dummy_preds_custom["custom_peratom"][..., i]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metrics_custom["custom_peratom"]["r2_per_component"][i],
        100 * cod(dummy_peratom[..., i], dummy_preds_custom["custom_peratom"][..., i]),
        rtol=1e-6,
    )

# Test get_stats for custom properties
stats_custom = get_stats(
    samples_custom,
    keys=keys_custom,
    normalization=custom_normalization,
    properties=custom_properties,
)
np.testing.assert_allclose(
    dummy_custom_scalar.mean(),
    stats_custom["custom_scalar"]["mean"],
    rtol=1e-6,
)
np.testing.assert_allclose(
    dummy_quaternion.mean(),
    stats_custom["custom_quat"]["mean"],
    rtol=1e-6,
)
np.testing.assert_allclose(
    dummy_peratom.mean(),
    stats_custom["custom_peratom"]["mean"],
    rtol=1e-6,
)
