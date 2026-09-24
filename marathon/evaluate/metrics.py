import numpy as np
import jax.numpy as jnp

from marathon import comms
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
    (`stats != None`)... R² is only computed for keys present in `stats`;
    `get_stats` omits keys without any valid labels.

    Shape inference from aux data:
    - aux[f"{key}_abs"].ndim == 1 → scalar (only batch dimension)
    - aux[f"{key}_abs"].ndim > 1 → has components (batch + component dims)

    For component properties, we compute both overall metrics (averaged
    over all components) and per-component metrics.

    """

    if stats is None and samples is not None:
        stats = get_stats(
            samples, keys=keys, normalization=normalization, properties=properties
        )
    if stats is None:
        stats = {}

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
                if key in stats:
                    metrics[key]["r2"] = 100 * (1.0 - rss / stats[key]["sum_of_squares"])

            else:
                # Has components: shape is (num_batches, *component_dims)
                num_components = np.prod(abs_data.shape[1:])

                sum_of_abs = abs_data.sum()
                rss = sq_data.sum()

                # Overall metrics (averaged over all components)
                metrics[key]["mae"] = sum_of_abs / (n * num_components)
                metrics[key]["rmse"] = jnp.sqrt(rss / (n * num_components))
                if key in stats:
                    metrics[key]["r2"] = 100 * (1.0 - rss / stats[key]["sum_of_squares"])

                # Per-component metrics
                sum_of_abs_per = abs_data.sum(axis=0)
                rss_per = sq_data.sum(axis=0)

                metrics[key]["mae_per_component"] = sum_of_abs_per / n
                metrics[key]["rmse_per_component"] = jnp.sqrt(rss_per / n)
                if key in stats:
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

    Samples with NaN or missing labels are skipped per key. Keys without
    any valid labels are omitted from the result, with a comms.warn.
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
            # _per_structure keys read from the base key
            base_key = (
                key[: -len("_per_structure")] if key.endswith("_per_structure") else key
            )

            # missing labels are treated like NaN labels
            if base_key not in l:
                continue
            item = l[base_key]
            if np.isnan(item).any():
                continue

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

    out = {}
    for key, val in tmp.items():
        if not val:
            comms.warn(f"no valid labels for '{key}', omitting from stats")
            continue

        arr = np.concatenate(val)
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
