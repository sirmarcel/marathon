import jax
import jax.numpy as jnp


def get_loss_fn(
    predict_fn,
    weights={"energy": 1.0, "forces": 1.0},
    loss_functions={"energy": "mse", "forces": "mse"},
):
    """Get a loss function."""
    assert weights.keys() == loss_functions.keys()

    keys = list(weights.keys())
    lossfs = {k: get_lossf(l) for k, l in loss_functions.items()}

    def loss_fn(params, batch):
        _, num_nodes_by_graph = jnp.unique(
            batch.node_to_graph, size=batch.graph_mask.shape[0], return_counts=True
        )

        predictions = predict_fn(params, batch)

        residuals = {}
        for key in predictions.keys():
            if key in ["energy", "forces", "stress"]:
                residuals[key] = predictions[key] - batch.labels[key]

        loss = jnp.array(0.0)
        for key in keys:
            weight = weights[key]
            lossf, need_var = lossfs[key]
            residual = residuals[key]

            if need_var:
                var = predictions[key + "_var"]
                l = lossf(residual, var)
            else:
                l = lossf(residual)

            l *= batch.labels[key + "_mask"]

            loss += weight * jnp.mean(l)

        aux = {}
        for key, residual in residuals.items():
            kv = key + "_var"
            if kv in predictions:
                var = predictions[kv]
                aux[key + "_nll"] = jnp.sum(
                    batch.labels[key + "_mask"] * nll(residual, var), axis=0
                )
                aux[key + "_crps"] = jnp.sum(
                    batch.labels[key + "_mask"] * crps(residual, var), axis=0
                )
                if key == "energy":
                    aux[key + "_var"] = jnp.sum(
                        batch.labels[key + "_mask"] * var / num_nodes_by_graph**2, axis=0
                    )
                    aux[key + "_std"] = jnp.sum(
                        jnp.sqrt(batch.labels[key + "_mask"] * var) / num_nodes_by_graph,
                        axis=0,
                    )

                else:
                    aux[key + "_var"] = jnp.sum(batch.labels[key + "_mask"] * var, axis=0)
                    aux[key + "_std"] = jnp.sum(
                        jnp.sqrt(batch.labels[key + "_mask"] * var), axis=0
                    )

        residuals["energy"] = residuals["energy"] / num_nodes_by_graph
        for key, residual in residuals.items():
            mask = batch.labels[key + "_mask"]

            aux[f"{key}_abs"] = jnp.abs(residual * mask).sum(axis=0)
            aux[f"{key}_sq"] = jax.lax.square(residual * mask).sum(axis=0)

            mask = mask.reshape(mask.shape[0], -1)
            aux[f"{key}_n"] = jnp.sum(mask.all(axis=1))

        return loss, aux

    return loss_fn


def get_lossf(s):
    if s == "mse":
        return mse, False
    elif s == "crps":
        return crps, True
    elif s == "nll":
        return nll, True
    else:
        raise ValueError


def mse(
    residual,
):
    return jax.lax.square(residual)


def crps(
    residual,
    var,
):
    """Continuous Ranked Probability Score."""

    eps = 1e-12  # clip variance to this value
    var = jnp.clip(var, a_min=eps)

    sigma = jnp.sqrt(var)

    norm_x = -residual / sigma  # needs to be true - pred
    cdf = 0.5 * (1 + jax.scipy.special.erf(norm_x / jnp.sqrt(2)))

    normalization = 1 / (jnp.sqrt(2.0 * jnp.pi))

    pdf = normalization * jnp.exp(-jax.lax.square(norm_x) / 2.0)

    crps = sigma * (norm_x * (2 * cdf - 1) + 2 * pdf - 1 / jnp.sqrt(jnp.pi))

    return crps


def nll(
    residual,
    var,
):
    """Negative Log Likelihood."""

    eps = 1e-6  # clip variance to this value
    var = jnp.clip(var, a_min=eps)

    x1 = jnp.log(var)
    x2 = jax.lax.square(residual) / var
    nll = 0.5 * (x1 + x2)

    return nll
