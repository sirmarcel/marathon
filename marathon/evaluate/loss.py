import jax
import jax.numpy as jnp

from marathon.evaluate.properties import DEFAULT_NORMALIZATION
from marathon.utils import masked


def _huber(residuals, delta):
    """Element-wise Huber loss: 0.5*x^2 for |x|<=delta, delta*(|x|-0.5*delta) otherwise."""
    abs_r = jnp.abs(residuals)
    quadratic = 0.5 * jax.lax.square(residuals)
    linear = delta * (abs_r - 0.5 * delta)
    return jnp.where(abs_r <= delta, quadratic, linear)


def compute_loss(residuals, loss_spec):
    """Compute element-wise loss given residuals and a loss spec.

    loss_spec: "mse" (default) or {"huber": {"delta": 0.5}}
    """
    from marathon.io.dicts import parse_dict

    name, kwargs = parse_dict(loss_spec, allow_stubs=True)
    if name == "mse":
        return jax.lax.square(residuals)
    elif name == "huber":
        return _huber(residuals, **kwargs)
    else:
        raise ValueError(f"unknown loss type: {name}")


def get_loss_fn(
    predict_fn,
    weights={"energy": 1.0, "forces": 1.0},
    correct_mean=False,
    normalization=DEFAULT_NORMALIZATION,
    loss="mse",
):
    """Get a loss function.

    A loss function is something that ingests a Batch and returns
    a loss (for optimisation) and summed residuals (for other metrics).

    loss: "mse" (default) or {"huber": {"delta": 0.5}}

    The assumption is that this can be vmapped or scanned across a whole
    bunch of batches at once.

    Hardcoded decisions for now:
        - We take all the means over flattened data, i.e. each
            atom contributes with the same weight, as opposed to
            averaging over structures (graphs) first, which would
            weigh smaller structures higher.
        - We expect the loss weights to take care of variance scaling.

    """

    def loss_fn(params, batch):
        predictions = predict_fn(params, batch)

        num_atoms_by_structure = batch.labels["num_atoms"]
        inverse_N = masked(
            lambda x: 1.0 / x,
            num_atoms_by_structure[:, None],
            num_atoms_by_structure > 0,
            fn_value=1e6,
        ).flatten()

        residuals = {}
        for key in predictions.keys():
            if key in batch.labels:
                residuals[key] = predictions[key] - batch.labels[key]

        # Store unnormalized residuals for _per_structure metrics
        unnormalized_residuals = {}

        for key, value in residuals.items():
            if key in normalization:
                if normalization[key] == "atom":
                    unnormalized_residuals[key] = value
                    # need to deal with same shape or extra trailing dimensions
                    residuals[key] = value * inverse_N.reshape(
                        inverse_N.shape + (1,) * (value.ndim - 1)
                    )

        total = jnp.array(0.0)
        for key, weight in weights.items():
            el = compute_loss(residuals[key], loss) * batch.labels[key + "_mask"]
            if correct_mean:
                summed = el.sum()
                factor = batch.labels[key + "_mask"].sum()
                factor = jnp.clip(factor, min=1.0)
                total += weight * (summed / factor)
            else:
                total += weight * jnp.mean(el)

        aux = {}
        for key, residual in residuals.items():
            mask = batch.labels[key + "_mask"]

            aux[f"{key}_abs"] = jnp.abs(residual * mask).sum(axis=0)
            aux[f"{key}_sq"] = jax.lax.square(residual * mask).sum(axis=0)

            # we need to count samples. so we reshape the mask to
            # [samples, flattened components] to give us the "real" samples
            mask = batch.labels[key + "_mask"]
            mask = mask.reshape(mask.shape[0], -1)
            aux[f"{key}_n"] = jnp.sum(mask.all(axis=1))

        # Add _per_structure aux for atom-normalized properties (unnormalized)
        for key, residual in unnormalized_residuals.items():
            ps_key = f"{key}_per_structure"
            mask = batch.labels[key + "_mask"]

            aux[f"{ps_key}_abs"] = jnp.abs(residual * mask).sum(axis=0)
            aux[f"{ps_key}_sq"] = jax.lax.square(residual * mask).sum(axis=0)

            mask = mask.reshape(mask.shape[0], -1)
            aux[f"{ps_key}_n"] = jnp.sum(mask.all(axis=1))

        return total, aux

    return loss_fn
