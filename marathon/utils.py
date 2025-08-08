import numpy as np
import jax
import jax.numpy as jnp


def masked(
    fn,
    x,
    mask,
    fn_value=0.0,
    return_value=0.0,
):
    # apply fn(x) where mask is False, otherwise apply to fn_value and return return_value;
    # we broadcast the mask across a feature dimension if present

    if len(x.shape) == 1:
        mask = mask
    else:
        mask = mask[..., None]

    fn_value = jnp.array(fn_value, dtype=x.dtype)
    return_value = jnp.array(return_value, dtype=x.dtype)

    return jnp.where(mask, fn(jnp.where(mask, x, fn_value)), return_value)


def tree_stack(trees):
    return jax.tree_util.tree_map(lambda *x: jnp.stack(x), *trees)


def tree_concatenate(trees):
    return jax.tree_util.tree_map(lambda *x: jnp.concatenate(x), *trees)


def tree_split_first_dim(tree, leading):
    def fn(x):
        old_shape = x.shape
        if len(old_shape) > 1:
            new_shape = (leading, int(old_shape[0] / leading), *old_shape[1:])
        else:
            new_shape = (leading, int(old_shape[0] / leading))
        return x.reshape(*new_shape)

    return jax.tree_util.tree_map(fn, tree)


def s_to_string(s, precision="ms"):
    if s > 5 * 60:  # >2m -> drop ms
        precision = "s"
    if s > 60 * 60:  # >1h -> drop s
        precision = "m"
    if s < 60:  # < 1m -> s
        precision = "s"
    if s <= 2:  # <= 2s -> ms
        precision = "ms"
    if s <= 0.003:  # <= 3ms -> µs
        precision = "µs"

    divisor = {
        "µs": 1,
        "ms": 1000,
        "s": 1000 * 1000,
        "m": 60 * 1000 * 1000,
        "h": 60 * 60 * 1000 * 1000,
    }[precision]

    mus = 1e6 * s
    mus = np.round(mus / divisor) * divisor

    ms, mus = np.divmod(mus, 1000)
    s, ms = np.divmod(ms, 1000)
    m, s = np.divmod(s, 60)
    h, m = np.divmod(m, 60)

    mus = int(mus)
    ms = int(ms)
    s = int(s)
    m = int(m)
    h = int(h)

    string = ""
    for key, value in {"h": h, "m": m, "s": s, "ms": ms, "µs": mus}.items():
        if value != 0:
            string += f"{value}{key}"

    return string


# -- test --

h = 3
m = 2
s = 59
ms = 999.488

assert s_to_string(ms / 1000 + s + 60 * m + 3600 * h, "m") == "3h3m"

h = 0
m = 6
s = 35
ms = 123

assert s_to_string(ms / 1000 + s + 60 * m + 3600 * h) == "6m35s"

h = 1
m = 6
s = 35
ms = 123

assert s_to_string(ms / 1000 + s + 60 * m + 3600 * h) == "1h7m"

h = 0
m = 0
s = 0
ms = 123

assert s_to_string(ms / 1000 + s + 60 * m + 3600 * h, precision="s") == "123ms"


h = 0
m = 1
s = 0
ms = 123

assert s_to_string(ms / 1000 + s + 60 * m + 3600 * h, precision="ms") == "1m123ms"


h = 0
m = 0
s = 0
ms = 1.512

assert s_to_string(ms / 1000 + s + 60 * m + 3600 * h, precision="s") == "1ms512µs"
