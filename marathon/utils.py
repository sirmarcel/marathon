import numpy as np
import jax
import jax.numpy as jnp


def tree_stack(trees):
    return jax.tree_util.tree_map(lambda *x: jnp.stack(x), *trees)


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

    divisor = {
        "ms": 1,
        "s": 1000,
        "m": 60 * 1000,
        "h": 60 * 60 * 1000,
    }[precision]

    ms = 1000 * s
    ms = np.round(ms / divisor) * divisor

    s, ms = np.divmod(ms, 1000)
    m, s = np.divmod(s, 60)
    h, m = np.divmod(m, 60)

    ms = int(ms)
    s = int(s)
    m = int(m)
    h = int(h)

    string = ""
    for key, value in {"h": h, "m": m, "s": s, "ms": ms}.items():
        if value != 0:
            string += f"{value}{key}"

    return string


# -- test --

h = 1
m = 2
s = 35
ms = 123

assert s_to_string(ms / 1000 + s + 60 * m + 3600 * h, "m") == "1h3m"


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
