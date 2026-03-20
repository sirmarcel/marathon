import numpy as np
import jax
import jax.numpy as jnp

from dataclasses import dataclass, fields

__all__ = [
    "frozen",
    "masked",
    "next_size",
    "tree_stack",
    "tree_concatenate",
    "tree_split_first_dim",
    "seconds_to_string",
]


# --- dataclass helpers ---


def frozen(cls):
    # frozen dataclass that auto-freezes dict fields for hashability
    from flax.core import freeze

    cls = dataclass(frozen=True)(cls)
    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for field in fields(cls):
            v = getattr(self, field.name)
            if isinstance(v, dict):
                object.__setattr__(self, field.name, freeze(v))

    cls.__init__ = __init__
    return cls


# --- padding size strategies ---


def next_size(minimum, strategy="powers_of_2"):
    minimum = max(1, minimum)

    if isinstance(strategy, int):
        assert strategy >= minimum
        return strategy

    if not isinstance(strategy, str):
        raise ValueError(f"unknown padding size strategy {strategy}")

    if strategy == "multiples":
        return multiples(minimum)

    prefix = "powers_of_"
    if strategy.startswith(prefix):
        exponent = int(strategy[len(prefix) :])
        return next_power(minimum, exponent)

    prefix = "multiples_of_"
    if strategy.startswith(prefix):
        x = int(strategy[len(prefix) :])
        return next_multiple(minimum, x)

    raise ValueError(f"unknown padding size strategy {strategy}")


def next_multiple(val, n):
    return n * (1 + int(val // n))


def next_power(val, x):
    return int(x ** np.ceil(np.log(val) / np.log(x)))


def multiples(val):
    if val <= 32:
        return next_multiple(val, 4)

    if val <= 64:
        return next_multiple(val, 16)

    if val <= 256:
        return next_multiple(val, 64)

    if val <= 1024:
        return next_multiple(val, 256)

    if val <= 4096:
        return next_multiple(val, 1024)

    if val <= 32768:
        return next_multiple(val, 4096)

    if val <= 65536:
        return next_multiple(val, 16384)

    return next_power(val, 2)


# --- masked JAX apply ---


def masked(
    fn,
    x,
    mask,
    fn_value=0.0,
    return_value=0.0,
):
    # apply fn(x) where mask is True; where mask is False,
    # feed fn_value to fn (to avoid NaN gradients) and return return_value.
    # broadcasts mask across trailing feature dimensions if present.

    if x.ndim > 1:
        mask = mask[..., None]

    fn_value = jnp.array(fn_value, dtype=x.dtype)
    return_value = jnp.array(return_value, dtype=x.dtype)

    return jnp.where(mask, fn(jnp.where(mask, x, fn_value)), return_value)


# --- pytree helpers ---


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


# --- formatting ---


def seconds_to_string(s, precision="ms"):
    # recommended: import as s2s ;)
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


def _test():
    def t(h, m, s, ms=0):
        return ms / 1000 + s + 60 * m + 3600 * h

    assert seconds_to_string(t(3, 2, 59, 999.488), "m") == "3h3m"
    assert seconds_to_string(t(0, 6, 35, 123)) == "6m35s"
    assert seconds_to_string(t(1, 6, 35, 123)) == "1h7m"
    assert seconds_to_string(t(0, 0, 0, 123), precision="s") == "123ms"
    assert seconds_to_string(t(0, 1, 0, 123), precision="ms") == "1m123ms"
    assert seconds_to_string(t(0, 0, 0, 1.512), precision="s") == "1ms512µs"

    assert next_multiple(3, 4) == 4
    assert next_power(7, 2) == 8
    assert next_size(31, strategy="powers_of_2") == 32
    assert next_size(32, strategy="powers_of_2") == 32
    assert next_size(31, strategy="powers_of_4") == 64
    assert next_size(31, strategy="multiples_of_17") == 34
    assert next_size(29, strategy="multiples") == 32
    assert next_size(11, strategy=15) == 15


_test()
