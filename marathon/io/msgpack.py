from flax.serialization import (
    from_bytes,
    msgpack_restore,
    register_serialization_state,
    to_bytes,
)

# -- read/write --


def write_msgpack(filename, thing):
    with open(filename, "wb") as f:
        f.write(to_bytes(thing))


def read_msgpack(filename, target=None):
    """Read msgpack; with target, restores into a matching pytree. Without target, returns raw dicts."""
    with open(filename, "rb") as f:
        data = f.read()

    if target is None:
        return msgpack_restore(data)
    else:
        return from_bytes(target, data)


# -- make object with .state_dict and .restore serializable --


def register(thing):
    register_serialization_state(
        thing, lambda x: x.state_dict, lambda x, y: x.restore(y), override=True
    )
