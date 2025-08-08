# `marathon.io`: Saving and loading things

## `dicts`: `dataclass` ↔️ `dict`

This implements functions to turn a `dataclass` into a `dict` and vice versa. This functionality is used to store and load models from disk: `flax.nn.Module`s are dataclasses, and we simply store the classname and the `__init__` arguments.

## `msgpack`: serialise efficiently

To store numerical data, like weights, we use `msgpack`. This is simply a wrapper to make the API consistent, and a small helper mixin class, `Storable`, that allows `msgpack` to save and restore objects with `.state_dict` attribute and a `.restore` method. We use this for checkpointers, which are very simple but have some state.

## `yaml`: read and write `.yaml`

Another wrapper for consistency. We also do some extra work to make `.yaml` nicely represent `ndarray` and similar objects. Note that this is not faithful: `.yaml` is not intended for serialisation of binary data! In `marathon`, `.yaml` is used for model configurations and to store metrics in checkpoints.
