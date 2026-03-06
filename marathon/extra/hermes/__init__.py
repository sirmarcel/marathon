import warnings

warnings.warn(
    "marathon.extra.hermes is deprecated, use marathon.grain instead",
    DeprecationWarning,
    stacklevel=2,
)

from .data_source import DataSource, prepare
from .pain import DataLoader, IndexSampler, ToStack
from .transforms import (
    FilterAboveNumAtoms,
    FilterEmpty,
    FilterMixedPBC,
    FilterNoop,
    FilterTooSmall,
    RandomRotation,
    ToEdgeToEdgeBatch,
    ToFixedLengthBatch,
    ToFixedShapeBatch,
    ToSample,
)


def prefetch_to_device(iterator, size):
    # same as flax, but w/o sharding
    import jax

    import collections
    import itertools

    queue = collections.deque()

    def _prefetch(x):
        return jax.device_put(x)

    def enqueue(n):
        for data in itertools.islice(iterator, n):
            queue.append(jax.tree_util.tree_map(_prefetch, data))

    enqueue(size)
    while queue:
        yield queue.popleft()
        enqueue(1)


__all__ = [
    "DataSource",
    "FilterAboveNumAtoms",
    "FilterEmpty",
    "FilterTooSmall",
    "FilterMixedPBC",
    "FilterNoop",
    "prepare",
    "ToSample",
    "ToStack",
    "ToFixedLengthBatch",
    "ToFixedShapeBatch",
    "DataLoader",
    "IndexSampler",
    "ToEdgeToEdgeBatch",
    "prefetch_to_device",
    "RandomRotation",
]
