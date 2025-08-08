from .data_source import DataSource, prepare
from .pain import DataLoader, IndexSampler, ToStack
from .transforms import (
    FilterEmpty,
    FilterTooSmall,
    RandomRotation,
    ToDenseBatch,
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
    "FilterTooSmall",
    "prepare",
    "ToSample",
    "ToStack",
    "ToFixedLengthBatch",
    "ToFixedShapeBatch",
    "DataLoader",
    "IndexSampler",
    "FilterEmpty",
    "ToDenseBatch",
    "prefetch_to_device",
    "RandomRotation",
]
