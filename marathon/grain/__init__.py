"""marathon.grain — scalable data pipelines with grain."""

from grain._src.core import sharding
from grain.python import Batch as ToStack
from grain.python import (
    DataLoader,
    FilterTransform,
    MapTransform,
    RandomMapTransform,
    Record,
    RecordMetadata,
)
from grain.python import IndexSampler as _OriginalIndexSampler

from .data_source import DataSource, prepare
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


def IndexSampler(num_records, shard_options=None, shuffle=True, num_epochs=None, seed=0):
    """Wrapper around grain.IndexSampler that defaults to NoSharding and shuffle=True."""
    if shard_options is None:
        shard_options = sharding.NoSharding()
    return _OriginalIndexSampler(
        num_records, shard_options, shuffle=shuffle, num_epochs=num_epochs, seed=seed
    )


def prefetch_to_device(iterator, size):
    """Eagerly transfers iterator elements to JAX default device, maintaining a prefetch queue of given size."""
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
    "FilterTransform",
    "MapTransform",
    "RandomMapTransform",
    "prepare",
    "Record",
    "RecordMetadata",
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
