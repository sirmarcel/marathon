# shortcuts for weird pygrain imports

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
from grain.python import IndexSampler as originalIndexSampler


def IndexSampler(num_records, shard_options=None, shuffle=True, num_epochs=None, seed=0):
    if shard_options is None:
        shard_options = sharding.NoSharding()
    return originalIndexSampler(
        num_records, shard_options, shuffle=shuffle, num_epochs=num_epochs, seed=seed
    )


__all__ = [
    "DataLoader",
    "IndexSampler",
    "MapTransform",
    "Record",
    "RecordMetadata",
    "FilterTransform",
    "ToStack",
    "RandomMapTransform",
]
