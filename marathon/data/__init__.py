import os
from pathlib import Path

from .batching import Batch, determine_sizes, get_batch
from .sample import Graph, Sample, to_graph, to_sample
from .splits import get_splits

dataset_folder = os.environ.get("DATASETS")

if dataset_folder is not None:
    datasets = Path(dataset_folder)
else:
    datasets = None


__all__ = [
    Sample,
    Graph,
    Batch,
    get_batch,
    determine_sizes,
    to_sample,
    to_graph,
    get_splits,
    datasets,
]
