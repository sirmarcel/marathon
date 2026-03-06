import os
from pathlib import Path

from .batching import Batch, batch_labels, batch_samples
from .sample import Sample, to_sample, to_structure
from .sizes import determine_max_sizes
from .splits import get_splits

dataset_folder = os.environ.get("DATASETS")

if dataset_folder is not None:
    datasets = Path(dataset_folder)
else:
    datasets = None


__all__ = [
    "Sample",
    "Batch",
    "batch_samples",
    "batch_labels",
    "determine_max_sizes",
    "to_sample",
    "to_structure",
    "get_splits",
    "datasets",
]
