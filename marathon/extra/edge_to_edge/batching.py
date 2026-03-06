"""PET-style batches.

High-level interface to process "normal" marathon batches into
the format used by PET and other edge-to-edge transformers.

See neighborlist.py for details.

"""

import numpy as np

from collections import namedtuple

from marathon.data import batch_samples as pre_batch
from marathon.data.properties import DEFAULT_PROPERTIES

from .neighborlist import get_neighborlist

Batch = namedtuple(
    "Batch",
    (
        "atomic_numbers",  # Z_i
        "displacements",  # R_ij
        "centers",  # i
        "others",  # j
        "reverse",  # ij -> ji
        "atom_to_structure",  # map atomic_numbers to original structures
        "pair_to_structure",  # map displacements to original structures
        "structure_mask",  # False for padding
        "atom_mask",  # False for padding
        "pair_mask",  # False for padding
        "labels",
    ),
)


def batch_samples(
    samples, num_structures, num_atoms, num_neighbors, keys, properties=DEFAULT_PROPERTIES
):
    num_input_structures = len(samples)
    assert num_input_structures + 1 <= num_structures

    for s in samples:
        if not s.structure["max_neighbors"] <= num_neighbors:
            raise ValueError(
                f"samples must <={num_neighbors} neighbors, got {s.structure['max_neighbors']}"
            )

    batch = pre_batch(
        samples,
        num_atoms,
        num_atoms * num_neighbors,
        keys,
        num_structures=num_structures,
        properties=properties,
    )

    return update_batch(samples, batch, num_atoms, num_neighbors)


def update_batch(samples, batch, num_atoms, num_neighbors):
    cell_shifts = np.concatenate([s.structure["cell_shifts"] for s in samples])

    centers, others, reverse, pair_mask = get_neighborlist(
        batch.centers,
        batch.others,
        batch.pair_mask,
        num_atoms,
        num_neighbors,
        cell_shifts=cell_shifts,
    )

    # time to translate
    displacements = np.zeros((centers.shape[0], 3), dtype=batch.displacements.dtype)

    displacements[pair_mask] = batch.displacements[batch.pair_mask]

    pair_to_structure = np.ones(
        pair_mask.shape[0], dtype=batch.pair_to_structure.dtype
    ) * len(samples)
    pair_to_structure[pair_mask] = batch.pair_to_structure[batch.pair_mask]

    return Batch(
        batch.atomic_numbers,
        displacements,
        centers,
        others,
        reverse,
        batch.atom_to_structure,
        pair_to_structure,
        batch.structure_mask,
        batch.atom_mask,
        pair_mask,
        batch.labels,
    )
