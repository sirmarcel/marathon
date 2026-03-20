import numpy as np

from collections import namedtuple

from .properties import DEFAULT_PROPERTIES, deduce_shape, is_per_atom
from .sample import Sample

# Padded, index-offset collation of multiple Samples into one disconnected graph.
Batch = namedtuple(
    "Batch",
    (
        "atomic_numbers",  # Z_i
        "displacements",  # R_ij
        "centers",  # i
        "others",  # j
        "atom_to_structure",  # map atomic_numbers to original structures
        "pair_to_structure",  # map displacements to original structures
        "structure_mask",  # False for padding
        "atom_mask",  # False for padding
        "pair_mask",  # False for padding
        "labels",
    ),
)


def batch_samples(
    samples,
    num_atoms,
    num_pairs,
    keys,
    num_structures=None,
    float_dtype=None,
    int_dtype=None,
    properties=DEFAULT_PROPERTIES,
):
    """Collate samples into a Batch, padding to fixed num_atoms/num_pairs with masks.

    num_atoms/num_pairs must exceed the real totals (padding needs at least one extra slot).
    """
    if float_dtype is None:
        float_dtype = samples[0].structure["displacements"].dtype
    if int_dtype is None:
        int_dtype = samples[0].structure["centers"].dtype

    if num_structures is None:
        num_structures = len(samples) + 1
    else:
        num_input_structures = len(samples)
        assert num_input_structures + 1 <= num_structures

    atomic_numbers = np.zeros(num_atoms, dtype=int_dtype)
    displacements = np.zeros((num_pairs, 3), dtype=float_dtype)
    centers = np.zeros(num_pairs, dtype=int_dtype)
    others = np.zeros(num_pairs, dtype=int_dtype)
    atom_to_structure = np.zeros(num_atoms, dtype=int_dtype)
    pair_to_structure = np.zeros(num_pairs, dtype=int_dtype)
    structure_mask = np.zeros(num_structures, dtype=bool)
    atom_mask = np.zeros(num_atoms, dtype=bool)
    pair_mask = np.zeros(num_pairs, dtype=bool)

    labels = batch_labels(
        [sample.labels for sample in samples],
        num_structures,
        num_atoms,
        keys,
        float_dtype=float_dtype,
        int_dtype=int_dtype,
        properties=properties,
    )

    atom_offset = 0
    pair_offset = 0
    for i, sample in enumerate(samples):
        s = sample.structure

        _n_atoms = s["atomic_numbers"].shape[0]
        _n_pairs = s["displacements"].shape[0]

        atom_slice = slice(atom_offset, atom_offset + _n_atoms)
        pair_slice = slice(pair_offset, pair_offset + _n_pairs)

        atomic_numbers[atom_slice] = s["atomic_numbers"]
        displacements[pair_slice] = s["displacements"]
        centers[pair_slice] = s["centers"] + atom_offset
        others[pair_slice] = s["others"] + atom_offset

        atom_to_structure[atom_slice] = i
        pair_to_structure[pair_slice] = i

        structure_mask[i] = True
        atom_mask[atom_slice] = True
        pair_mask[pair_slice] = True

        atom_offset += _n_atoms
        pair_offset += _n_pairs

    assert atom_offset < len(atom_mask), "no room for padding!"

    # now we add the padding

    # skip atomic_numbers -- there is no element 0
    # skip displacements -- already zero
    centers[pair_offset:] = atom_offset
    others[pair_offset:] = atom_offset

    atom_to_structure[atom_offset:] = num_structures - 1
    pair_to_structure[pair_offset:] = num_structures - 1

    # skip masks -- already False

    # skip labels -- already zero

    return Batch(
        atomic_numbers,
        displacements,
        centers,
        others,
        atom_to_structure,
        pair_to_structure,
        structure_mask,
        atom_mask,
        pair_mask,
        labels,
    )


def batch_labels(
    list_of_labels,
    num_structures,
    num_atoms,
    keys,
    float_dtype=np.float64,
    int_dtype=np.int64,
    properties=DEFAULT_PROPERTIES,
):
    """Stack label dicts into padded arrays with per-key NaN-aware masks."""
    labels = {}

    for key in keys:
        if key not in properties:
            raise KeyError(f"unknown key: {key}")

        shape = deduce_shape(num_structures, num_atoms, properties[key]["shape"])
        labels[key] = np.zeros(shape, dtype=float_dtype)
        labels[key + "_mask"] = labels[key].astype(bool)

    labels["num_atoms"] = np.ones(num_structures, dtype=int_dtype)

    atom_offset = 0
    for i, l in enumerate(list_of_labels):
        num_atoms = l["num_atoms"]
        atom_slice = slice(atom_offset, atom_offset + num_atoms)

        labels["num_atoms"][i] = num_atoms

        for key in keys:
            per_atom = is_per_atom(properties[key]["shape"])
            values = l[key]
            if not np.isnan(values).any():
                if per_atom:
                    labels[key][atom_slice] = values
                    labels[key + "_mask"][atom_slice] = True
                else:
                    # scalar per-structure: squeeze to avoid deprecation warning
                    labels[key][i] = np.squeeze(values)
                    labels[key + "_mask"][i] = True
            # else: stays zero, mask False

        atom_offset += num_atoms

    return labels


# -- test --

test_samples = [
    Sample(
        dict(
            displacements=np.ones((6, 3)),
            atomic_numbers=np.array([0, 0, 0]),
            centers=np.array([0, 0, 1, 1, 2, 2]),
            others=np.array([1, 2, 0, 2, 0, 1]),
        ),
        {
            "energy": 0.1,
            "forces": np.ones((3, 3)),
            "num_atoms": 3,
        },
    ),
    Sample(
        dict(
            displacements=np.ones((2, 3)),
            atomic_numbers=np.array([0, 0]),
            centers=np.array([0, 1]),
            others=np.array([1, 0]),
        ),
        {
            "energy": 0.2,
            "forces": np.ones((2, 3)),
            "num_atoms": 2,
        },
    ),
]

num_pairs = 10
num_atoms = 8

test_batch = batch_samples(test_samples, num_atoms, num_pairs, ["energy", "forces"])


assert test_batch.displacements.shape[0] == num_pairs
assert test_batch.centers.shape[0] == num_pairs
assert test_batch.others.shape[0] == num_pairs
assert test_batch.atomic_numbers.shape[0] == num_atoms
np.testing.assert_equal(
    test_batch.displacements, np.concatenate((np.ones((8, 3)), np.zeros((2, 3))))
)

np.testing.assert_equal(test_batch.centers, np.array([0, 0, 1, 1, 2, 2, 3, 4, 5, 5]))
np.testing.assert_equal(test_batch.others, np.array([1, 2, 0, 2, 0, 1, 4, 3, 5, 5]))
np.testing.assert_equal(test_batch.atomic_numbers, np.array([0, 0, 0, 0, 0, 0, 0, 0]))
np.testing.assert_equal(test_batch.atom_to_structure, np.array([0, 0, 0, 1, 1, 2, 2, 2]))
np.testing.assert_equal(
    test_batch.pair_to_structure, np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2])
)

np.testing.assert_equal(
    test_batch.atom_mask, np.array([True, True, True, True, True, False, False, False])
)
np.testing.assert_equal(test_batch.structure_mask, np.array([True, True, False]))

np.testing.assert_equal(test_batch.labels["energy"], np.array([0.1, 0.2, 0.0]))
np.testing.assert_equal(test_batch.labels["energy_mask"], np.array([True, True, False]))
assert test_batch.labels["forces"].shape == (8, 3)

np.testing.assert_array_equal(test_batch.labels["num_atoms"], np.array([3, 2, 1]))
