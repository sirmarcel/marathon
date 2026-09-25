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
        "labels",  # properties for evaluate (loss, metrics), with masks
        "inputs",  # additional properties batched alongside the graph, with masks
    ),
)


def batch_samples(
    samples,
    num_atoms,
    num_pairs,
    keys,
    inputs=(),
    num_structures=None,
    float_dtype=None,
    int_dtype=None,
    properties=DEFAULT_PROPERTIES,
):
    """Collate samples into a Batch, padding to fixed num_atoms/num_pairs with masks.

    num_atoms/num_pairs must exceed the real totals (padding needs at least one extra slot).
    `keys` are read from sample.labels, `inputs` from sample.structure.
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

    batched_inputs = batch_properties(
        [sample.structure for sample in samples],
        inputs,
        [sample.structure["atomic_numbers"].shape[0] for sample in samples],
        num_structures,
        num_atoms,
        float_dtype=float_dtype,
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
        batched_inputs,
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
    """batch_properties for label dicts, which carry num_atoms."""
    num_atoms_per_sample = [l["num_atoms"] for l in list_of_labels]

    labels = batch_properties(
        list_of_labels,
        keys,
        num_atoms_per_sample,
        num_structures,
        num_atoms,
        float_dtype=float_dtype,
        properties=properties,
    )

    labels["num_atoms"] = np.ones(num_structures, dtype=int_dtype)
    labels["num_atoms"][: len(list_of_labels)] = num_atoms_per_sample

    return labels


def batch_properties(
    dicts,
    keys,
    num_atoms_per_sample,
    num_structures,
    num_atoms,
    float_dtype=np.float64,
    properties=DEFAULT_PROPERTIES,
):
    """Stack `keys` from a list of dicts into padded arrays with NaN-aware masks."""
    out = {}

    for key in keys:
        if key not in properties:
            raise KeyError(f"unknown key: {key}")

        shape = deduce_shape(num_structures, num_atoms, properties[key]["shape"])
        out[key] = np.zeros(shape, dtype=float_dtype)
        out[key + "_mask"] = out[key].astype(bool)

    atom_offset = 0
    for i, (d, n) in enumerate(zip(dicts, num_atoms_per_sample)):
        atom_slice = slice(atom_offset, atom_offset + n)

        for key in keys:
            per_atom = is_per_atom(properties[key]["shape"])
            values = d[key]
            if not np.isnan(values).any():
                if per_atom:
                    out[key][atom_slice] = values
                    out[key + "_mask"][atom_slice] = True
                else:
                    # scalar per-structure: squeeze to avoid deprecation warning
                    out[key][i] = np.squeeze(values)
                    out[key + "_mask"][i] = True
            # else: stays zero, mask False

        atom_offset += n

    return out


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
assert test_batch.inputs == {}

# inputs: read from structure, padded like labels; NaN -> zero + mask False
test_properties = {
    "energy": {"shape": (1,), "storage": "atoms.calc"},
    "forces": {"shape": ("atom", 3), "storage": "atoms.calc"},
    "total_charge": {"shape": (1,), "storage": "atoms.info"},
    "spins": {"shape": ("atom",), "storage": "atoms.arrays"},
}
test_samples_with_inputs = [
    Sample({**s.structure, "total_charge": q, "spins": m}, s.labels)
    for s, q, m in zip(
        test_samples, [1.0, float("nan")], [np.array([1.0, -1.0, 1.0]), np.zeros(2)]
    )
]
test_batch = batch_samples(
    test_samples_with_inputs,
    num_atoms,
    num_pairs,
    ["energy"],
    inputs=["total_charge", "spins"],
    properties=test_properties,
)
np.testing.assert_equal(test_batch.inputs["total_charge"], np.array([1.0, 0.0, 0.0]))
np.testing.assert_equal(
    test_batch.inputs["total_charge_mask"], np.array([True, False, False])
)
np.testing.assert_equal(
    test_batch.inputs["spins"], np.array([1.0, -1.0, 1.0, 0, 0, 0, 0, 0])
)
np.testing.assert_equal(test_batch.inputs["spins_mask"], test_batch.atom_mask)
assert "total_charge" not in test_batch.labels
