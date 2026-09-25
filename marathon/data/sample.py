import numpy as np

from collections import namedtuple

from .properties import DEFAULT_PROPERTIES

Sample = namedtuple("Sample", ("structure", "labels"))


def to_sample(
    atoms,
    cutoff,
    keys=("energy", "forces"),
    inputs=(),
    float_dtype=np.float64,
    int_dtype=np.int64,
    properties=DEFAULT_PROPERTIES,
):
    """ase.Atoms -> Sample; `keys` become labels, `inputs` go into structure."""
    structure = to_structure(atoms, cutoff, float_dtype=float_dtype, int_dtype=int_dtype)

    values = read_properties(atoms, inputs, float_dtype=float_dtype, properties=properties)
    for key, value in values.items():
        if key in structure:
            raise KeyError(f"input {key} collides with structure key")
        structure[key] = value

    labels = to_labels(
        atoms, keys, float_dtype=float_dtype, int_dtype=int_dtype, properties=properties
    )

    return Sample(structure, labels)


def to_structure(atoms, cutoff, float_dtype=np.float64, int_dtype=np.int64):
    from vesin import ase_neighbor_list as neighbor_list

    structure = {}
    structure["cell"] = atoms.get_cell().array.astype(float_dtype)
    structure["positions"] = atoms.get_positions().astype(float_dtype)
    structure["atomic_numbers"] = atoms.get_atomic_numbers().astype(int_dtype)

    if atoms.pbc.any():
        i, j, D, S = neighbor_list("ijDS", atoms, cutoff)
    else:
        i, j, D = neighbor_list("ijD", atoms, cutoff)
        S = np.zeros((len(i), 3), dtype=int_dtype)
        if (structure["cell"] == 0).all():
            structure["cell"] = np.eye(3, dtype=float_dtype)

    structure["centers"] = i.astype(int_dtype)
    structure["others"] = j.astype(int_dtype)
    structure["cell_shifts"] = S.astype(float_dtype)
    structure["displacements"] = D.astype(float_dtype)
    structure["pbc"] = atoms.get_pbc()

    if len(i) > 0:
        structure["max_neighbors"] = np.unique(i, return_counts=True)[1].max()
    else:
        structure["max_neighbors"] = 0

    return structure


def to_labels(
    atoms,
    keys=("energy", "forces"),
    float_dtype=np.float64,
    int_dtype=np.int64,
    properties=DEFAULT_PROPERTIES,
):
    labels = read_properties(atoms, keys, float_dtype=float_dtype, properties=properties)
    labels["num_atoms"] = np.array(len(atoms), dtype=int_dtype)

    return labels


def read_properties(atoms, keys, float_dtype=np.float64, properties=DEFAULT_PROPERTIES):
    """Read `keys` from ase.Atoms per `properties`; role-agnostic (labels or inputs)."""
    try:
        volume = atoms.get_volume()
    except ValueError:
        volume = 1.0

    out = {}

    for key in keys:
        if key not in properties:
            raise KeyError(f"unknown key: {key}")

        # where do we find this property?
        storage = properties[key]["storage"]

        if storage == "atoms.info":
            out[key] = np.array(atoms.info[key], dtype=float_dtype)

        elif storage == "atoms.arrays":
            out[key] = np.array(atoms.arrays[key], dtype=float_dtype)

        # properties with special treatment (need to extract from calculator):
        elif storage == "atoms.calc":
            if key == "energy":
                out[key] = np.array(atoms.get_potential_energy(), dtype=float_dtype)

            elif key == "forces":
                out[key] = atoms.get_forces().astype(float_dtype)

            elif key == "stress":
                from ase.calculators.calculator import PropertyNotImplementedError

                try:
                    raw_stress = np.array(
                        [atoms.get_stress(voigt=False, include_ideal_gas=False) * volume]
                    )
                except PropertyNotImplementedError:
                    raw_stress = np.zeros((3, 3))

                # special case: assume precisely zero means missing stress
                if (raw_stress == 0.0).all():
                    raw_stress *= float("nan")

                out["stress"] = raw_stress.astype(float_dtype)

            else:
                raise ValueError(f"do not know how to extract {key} from calculator")

        else:
            raise ValueError(f"Unknown storage: {storage}")

    return out


# -- test --


def test_sample():
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    # Create test atoms with calculator
    atoms = Atoms(
        "H2O", positions=[[0, 0, 0], [0, 0, 1], [1, 0, 0]], pbc=True, cell=np.eye(3) * 5
    )
    energy = 10.0
    forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    stress = np.random.rand(6)
    calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
    atoms.calc = calc

    # Test to_labels with default (energy + forces)
    labels = to_labels(atoms)
    assert "energy" in labels
    assert "forces" in labels
    assert "stress" not in labels
    assert np.isclose(labels["energy"], energy)
    assert np.allclose(labels["forces"], forces)

    labels = to_labels(atoms, keys=["energy", "forces", "stress"])
    assert "stress" in labels

    labels = to_labels(atoms, keys=["energy"])
    assert "energy" in labels
    assert "forces" not in labels

    # Test custom properties
    atoms.info["custom_scalar"] = np.array([42.0])
    atoms.arrays["custom_peratom"] = np.random.rand(3, 2)

    custom_props = {
        **DEFAULT_PROPERTIES,
        "custom_scalar": {"shape": (1,), "storage": "atoms.info"},
        "custom_peratom": {"shape": ("atom", 2), "storage": "atoms.arrays"},
    }
    labels = to_labels(
        atoms,
        keys=["custom_scalar", "custom_peratom"],
        properties=custom_props,
    )
    assert np.isclose(labels["custom_scalar"], 42.0)
    assert labels["custom_peratom"].shape == (3, 2)

    # Test to_sample (requires cutoff)
    sample = to_sample(atoms, cutoff=2.0)
    assert "positions" in sample.structure
    assert "energy" in sample.labels

    # Test inputs: same properties, different role -> land in structure, not labels
    sample = to_sample(
        atoms,
        cutoff=2.0,
        inputs=["custom_scalar", "custom_peratom"],
        properties=custom_props,
    )
    assert np.isclose(sample.structure["custom_scalar"], 42.0)
    assert sample.structure["custom_peratom"].shape == (3, 2)
    assert "custom_scalar" not in sample.labels

    # inputs must not shadow geometry
    try:
        to_sample(
            atoms,
            cutoff=2.0,
            inputs=["positions"],
            properties={
                **custom_props,
                "positions": {"shape": ("atom", 3), "storage": "atoms.arrays"},
            },
        )
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


test_sample()
