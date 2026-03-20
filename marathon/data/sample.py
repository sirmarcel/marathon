import numpy as np

from collections import namedtuple

from .properties import DEFAULT_PROPERTIES

Sample = namedtuple("Sample", ("structure", "labels"))


def to_sample(
    atoms,
    cutoff,
    keys=None,
    energy=True,
    forces=True,
    stress=False,
    float_dtype=np.float64,
    int_dtype=np.int64,
    properties=DEFAULT_PROPERTIES,
):
    """Convert an ase.Atoms (with calculator or custom properties) to a Sample."""
    structure = to_structure(atoms, cutoff, float_dtype=float_dtype, int_dtype=int_dtype)

    labels = to_labels(
        atoms,
        keys=keys,
        energy=energy,
        forces=forces,
        stress=stress,
        float_dtype=float_dtype,
        int_dtype=int_dtype,
        properties=properties,
    )

    return Sample(structure, labels)


def to_structure(atoms, cutoff, float_dtype=np.float64, int_dtype=np.int64):
    from vesin import ase_neighbor_list as neighbor_list

    structure = {}
    structure["cell"] = atoms.get_cell().array.astype(float_dtype)
    structure["positions"] = atoms.get_positions().astype(float_dtype)
    structure["atomic_numbers"] = atoms.get_atomic_numbers().astype(int_dtype)
    structure["charges"] = atoms.get_initial_charges().astype(float_dtype)

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
    keys=None,
    energy=True,
    forces=True,
    stress=False,
    float_dtype=np.float64,
    int_dtype=np.int64,
    properties=DEFAULT_PROPERTIES,
):
    if keys is None:
        keys = [
            k for k, v in [("energy", energy), ("forces", forces), ("stress", stress)] if v
        ]
    # else: keys overrides energy/forces/stress kwargs

    try:
        volume = atoms.get_volume()
    except ValueError:
        volume = 1.0

    labels = {}

    for key in keys:
        if key not in properties:
            raise KeyError(f"unknown key: {key}")

        # where do we find this property?
        storage = properties[key]["storage"]

        if storage == "atoms.info":
            labels[key] = np.array(atoms.info[key], dtype=float_dtype)

        elif storage == "atoms.arrays":
            labels[key] = np.array(atoms.arrays[key], dtype=float_dtype)

        # properties with special treatment (need to extract from calculator):
        elif storage == "atoms.calc":
            if key == "energy":
                labels[key] = np.array(atoms.get_potential_energy(), dtype=float_dtype)

            elif key == "forces":
                labels[key] = atoms.get_forces().astype(float_dtype)

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

                labels["stress"] = raw_stress.astype(float_dtype)

            else:
                raise ValueError(f"do not know how to extract {key} from calculator")

        else:
            raise ValueError(f"Unknown storage: {storage}")

    labels["num_atoms"] = np.array(len(atoms), dtype=int_dtype)

    return labels


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

    # Test to_labels with stress
    labels = to_labels(atoms, stress=True)
    assert "stress" in labels

    # Test to_labels with explicit keys (overrides kwargs)
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
        energy=False,
        forces=False,
        properties=custom_props,
    )
    assert np.isclose(labels["custom_scalar"], 42.0)
    assert labels["custom_peratom"].shape == (3, 2)

    # Test to_sample (requires cutoff)
    sample = to_sample(atoms, cutoff=2.0)
    assert "positions" in sample.structure
    assert "energy" in sample.labels


test_sample()
