import numpy as np

from ase import Atoms

from .properties import DEFAULT_PROPERTIES, extract_from_atoms, store_in_atoms


def flatten_atoms(atoms, properties=DEFAULT_PROPERTIES):
    # Atoms -> ndarray
    # missing data is NaN
    structure = flatten_structure(atoms)
    other = flatten_properties(atoms, properties)
    return np.concatenate([structure, other])


def unflatten_atoms(flattened_data, properties=DEFAULT_PROPERTIES):
    # ndarray -> Atoms
    atoms, idx = unflatten_structure(flattened_data)
    store_in_atoms(atoms, flattened_data[idx:], properties)
    return atoms


# -- internal --


def flatten_structure(atoms):
    # Atoms -> ndarray (structure only: num_atoms, pbc, positions, cell, atomic_numbers)
    num_atoms = len(atoms)
    pbc = atoms.pbc
    positions = atoms.get_positions().flatten()
    cell = atoms.get_cell().array.flatten()
    atomic_numbers = atoms.get_atomic_numbers()

    return np.concatenate(
        [
            [num_atoms],
            pbc.astype(float),
            positions,
            cell,
            atomic_numbers,
        ]
    )


def flatten_properties(atoms, properties=DEFAULT_PROPERTIES):
    # Atoms -> ndarray (labels only, according to properties)
    # missing data is NaN
    return extract_from_atoms(atoms, properties)


def unflatten_structure(flattened_data):
    # ndarray -> (Atoms, idx)
    # returns atoms without calc, and the index where structure data ends
    num_atoms = int(flattened_data[0])

    pbc = flattened_data[1:4].astype(bool)
    idx = 4

    positions = flattened_data[idx : idx + 3 * num_atoms].reshape((num_atoms, 3))
    idx += 3 * num_atoms

    cell = flattened_data[idx : idx + 9].reshape((3, 3))
    idx += 9

    atomic_numbers = flattened_data[idx : idx + num_atoms].astype(int)
    idx += num_atoms

    atoms = Atoms(numbers=atomic_numbers, positions=positions, pbc=pbc, cell=cell)

    return atoms, idx


# -- test --


def test_flatten_unflatten_atoms():
    from ase.calculators.singlepoint import SinglePointCalculator

    # end-to-end test with custom properties from atoms.arrays and atoms.info
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [1, 0, 0]])
    atoms.set_pbc([True, False, True])
    atoms.set_cell(np.eye(3) * 2)

    # standard calculator properties
    energy = 10.0
    forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    stress = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
    atoms.calc = calc

    # custom properties: per-atom tensor in arrays, global vector in info
    bec = np.arange(27.0).reshape(3, 3, 3)  # born effective charges: (N, 3, 3)
    polarization = np.array([0.1, 0.2, 0.3])  # global property

    atoms.arrays["bec"] = bec
    atoms.info["polarization"] = polarization

    # property specs including custom properties
    properties = {
        "energy": {"shape": (1,), "storage": "atoms.calc"},
        "forces": {"shape": ("atom", 3), "storage": "atoms.calc"},
        "stress": {"shape": (6,), "storage": "atoms.calc"},
        "bec": {"shape": ("atom", 3, 3), "storage": "atoms.arrays"},
        "polarization": {"shape": (3,), "storage": "atoms.info"},
    }

    # flatten and unflatten
    flattened = flatten_atoms(atoms, properties=properties)
    result = unflatten_atoms(flattened, properties=properties)

    # verify structure (atoms == atoms checks positions, numbers, cell, pbc)
    assert result == atoms

    # explicitly verify all properties since Atoms.__eq__ doesn't check these
    assert np.isclose(result.calc.results["energy"], energy)
    assert np.allclose(result.calc.results["forces"], forces)
    assert np.allclose(result.calc.results["stress"], stress)
    assert np.allclose(result.arrays["bec"], bec)
    assert np.allclose(result.info["polarization"], polarization)


test_flatten_unflatten_atoms()
