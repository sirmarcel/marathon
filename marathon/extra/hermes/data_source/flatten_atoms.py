import numpy as np

from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.singlepoint import SinglePointCalculator


def flatten_atoms(atoms):
    # Atoms -> ndarray
    # missing data is NaN
    num_atoms = len(atoms)
    pbc = atoms.pbc
    positions = atoms.get_positions().flatten()
    cell = atoms.get_cell().array.flatten()
    atomic_numbers = atoms.get_atomic_numbers()

    try:
        energy = atoms.get_potential_energy()
    except PropertyNotImplementedError:
        energy = np.nan

    try:
        forces = atoms.get_forces().flatten()
    except PropertyNotImplementedError:
        forces = np.full(3 * num_atoms, np.nan)

    try:
        stress = atoms.get_stress().flatten()
    except PropertyNotImplementedError:
        stress = np.full(6, np.nan)

    flattened_data = np.concatenate(
        [
            [num_atoms],
            pbc.astype(float),
            positions,
            cell,
            atomic_numbers,
            [energy],
            forces,
            stress,
        ]
    )

    return flattened_data


def unflatten_atoms(flattened_data):
    # ndarray -> Atoms
    num_atoms = int(flattened_data[0])

    pbc = flattened_data[1:4].astype(bool)
    idx = 4

    positions = flattened_data[idx : idx + 3 * num_atoms].reshape((num_atoms, 3))
    idx += 3 * num_atoms

    cell = flattened_data[idx : idx + 9].reshape((3, 3))
    idx += 9

    atomic_numbers = flattened_data[idx : idx + num_atoms].astype(int)
    idx += num_atoms

    energy = flattened_data[idx]
    idx += 1

    forces = flattened_data[idx : idx + 3 * num_atoms].reshape((num_atoms, 3))
    idx += 3 * num_atoms

    stress = flattened_data[idx : idx + 6]

    atoms = Atoms(numbers=atomic_numbers, positions=positions, pbc=pbc, cell=cell)
    calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
    atoms.set_calculator(calc)

    return atoms


def test_flatten_unflatten_atoms():
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [1, 0, 0]])
    atoms.set_pbc([True, False, True])
    atoms.set_cell(np.eye(3) * 2)
    calc = SinglePointCalculator(
        atoms,
        energy=10.0,
        forces=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
        stress=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    atoms.set_calculator(calc)

    flattened_data = flatten_atoms(atoms)

    unflattened_atoms = unflatten_atoms(flattened_data)

    assert unflattened_atoms == atoms


test_flatten_unflatten_atoms()
