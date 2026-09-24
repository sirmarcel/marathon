import numpy as np

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress

from marathon.grain import RandomRotation


def make_atoms(stress_shape, seed=0):
    rng = np.random.default_rng(seed)
    n = 5
    cell = np.eye(3) * 8 + rng.random((3, 3))
    atoms = Atoms(
        numbers=[1] * (n - 1) + [8],
        positions=rng.random((n, 3)) * 5,
        cell=cell,
        pbc=True,
    )
    stress = rng.random((3, 3))
    stress = stress + stress.T
    if stress_shape == (6,):
        stress = full_3x3_to_voigt_6_stress(stress)
    calc = SinglePointCalculator(
        atoms, energy=-1.23, forces=rng.random((n, 3)), stress=stress
    )
    atoms.calc = calc
    return atoms


def infer_rotation(old_cell, new_cell):
    # new_cell = old_cell @ R.T  =>  R = new_cell.T @ inv(old_cell.T)
    return new_cell.T @ np.linalg.inv(old_cell.T)


def as_3x3(stress):
    return voigt_6_to_full_3x3_stress(stress) if stress.shape == (6,) else stress


def check_rotation(stress_shape):
    atoms = make_atoms(stress_shape)
    original = atoms.copy()
    original_results = {k: np.array(v) for k, v in atoms.calc.results.items()}

    rotated = RandomRotation().random_map(atoms, np.random.default_rng(1))

    # input atoms untouched (issue #5: results dict was mutated in place)
    assert np.allclose(atoms.get_positions(), original.get_positions())
    assert np.allclose(atoms.get_cell().array, original.get_cell().array)
    for k, v in original_results.items():
        assert np.allclose(atoms.calc.results[k], v), f"input {k} was mutated"

    R = infer_rotation(original.get_cell().array, rotated.get_cell().array)
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(abs(np.linalg.det(R)), 1.0)

    assert np.allclose(rotated.get_positions(), original.get_positions() @ R.T)
    assert np.allclose(rotated.calc.results["forces"], original_results["forces"] @ R.T)
    assert np.isclose(rotated.calc.results["energy"], original_results["energy"])

    stress = rotated.calc.results["stress"]
    assert stress.shape == stress_shape
    assert np.allclose(as_3x3(stress), R @ as_3x3(original_results["stress"]) @ R.T)


def test_random_rotation_voigt_stress():
    check_rotation((6,))


def test_random_rotation_full_stress():
    check_rotation((3, 3))


def test_random_rotation_without_stress():
    atoms = make_atoms((6,))
    del atoms.calc.results["stress"]
    rotated = RandomRotation().random_map(atoms, np.random.default_rng(1))
    assert "stress" not in rotated.calc.results
    assert "forces" in rotated.calc.results
