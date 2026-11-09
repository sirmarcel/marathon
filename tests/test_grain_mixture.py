import numpy as np

import pickle
import shutil
import tempfile
from pathlib import Path

import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from marathon.grain import DataLoader, DataSource, IndexSampler, Mixture, prepare


def make_atoms(n_structures, energy):
    atoms_list = []
    for i in range(n_structures):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1 + 0.1 * i]])
        atoms.calc = SinglePointCalculator(
            atoms, energy=energy + i, forces=np.zeros((2, 3))
        )
        atoms_list.append(atoms)
    return atoms_list


def test_mixture():
    a = make_atoms(3, energy=0.0)
    b = make_atoms(2, energy=100.0)
    mixture = Mixture([(a, 2), (b, 1)])

    assert len(mixture) == 8
    assert mixture.starts.tolist() == [0, 6]

    # A A B: full passes are the identity modulo len
    energies = [mixture[i].get_potential_energy() for i in range(8)]
    assert energies == [0, 1, 2, 0, 1, 2, 100, 101]
    assert [mixture[i].info["source_id"] for i in range(8)] == [0] * 6 + [1] * 2
    assert mixture[np.int64(7)].get_potential_energy() == 101
    assert mixture[-1].get_potential_energy() == 101
    assert [a.get_potential_energy() for a in mixture[5:]] == [2, 100, 101]
    assert len(list(mixture)) == 8

    for i in [-9, 8]:
        with pytest.raises(IndexError):
            mixture[i]

    assert repr(Mixture([(a, 1)])) != repr(Mixture([(a, 2)]))


def test_mixture_of_data_sources():
    tmpdir = Path(tempfile.mkdtemp())
    try:
        prepare(make_atoms(3, energy=0.0), folder=tmpdir / "a", baseline=False)
        prepare(make_atoms(2, energy=100.0), folder=tmpdir / "b", baseline=False)
        a = DataSource(tmpdir / "a", remove_baseline=False)
        b = DataSource(tmpdir / "b", remove_baseline=False)
        mixture = Mixture([(a, 1), (b, 2)])

        assert repr(a) in repr(mixture) and repr(b) in repr(mixture)

        copy = pickle.loads(pickle.dumps(mixture))
        assert copy[5].get_potential_energy() == 100
        assert copy[5].info["source_id"] == 1

        loader = DataLoader(
            data_source=mixture,
            sampler=IndexSampler(len(mixture), shuffle=False, num_epochs=1),
            worker_count=0,
        )
        energies = [atoms.get_potential_energy() for atoms in loader]
        assert energies == [0, 1, 2, 100, 101, 100, 101]
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    test_mixture()
    test_mixture_of_data_sources()
    print("All tests passed!")
