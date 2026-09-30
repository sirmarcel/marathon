import numpy as np

import pickle
import shutil
import tempfile
from pathlib import Path

import ase.db
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from marathon.grain import AseDB, prepare


def make_atoms(n_structures, seed):
    rng = np.random.default_rng(seed)
    atoms_list = []
    for i in range(n_structures):
        n_atoms = 2 + (i % 3)
        atoms = Atoms(
            numbers=[1] * (n_atoms - 1) + [8],
            positions=rng.random((n_atoms, 3)) * 5,
            cell=np.eye(3) * 10,
            pbc=True,
        )
        atoms.calc = SinglePointCalculator(
            atoms, energy=-10.0 - i, forces=rng.random((n_atoms, 3))
        )
        atoms.info["flavor"] = i % 3 - 1
        atoms.info["partial"] = rng.random(n_atoms)
        atoms_list.append(atoms)
    return atoms_list


def write_dbs(folder, suffix, sizes):
    """One db per size, returning the atoms in file-then-id order. Deletes one row
    of the first file so ids have a gap."""
    expected = []
    for k, n in enumerate(sizes):
        atoms_list = make_atoms(n + (k == 0), seed=k)
        with ase.db.connect(folder / f"{k}{suffix}", use_lock_file=False) as db:
            for atoms in atoms_list:
                db.write(
                    atoms,
                    flavor=atoms.info["flavor"],
                    data={"partial": atoms.info["partial"]},
                )
        if k == 0:
            with ase.db.connect(folder / f"{k}{suffix}", use_lock_file=False) as db:
                db.delete([2])
            del atoms_list[1]
        expected.extend(atoms_list)
    return expected


def assert_same(a, b):
    assert np.array_equal(a.numbers, b.numbers)
    assert np.allclose(a.positions, b.positions)
    assert np.allclose(a.cell, b.cell)
    assert a.get_potential_energy() == b.get_potential_energy()
    assert np.allclose(a.get_forces(), b.get_forces())
    assert a.info["flavor"] == b.info["flavor"]
    assert np.allclose(a.info["partial"], b.info["partial"])


def check_reader(folder, suffix):
    expected = write_dbs(folder, suffix, sizes=[7, 5, 4])

    reader = AseDB(folder)
    assert len(reader) == len(expected) == 16
    assert reader.offsets.tolist() == [0, 7, 12, 16]
    for i in [0, 6, 7, 11, 12, 15]:
        assert_same(reader[i], expected[i])
    assert_same(reader[np.int64(3)], expected[3])
    for i in [-1, 16]:
        with pytest.raises(IndexError):
            reader[i]

    # sequence protocol: iteration and list() work without __iter__
    assert len(list(reader)) == 16

    # pickling carries no handles; a copy reads on its own, also after close()
    assert len(pickle.dumps(reader)) < 2000
    copy = pickle.loads(pickle.dumps(reader))
    reader.close()
    assert_same(copy[13], expected[13])

    # explicit files, custom to_atoms
    reader = AseDB([folder / f"1{suffix}"], to_atoms=lambda row: row.toatoms())
    assert len(reader) == 5
    assert reader[0].info == {}
    return expected


def test_ase_db_sqlite():
    tmpdir = Path(tempfile.mkdtemp())
    try:
        check_reader(tmpdir, ".db")
    finally:
        shutil.rmtree(tmpdir)


def test_ase_db_json():
    tmpdir = Path(tempfile.mkdtemp())
    try:
        check_reader(tmpdir, ".json")
    finally:
        shutil.rmtree(tmpdir)


def test_ase_db_aselmdb():
    pytest.importorskip("ase_db_backends")
    tmpdir = Path(tempfile.mkdtemp())
    try:
        check_reader(tmpdir, ".aselmdb")
    finally:
        shutil.rmtree(tmpdir)


def test_prepare_from_reader():
    properties = {
        "energy": {"shape": (1,), "storage": "atoms.calc"},
        "forces": {"shape": ("atom", 3), "storage": "atoms.calc"},
        "flavor": {"shape": (1,), "storage": "atoms.info"},
        "partial": {"shape": ("atom",), "storage": "atoms.info"},
    }
    tmpdir = Path(tempfile.mkdtemp())
    try:
        expected = write_dbs(tmpdir, ".db", sizes=[7, 5, 4])
        reader = AseDB(tmpdir)

        prepare(expected, folder=tmpdir / "list", properties=properties)
        prepare(reader, folder=tmpdir / "seq", properties=properties)
        prepare(reader, folder=tmpdir / "par", properties=properties, num_workers=2)

        def read(folder):
            mmap = folder / "mmap"
            return {
                str(f.relative_to(mmap)): f.read_bytes()
                for f in mmap.rglob("*")
                if f.is_file()
            }

        assert read(tmpdir / "seq") == read(tmpdir / "list")
        assert read(tmpdir / "par") == read(tmpdir / "list")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    test_ase_db_sqlite()
    test_ase_db_json()
    test_prepare_from_reader()
    print("All tests passed!")
