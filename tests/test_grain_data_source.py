import numpy as np

import shutil
import tempfile
from pathlib import Path
from types import GeneratorType

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from marathon.grain import DataSource, fit_baseline, prepare


def make_fake_atoms(n_structures=10, seed=42):
    """Generate fake Atoms with energy, forces, stress, and custom properties."""
    rng = np.random.default_rng(seed)
    atoms_list = []

    for i in range(n_structures):
        # Vary structure size: 3, 4, or 5 atoms
        n_atoms = 3 + (i % 3)
        positions = rng.random((n_atoms, 3)) * 5
        # Mix of H and O for baseline testing
        numbers = [1] * (n_atoms - 1) + [8]

        atoms = Atoms(numbers=numbers, positions=positions, pbc=True, cell=np.eye(3) * 10)

        # Calculator results
        energy = -10.0 - i * 0.5
        forces = rng.random((n_atoms, 3)) * 0.1
        stress = rng.random(6) * 0.01

        calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
        atoms.calc = calc

        # Custom per-atom property in atoms.arrays
        atoms.arrays["wiggles"] = rng.random((n_atoms, 2))

        # Custom scalar in atoms.info
        atoms.info["mood"] = rng.random(3)

        atoms_list.append(atoms)

    return atoms_list


def test_data_source_roundtrip():
    """Full integration test for prepare() and DataSource."""
    tmpdir = tempfile.mkdtemp()

    try:
        # Custom property specs
        custom_specs = {
            "energy": {"shape": (1,), "storage": "atoms.calc"},
            "forces": {"shape": ("atom", 3), "storage": "atoms.calc"},
            "stress": {"shape": (6,), "storage": "atoms.calc"},
            "wiggles": {"shape": ("atom", 2), "storage": "atoms.arrays"},
            "mood": {"shape": (3,), "storage": "atoms.info"},
        }

        # Create fake data
        original_atoms = make_fake_atoms(n_structures=10)

        # Prepare dataset
        dataset_path = f"{tmpdir}/dataset"
        prepare(original_atoms, folder=dataset_path, properties=custom_specs)

        # Open DataSource (with baseline removal disabled for exact comparison)
        ds = DataSource(dataset_path, remove_baseline=False)

        # Test __len__
        assert len(ds) == 10

        # Test __getitem__(int) - single access
        atoms0 = ds[0]
        orig0 = original_atoms[0]
        assert len(atoms0) == len(orig0)
        np.testing.assert_allclose(atoms0.positions, orig0.positions)
        np.testing.assert_array_equal(
            atoms0.get_atomic_numbers(), orig0.get_atomic_numbers()
        )
        np.testing.assert_allclose(atoms0.cell.array, orig0.cell.array)
        np.testing.assert_array_equal(atoms0.pbc, orig0.pbc)
        np.testing.assert_allclose(
            atoms0.calc.results["energy"], orig0.calc.results["energy"]
        )
        np.testing.assert_allclose(
            atoms0.calc.results["forces"], orig0.calc.results["forces"]
        )
        np.testing.assert_allclose(
            atoms0.calc.results["stress"], orig0.calc.results["stress"]
        )
        np.testing.assert_allclose(atoms0.arrays["wiggles"], orig0.arrays["wiggles"])
        np.testing.assert_allclose(atoms0.info["mood"], orig0.info["mood"])

        # Test __getitem__(slice) - returns generator
        sliced = ds[2:5]
        assert isinstance(sliced, GeneratorType)
        sliced_list = list(sliced)
        assert len(sliced_list) == 3
        for i, (got, orig) in enumerate(zip(sliced_list, original_atoms[2:5])):
            np.testing.assert_allclose(
                got.positions, orig.positions, err_msg=f"slice index {i}"
            )
            np.testing.assert_allclose(
                got.calc.results["forces"],
                orig.calc.results["forces"],
                err_msg=f"slice index {i}",
            )

        # Test __iter__ - full iteration
        iterated = list(ds)
        assert len(iterated) == 10
        for i, (got, orig) in enumerate(zip(iterated, original_atoms)):
            np.testing.assert_allclose(
                got.positions, orig.positions, err_msg=f"iter index {i}"
            )

        # Test with baseline removal enabled
        ds_baseline = DataSource(dataset_path, remove_baseline=True)
        atoms_with_baseline = ds_baseline[0]
        # Energy should be different (baseline subtracted)
        # Just verify it doesn't crash and energy exists
        assert "energy" in atoms_with_baseline.calc.results

    finally:
        shutil.rmtree(tmpdir)


def test_pipeline_with_custom_properties():
    """Integration test: DataSource → ToSample → batch_samples with custom properties."""
    tmpdir = tempfile.mkdtemp()

    try:
        custom_specs = {
            "energy": {"shape": (1,), "storage": "atoms.calc"},
            "forces": {"shape": ("atom", 3), "storage": "atoms.calc"},
            "wiggles": {"shape": ("atom", 2), "storage": "atoms.arrays"},
            "mood": {"shape": (3,), "storage": "atoms.info"},
        }

        original_atoms = make_fake_atoms(n_structures=10)
        dataset_path = f"{tmpdir}/dataset"
        prepare(original_atoms, folder=dataset_path, properties=custom_specs)

        ds = DataSource(dataset_path, remove_baseline=False)

        from marathon.data import batch_samples
        from marathon.grain import ToSample

        to_sample = ToSample(
            cutoff=5.0,
            keys=("energy", "forces", "wiggles", "mood"),
            properties=custom_specs,
        )

        samples = [to_sample.map(atoms) for atoms in ds]
        assert len(samples) == 10

        # verify samples have custom properties
        assert "wiggles" in samples[0].labels
        assert "mood" in samples[0].labels

        batch = batch_samples(
            samples[:3],
            num_atoms=20,
            num_pairs=100,
            keys=("energy", "forces", "wiggles", "mood"),
            properties=custom_specs,
        )

        # verify batch has custom properties with masks
        assert "wiggles" in batch.labels
        assert "wiggles_mask" in batch.labels
        assert "mood" in batch.labels
        assert "mood_mask" in batch.labels

        # verify shapes
        assert batch.labels["wiggles"].shape[1] == 2  # per-atom, 2 features
        assert batch.labels["mood"].shape == (4, 3)  # 3 structures + 1 padding, 3 features

    finally:
        shutil.rmtree(tmpdir)


def test_info_yaml_merges_into_atoms_info():
    """info.yaml (non-numeric per-record metadata) must not clobber atoms.info properties."""
    from marathon.io import write_yaml

    properties = {
        "energy": {"shape": (1,), "storage": "atoms.calc"},
        "mood": {"shape": (3,), "storage": "atoms.info"},
    }
    atoms_list = make_fake_atoms(n_structures=3)

    tmpdir = tempfile.mkdtemp()
    try:
        prepare(atoms_list, folder=f"{tmpdir}/ds", properties=properties)
        write_yaml(f"{tmpdir}/ds/info.yaml", [{"fragment": f"frag{i}"} for i in range(3)])

        ds = DataSource(f"{tmpdir}/ds", remove_baseline=False)
        atoms = ds[1]
        assert atoms.info["fragment"] == "frag1"
        np.testing.assert_allclose(atoms.info["mood"], atoms_list[1].info["mood"])
    finally:
        shutil.rmtree(tmpdir)


def _read_folder(folder):
    from marathon.io import read_yaml

    mmap = Path(folder) / "mmap"
    files = {
        str(f.relative_to(mmap)): f.read_bytes() for f in mmap.rglob("*") if f.is_file()
    }
    baseline = Path(folder) / "baseline.yaml"
    return files, read_yaml(baseline) if baseline.exists() else None


def test_prepare_parallel_matches_sequential():
    import multiprocessing

    from marathon.grain.data_source.prepare import OffsetHelper, _prepare_parallel

    atoms_list = make_fake_atoms(n_structures=40)
    properties = {
        "energy": {"shape": (1,), "storage": "atoms.calc"},
        "forces": {"shape": ("atom", 3), "storage": "atoms.calc"},
        "wiggles": {"shape": ("atom", 2), "storage": "atoms.arrays"},
    }
    kwargs = dict(batch_size=7, samples_per_composition=5, properties=properties)

    tmpdir = Path(tempfile.mkdtemp())
    try:
        prepare(atoms_list, folder=tmpdir / "seq", **kwargs)
        prepare(atoms_list, folder=tmpdir / "par", num_workers=3, **kwargs)
        prepare(atoms_list, folder=tmpdir / "nob", num_workers=3, baseline=False, **kwargs)

        seq_files, seq_baseline = _read_folder(tmpdir / "seq")
        par_files, par_baseline = _read_folder(tmpdir / "par")
        nob_files, nob_baseline = _read_folder(tmpdir / "nob")

        assert {"data.ninja", "starts/data.ninja", "ends/data.ninja"} <= set(seq_files)
        assert par_files == seq_files
        assert nob_files == seq_files
        assert par_baseline == seq_baseline
        assert nob_baseline is None
        assert not (tmpdir / "par" / "shards").exists()

        offsetter = OffsetHelper(samples_per_composition=5)
        for atoms in atoms_list:
            offsetter(atoms)
        assert seq_baseline == offsetter.get_species_weights()

        (tmpdir / "seq" / "baseline.yaml").unlink()
        assert fit_baseline(tmpdir / "seq", samples_per_composition=5) == seq_baseline
        assert _read_folder(tmpdir / "seq")[1] == seq_baseline

        (tmpdir / "spawn").mkdir()
        (tmpdir / "spawn" / "mmap").mkdir()
        _prepare_parallel(
            atoms_list,
            tmpdir / "spawn",
            7,
            properties,
            4,
            multiprocessing.get_context("spawn"),
        )
        assert _read_folder(tmpdir / "spawn")[0] == seq_files

    finally:
        shutil.rmtree(tmpdir)


def test_compute_weights():
    from marathon.elemental import compute_weights

    weights = {1: -0.5, 6: -3.0, 8: -2.0}
    compositions = [(1, 1, 8), (6, 8, 8), (6, 1, 1, 1, 1), (8,), (1, 6, 8)]
    energy = np.array([sum(weights[Z] for Z in c) for c in compositions])

    result = compute_weights(compositions, energy)
    assert list(result) == [1, 6, 8]
    for Z, w in weights.items():
        np.testing.assert_allclose(result[Z], w)


if __name__ == "__main__":
    test_data_source_roundtrip()
    test_pipeline_with_custom_properties()
    test_info_yaml_merges_into_atoms_info()
    test_prepare_parallel_matches_sequential()
    test_compute_weights()
    print("All tests passed!")
