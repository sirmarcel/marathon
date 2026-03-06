import numpy as np

import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from marathon.data.batching import batch_samples
from marathon.data.properties import DEFAULT_PROPERTIES
from marathon.data.sample import to_sample
from marathon.data.sizes import determine_max_sizes

# -- Fixtures --


@pytest.fixture
def make_atoms():
    """Factory fixture for creating test atoms."""

    def _make(pbc_mode, n_atoms=3, seed=42, custom=False):
        rng = np.random.default_rng(seed)
        positions = rng.random((n_atoms, 3)) * 5

        pbc = {
            "full": [True, True, True],
            "mixed": [True, False, True],
            "none": [False, False, False],
        }[pbc_mode]

        atoms = Atoms("H" * n_atoms, positions=positions, pbc=pbc, cell=np.eye(3) * 10)

        energy = rng.random() * 10
        forces = rng.random((n_atoms, 3))
        stress = rng.random(6)
        calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
        atoms.calc = calc

        if custom:
            atoms.info["custom_scalar"] = np.array([42.0])
            atoms.arrays["custom_peratom"] = rng.random((n_atoms, 2))

        return atoms

    return _make


@pytest.fixture
def custom_properties():
    return {
        **DEFAULT_PROPERTIES,
        "custom_scalar": {"shape": (1,), "storage": "atoms.info"},
        "custom_peratom": {"shape": ("atom", 2), "storage": "atoms.arrays"},
    }


# -- Tests --


@pytest.mark.parametrize("pbc_mode", ["full", "mixed", "none"])
@pytest.mark.parametrize("stress", [False, True])
def test_sample_creation(make_atoms, pbc_mode, stress):
    """Test to_sample with different PBC modes and stress option."""
    atoms = make_atoms(pbc_mode)
    sample = to_sample(atoms, cutoff=3.0, stress=stress)

    assert "positions" in sample.structure
    assert "atomic_numbers" in sample.structure
    assert "energy" in sample.labels
    assert "forces" in sample.labels
    if stress:
        assert "stress" in sample.labels


@pytest.mark.parametrize("pbc_mode", ["full", "mixed", "none"])
def test_batch_single_pbc(make_atoms, pbc_mode):
    """Test batching samples with same PBC mode."""
    samples = [to_sample(make_atoms(pbc_mode, seed=i), cutoff=3.0) for i in range(3)]
    num_atoms, num_pairs = determine_max_sizes(samples, batch_size=3)
    batch = batch_samples(samples, num_atoms + 1, num_pairs + 1, ["energy", "forces"])

    assert batch.structure_mask.sum() == 3
    assert "energy" in batch.labels
    assert "energy_mask" in batch.labels


@pytest.mark.parametrize("use_custom", [False, True])
def test_batch_mixed_pbc_modes(make_atoms, custom_properties, use_custom):
    """Test batching samples with ALL THREE PBC modes in one batch."""
    if use_custom:
        keys = ["energy", "forces", "custom_scalar", "custom_peratom"]
        props = custom_properties
    else:
        keys = ["energy", "forces"]
        props = DEFAULT_PROPERTIES

    samples = [
        to_sample(
            make_atoms("full", seed=0, custom=use_custom),
            cutoff=3.0,
            keys=keys,
            properties=props,
        ),
        to_sample(
            make_atoms("mixed", seed=1, custom=use_custom),
            cutoff=3.0,
            keys=keys,
            properties=props,
        ),
        to_sample(
            make_atoms("none", seed=2, custom=use_custom),
            cutoff=3.0,
            keys=keys,
            properties=props,
        ),
    ]
    num_atoms, num_pairs = determine_max_sizes(samples, batch_size=3)
    batch = batch_samples(samples, num_atoms + 1, num_pairs + 1, keys, properties=props)

    assert batch.structure_mask.sum() == 3
    assert batch.labels["energy"].shape[0] == 4  # 3 structures + 1 padding
    assert batch.labels["energy_mask"][:3].all()  # first 3 are real
    assert not batch.labels["energy_mask"][3]  # last is padding

    if use_custom:
        assert "custom_scalar" in batch.labels
        assert "custom_peratom" in batch.labels
