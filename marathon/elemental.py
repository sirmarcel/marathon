"""Compute per-element offsets."""

import numpy as np


def get_weights(samples):
    energy = np.array([s.labels["energy"] for s in samples])
    compositions = [s.structure["atomic_numbers"] for s in samples]

    return compute_weights(compositions, energy)


def compute_weights(compositions, energy):
    compositions = [np.asarray(c, dtype=int) for c in compositions]
    species = np.unique(np.concatenate(compositions))
    N_species = len(species)

    coefficients = np.stack(
        [
            np.bincount(np.searchsorted(species, c), minlength=N_species)
            for c in compositions
        ]
    ).astype(np.float64)

    x, residuals, rank, s = np.linalg.lstsq(coefficients, energy, rcond=None)

    species_contributions = x.flatten()

    species_to_weight = {
        int(s): float(species_contributions[i]) for i, s in enumerate(species)
    }

    return species_to_weight


def get_energy_fn(species_to_weight):
    def energy_fn(structure):
        return np.sum([species_to_weight[Z] for Z in structure["atomic_numbers"]])

    return energy_fn
