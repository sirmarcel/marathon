"""Compute per-element offsets."""

import numpy as np


def elemental(samples):
    energy = np.array([s.labels["energy"] for s in samples])

    species = []
    for s in samples:
        for Z in s.graph.nodes:
            if Z not in species:
                species.append(Z)

    species = list(sorted(species))
    N_species = len(species)

    species_to_vector = {}
    for s in species:
        vec = np.zeros(N_species, dtype=np.float64)
        vec[species.index(s)] = 1.0

        species_to_vector[s] = vec

    coefficients = np.zeros((len(samples), N_species), dtype=np.float64)
    for i, sample in enumerate(samples):
        for Z in sample.graph.nodes:
            coefficients[i] += species_to_vector[Z]

    x, residuals, rank, s = np.linalg.lstsq(coefficients, energy, rcond=None)

    species_contributions = x.flatten()

    species_to_weight = {
        int(s): float(species_contributions[i]) for i, s in enumerate(species)
    }

    def energy_fn(graph):
        return np.sum([species_to_weight[Z] for Z in graph.nodes])

    return species_to_weight, energy_fn
