"""Compute per-element offsets."""

import numpy as np


def get_weights(samples):
    energy = np.array([s.labels["energy"] for s in samples])
    compositions = [s.graph.nodes for s in samples]

    return compute_weights(compositions, energy)


def compute_weights(compositions, energy):
    species = []
    for c in compositions:
        for Z in c:
            if Z not in species:
                species.append(Z)

    species = list(sorted(species))
    N_species = len(species)

    species_to_vector = {}
    for s in species:
        vec = np.zeros(N_species, dtype=np.float64)
        vec[species.index(s)] = 1.0

        species_to_vector[s] = vec

    coefficients = np.zeros((len(compositions), N_species), dtype=np.float64)
    for i, c in enumerate(compositions):
        for Z in c:
            coefficients[i] += species_to_vector[Z]

    x, residuals, rank, s = np.linalg.lstsq(coefficients, energy, rcond=None)

    species_contributions = x.flatten()

    species_to_weight = {
        int(s): float(species_contributions[i]) for i, s in enumerate(species)
    }

    return species_to_weight


def get_energy_fn(species_to_weight):
    def energy_fn(graph):
        return np.sum([species_to_weight[Z] for Z in graph.nodes])

    return energy_fn
