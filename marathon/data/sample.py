import numpy as np

from collections import namedtuple

Sample = namedtuple("Sample", ("graph", "labels"))
Graph = namedtuple("Graph", ("edges", "nodes", "centers", "others"))


def to_sample(atoms, cutoff, energy=True, forces=True, stress=False):
    graph = to_graph(atoms, cutoff)

    labels = to_labels(
        atoms,
        energy=energy,
        forces=forces,
        stress=stress,
    )

    return Sample(graph, labels)


def to_labels(atoms, energy=True, forces=True, stress=False):
    labels = {}

    if energy:
        labels["energy"] = np.array(atoms.get_potential_energy())

    if forces:
        labels["forces"] = atoms.get_forces()

    if stress:
        raw_stress = np.array(
            [atoms.get_stress(voigt=False, include_ideal_gas=False) * atoms.get_volume()]
        )

        # special case: FHI-aims + vibes return precisely zero if stress was not computed;
        # in this case we set it to nan so we can mask it out later
        if (raw_stress == 0.0).all():
            raw_stress *= float("nan")

        labels["stress"] = raw_stress

    return labels


def to_graph(atoms, cutoff):
    from vesin import ase_neighbor_list as neighbor_list

    i, j, D = neighbor_list(
        "ijD", atoms, cutoff
    )  # they follow the R_ij = R_j - R_i convention
    Z = atoms.get_atomic_numbers().astype(int)

    sort_idx = np.argsort(i)

    return Graph(D[sort_idx], Z, i[sort_idx], j[sort_idx])
