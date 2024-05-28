import numpy as np

from collections import namedtuple

Sample = namedtuple("Sample", ("graph", "labels"))
Graph = namedtuple("Graph", ("edges", "nodes", "centers", "others"))


def to_sample(atoms, cutoff, stress=False):
    graph = to_graph(atoms, cutoff)

    labels = {
        "energy": np.array(atoms.get_potential_energy()),
        "forces": atoms.get_forces(),
    }

    if stress:
        labels["stress"] = np.array(
            [atoms.get_stress(voigt=False, include_ideal_gas=False) * atoms.get_volume()]
        )

    return Sample(graph, labels)


def to_graph(atoms, cutoff):
    from matscipy.neighbours import neighbour_list as neighbor_list

    i, j, D = neighbor_list(
        "ijD", atoms, cutoff
    )  # they follow the R_ij = R_j - R_i convention
    Z = atoms.get_atomic_numbers().astype(int)

    return Graph(D, Z, i, j)
