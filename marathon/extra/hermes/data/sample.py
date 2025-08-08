import numpy as np

from collections import namedtuple

from marathon.data.sample import to_labels

Sample = namedtuple("Sample", ("graph", "labels"))
Graph = namedtuple("Graph", ("edges", "nodes", "centers", "others", "info"))


def to_sample(atoms, cutoff, energy=True, forces=True, stress=False):
    graph = to_graph(atoms, cutoff)

    labels = to_labels(
        atoms,
        energy=energy,
        forces=forces,
        stress=stress,
    )

    return Sample(graph, labels)


def to_graph(atoms, cutoff):
    from vesin import ase_neighbor_list as neighbor_list
    # todo: pin vesin version and remove the sorting step

    Z = atoms.get_atomic_numbers().astype(int)

    if atoms.pbc.all():
        i, j, D, S = neighbor_list("ijDS", atoms, cutoff)
        sort_idx = np.argsort(i)
        info = {"cell_shifts": S[sort_idx], "cell": atoms.get_cell().array, "pbc": True}
    elif atoms.pbc.any():
        # mixed pbc are not treated by vesin, we need help
        from matscipy.neighbours import neighbour_list

        i, j, D, S = neighbour_list("ijDS", atoms, cutoff)
        sort_idx = np.argsort(i)
        info = {"cell_shifts": S[sort_idx], "cell": atoms.get_cell().array, "pbc": "mixed"}
    else:
        assert not atoms.pbc.any()

        i, j, D = neighbor_list("ijD", atoms, cutoff)
        sort_idx = np.argsort(i)
        # todo: do we need to retain the cell?
        info = {"pbc": False, "cell_shifts": np.zeros((len(i), 3), dtype=int)}

    if len(i) > 0:
        info["max_neighbors"] = np.unique(i, return_counts=True)[1].max()
    else:
        info["max_neighbors"] = 0
    info["positions"] = atoms.get_positions()

    return Graph(D[sort_idx], Z, i[sort_idx], j[sort_idx], info)
