"""Infrastructure for dense neighborlists.

This is custom batching functionality for the neighborlist format required by
the PET model (and other edge-to-edge transformers): We need both a neighborlist
that can be reshaped into `[num_nodes, num_neighbors]` and a `reverse` index
array that maps each pair `(i,j)` in the neighborlist to the corresponding
"inverse" pair `(j,i)`. The search for this is not entirely trivial, we implement
a simple and good enough approach using `numba` below.

To keep things simple, we implement this on top of the usual batching tools:
We start out with a regular "spare" neighborlist and then both pad it out to
fixed number of neighbors *and* find the reverse in one `numba` loop.

This has the advantage of being agnostic to the number of samples by design:
We already start with a "supergraph" of multiple samples, so we don't actually
see the number of samples!

"""

import numpy as np

from collections import namedtuple

import numba
from numba import types
from numba.typed import Dict

from marathon.data.batching import get_batch as pre_batch

Batch = namedtuple(
    "Batch",
    (
        "nodes",  # Z_i
        "edges",  # R_ij
        "centers",  # i
        "others",  # j
        "reverse",  # ij -> ji
        "node_to_graph",  # map nodes to original graphs
        "edge_to_graph",  # map edges to original graphs
        "graph_mask",  # False for padding
        "node_mask",  # False for padding
        "edge_mask",  # False for padding
        "labels",
    ),
)


def get_batch(samples, num_nodes, num_graphs, num_neighbors, keys):
    num_input_graphs = len(samples)
    assert num_input_graphs + 1 <= num_graphs

    for s in samples:
        if not s.graph.info["max_neighbors"] <= num_neighbors:
            raise ValueError(
                f"samples must <={num_neighbors} neighbors, got {s.graph.info['max_neighbors']}"
            )

    (
        nodes,
        edges,
        _centers,
        _others,
        node_to_graph,
        edge_to_graph,
        graph_mask,
        node_mask,
        edge_mask,
        labels,
    ) = pre_batch(
        samples, num_nodes, num_nodes * num_neighbors, keys, num_graphs=num_graphs
    )

    cell_shifts = np.concatenate([s.graph.info["cell_shifts"] for s in samples])

    centers, others, reverse = make_nls(
        _centers[edge_mask], _others[edge_mask], num_nodes, num_neighbors, S=cell_shifts
    )
    new_edge_mask = centers != num_nodes - 1

    # time to translate
    new_edges = np.zeros((centers.shape[0], 3), dtype=edges.dtype)

    new_edges[new_edge_mask] = edges[edge_mask]

    new_edge_to_graph = np.ones(new_edge_mask.shape[0], dtype=edge_to_graph.dtype) * len(
        samples
    )
    new_edge_to_graph[new_edge_mask] = edge_to_graph[edge_mask]

    return Batch(
        nodes,
        new_edges,
        centers,
        others,
        reverse,
        node_to_graph,
        new_edge_to_graph,
        graph_mask,
        node_mask,
        new_edge_mask,
        labels,
    )


def make_nls(i, j, num_nodes, num_neighbors, S=None):
    # i, j are UNPADDED neighborlists
    # num_nodes is expected to be INCLUDING at least
    #   ONE node of padding -- we will use the last one
    #   as general dummy index
    # we promise to not change the ordering of i,j in here
    # S is an array int[pair, 3] with offsets
    # (it can just be zeroes if not pbc)

    i = i.astype(int)
    j = j.astype(int)

    if S is not None:
        S = S.astype(int)
        assert len(i) == len(S)

    num_nodes = int(num_nodes)
    num_neighbors = int(num_neighbors)

    padding_value = num_nodes - 1

    idx = np.ones((num_nodes, num_neighbors), dtype=int) * padding_value
    reverse = np.ones((num_nodes, num_neighbors), dtype=int)

    reverse_dict = Dict.empty(
        key_type=types.UniTuple(types.int64, 5), value_type=types.int64
    )
    return _make_nls(i, j, num_nodes, num_neighbors, idx, reverse, reverse_dict, S)


@numba.njit
def _make_nls(i, j, num_nodes, num_neighbors, idx, reverse, reverse_dict, S):
    # note: there are no fake edges here, they are already masked out

    padding_value = num_nodes - 1

    j_offset = -1
    current_i = 0
    current_j = 0
    for p in range(len(i)):
        current_i = i[p]
        current_j = j[p]
        if p > 0:
            last_i = i[p - 1]

        if last_i == current_i:
            j_offset += 1
        elif last_i > current_i:
            raise ValueError("i is not sorted")
        else:
            j_offset = 0

        idx[current_i][j_offset] = current_j

        p_in_new_list = current_i * num_neighbors + j_offset

        shifts = S[p]
        if current_i != current_j:
            shifts = -shifts

        reverse_dict[(current_j, current_i, shifts[0], shifts[1], shifts[2])] = (
            p_in_new_list
        )

    # the reverse of all padded edges will be the first entry in the neighborlist
    # for the last node
    reverse_padding_value = padding_value * num_neighbors
    reverse = reverse * reverse_padding_value

    j_offset = 0
    current_i = 0
    current_j = 0
    for p in range(len(i)):
        current_i = i[p]
        current_j = j[p]
        if p > 0:
            last_i = i[p - 1]

            if last_i == current_i:
                j_offset += 1
            else:
                j_offset = 0

        shifts = S[p]
        reverse[current_i][j_offset] = reverse_dict[
            (current_i, current_j, shifts[0], shifts[1], shifts[2])
        ]

    j = idx.flatten()
    i = np.arange(num_nodes).repeat(num_neighbors)
    i[j == padding_value] = padding_value

    reverse = reverse.flatten()

    return i, j, reverse
