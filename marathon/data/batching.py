import numpy as np

from collections import namedtuple

from .sample import Sample

Batch = namedtuple(
    "Batch",
    (
        "nodes",  # Z_i
        "edges",  # R_ij
        "centers",  # i
        "others",  # j
        "node_to_graph",  # map nodes to original graphs
        "edge_to_graph",  # map edges to original graphs
        "graph_mask",  # False for padding
        "node_mask",  # False for padding
        "edge_mask",  # False for padding
        "labels",
    ),
)


def get_batch(samples, num_nodes, num_edges, keys, num_graphs=None):
    if num_graphs is None:
        num_graphs = len(samples) + 1
    else:
        num_input_graphs = len(samples)
        assert num_input_graphs + 1 <= num_graphs

    nodes = np.zeros(num_nodes, dtype=int)
    edges = np.zeros((num_edges, 3), dtype=float)
    centers = np.zeros(num_edges, dtype=int)
    others = np.zeros(num_edges, dtype=int)
    node_to_graph = np.zeros(num_nodes, dtype=int)
    edge_to_graph = np.zeros(num_edges, dtype=int)
    graph_mask = np.zeros(num_graphs, dtype=bool)
    node_mask = np.zeros(num_nodes, dtype=bool)
    edge_mask = np.zeros(num_edges, dtype=bool)

    labels = {}
    if "energy" in keys:
        labels["energy"] = np.zeros(num_graphs, dtype=float)
        labels["energy_mask"] = labels["energy"].astype(bool)
    if "forces" in keys:
        labels["forces"] = np.zeros((num_nodes, 3), dtype=float)
        labels["forces_mask"] = labels["forces"].astype(bool)
    if "stress" in keys:
        labels["stress"] = np.zeros((num_graphs, 3, 3), dtype=float)
        labels["stress_mask"] = labels["stress"].astype(bool)

    node_offset = 0
    edge_offset = 0
    for i, sample in enumerate(samples):
        g = sample.graph
        l = sample.labels

        num_n = sample.graph.nodes.shape[0]
        num_e = sample.graph.edges.shape[0]

        node_slice = slice(node_offset, node_offset + num_n)
        edge_slice = slice(edge_offset, edge_offset + num_e)

        nodes[node_slice] = g.nodes
        edges[edge_slice] = g.edges
        centers[edge_slice] = g.centers + node_offset
        others[edge_slice] = g.others + node_offset

        node_to_graph[node_slice] = i
        edge_to_graph[edge_slice] = i

        graph_mask[i] = True
        node_mask[node_slice] = True
        edge_mask[edge_slice] = True

        # NaNs get replaced with zero, and then
        # later masked out in the loss
        if "energy" in keys:
            if not np.isnan(l["energy"]).any():
                labels["energy"][i] = l["energy"]
                labels["energy_mask"][i] = True
            else:
                labels["energy"][i] = 0.0

        if "forces" in keys:
            if not np.isnan(l["forces"]).any():
                labels["forces"][node_slice] = l["forces"]
                labels["forces_mask"][node_slice] = True
            else:
                labels["forces"][node_slice] = 0.0

        if "stress" in keys:
            if not np.isnan(l["stress"]).any():
                labels["stress"][i] = l["stress"]
                labels["stress_mask"][i] = True
            else:
                labels["stress"][i] = 0.0

        node_offset += num_n
        edge_offset += num_e

    # now we add the padding

    nodes[node_offset:] = samples[0].graph.nodes[0]  # todo: is this ok?
    # skip edges -- already zero
    centers[edge_offset:] = node_offset
    others[edge_offset:] = node_offset

    node_to_graph[node_offset:] = num_graphs - 1
    edge_to_graph[edge_offset:] = num_graphs - 1

    # skip masks -- already False

    # skip labels -- already zero

    return Batch(
        nodes,
        edges,
        centers,
        others,
        node_to_graph,
        edge_to_graph,
        graph_mask,
        node_mask,
        edge_mask,
        labels,
    )


def next_multiple(val, n):
    return n * (1 + int(val // n))


def determine_sizes(samples, batch_size, multiple_of=2):
    n_pairs = np.array([len(sample.graph.centers) for sample in samples])
    n_nodes = np.array([len(sample.graph.nodes) for sample in samples])

    # in the worst case we get the largest ones
    max_pairs = np.sum(np.sort(n_pairs)[::-1][0:batch_size])
    max_nodes = np.sum(np.sort(n_nodes)[::-1][0:batch_size])

    # always a bit of extra padding... makes it easier to think about
    num_edges = next_multiple(max_pairs + 1, multiple_of)
    num_nodes = next_multiple(max_nodes + 1, multiple_of)

    return num_nodes, num_edges


# TESTING

Graph = namedtuple("Graph", ("edges", "nodes", "centers", "others"))

test_samples = [
    Sample(
        Graph(
            np.ones((6, 3)),
            np.array([0, 0, 0]),
            np.array([0, 0, 1, 1, 2, 2]),
            np.array([1, 2, 0, 2, 0, 1]),
        ),
        {"energy": 0.1},
    ),
    Sample(
        Graph(
            np.ones((2, 3)),
            np.array([0, 0]),
            np.array([0, 1]),
            np.array([1, 0]),
        ),
        {"energy": 0.2},
    ),
]

num_edges = 10
num_nodes = 8

test_batch = get_batch(test_samples, num_nodes, num_edges, ["energy"])

assert test_batch.edges.shape[0] == num_edges
assert test_batch.centers.shape[0] == num_edges
assert test_batch.others.shape[0] == num_edges
assert test_batch.nodes.shape[0] == num_nodes
np.testing.assert_equal(
    test_batch.edges, np.concatenate((np.ones((8, 3)), np.zeros((2, 3))))
)

np.testing.assert_equal(test_batch.centers, np.array([0, 0, 1, 1, 2, 2, 3, 4, 5, 5]))
np.testing.assert_equal(test_batch.others, np.array([1, 2, 0, 2, 0, 1, 4, 3, 5, 5]))
np.testing.assert_equal(test_batch.nodes, np.array([0, 0, 0, 0, 0, 0, 0, 0]))
np.testing.assert_equal(test_batch.node_to_graph, np.array([0, 0, 0, 1, 1, 2, 2, 2]))
np.testing.assert_equal(test_batch.edge_to_graph, np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2]))

np.testing.assert_equal(
    test_batch.node_mask, np.array([True, True, True, True, True, False, False, False])
)
np.testing.assert_equal(test_batch.graph_mask, np.array([True, True, False]))

np.testing.assert_equal(test_batch.labels["energy"], np.array([0.1, 0.2, 0.0]))
np.testing.assert_equal(test_batch.labels["energy_mask"], np.array([True, True, False]))
