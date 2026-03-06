import numpy as np

from marathon.utils import next_size


def determine_max_sizes(samples, batch_size, size_strategy="multiples"):
    n_pairs = np.array([len(sample.structure["centers"]) for sample in samples])
    n_nodes = np.array([len(sample.structure["atomic_numbers"]) for sample in samples])

    # in the worst case we get the largest ones
    max_pairs = np.sum(np.sort(n_pairs)[::-1][0:batch_size])
    max_nodes = np.sum(np.sort(n_nodes)[::-1][0:batch_size])

    # always a bit of extra padding... makes it easier to think about
    num_edges = next_size(max_pairs + 1, strategy=size_strategy)
    num_nodes = next_size(max_nodes + 1, strategy=size_strategy)

    return num_nodes, num_edges
