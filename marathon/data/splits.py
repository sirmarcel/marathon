import numpy as np
from jax import random


def get_splits(n_data, n_train, n_valid, n_test, key):
    assert n_train + n_valid + n_test <= n_data

    key_train, key_valid, key_test = random.split(key, 3)

    all_idx = np.arange(n_data)

    def distinct_sets(full, k, key):
        picked = random.choice(key, full, replace=False, shape=(k,))
        rest = np.setdiff1d(full, picked)

        return rest, picked

    rest, idx_train = distinct_sets(all_idx, n_train, key_train)
    rest, idx_valid = distinct_sets(rest, n_valid, key_valid)
    rest, idx_test = distinct_sets(rest, n_test, key_test)

    return idx_train, idx_valid, idx_test
