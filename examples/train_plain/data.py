import jax

from marathon.data import get_splits


def get_data(seed=0):
    from lj_data import steps

    n_data = len(steps)
    n_train = 150
    n_valid = 50
    n_test = 40

    key = jax.random.key(seed)

    idx_train, idx_valid, idx_test = get_splits(n_data, n_train, n_valid, n_test, key)

    data_train = [steps[i] for i in idx_train]
    data_valid = [steps[i] for i in idx_valid]
    data_test = [steps[i] for i in idx_test]

    return data_train, data_valid, data_test
