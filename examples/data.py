import jax

from marathon.data import get_splits


def get_data():
    from vibes.trajectory import reader

    input_trajectory = "data.son"
    steps = reader(input_trajectory)

    seed = 0
    n_data = len(steps)
    n_train = ...
    n_valid = ...
    n_test = ...

    key = jax.random.key(seed)

    idx_train, idx_valid, idx_test = get_splits(n_data, n_train, n_valid, n_test, key)

    data_train = [steps[i] for i in idx_train]
    data_valid = [steps[i] for i in idx_valid]
    data_test = [steps[i] for i in idx_test]

    return data_train, data_valid, data_test
