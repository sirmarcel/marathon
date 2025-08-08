import numpy as np

import yaml


def ndarray_representer(dumper, array):
    if len(array.shape) == 0:
        # this is not actually an array
        item = array.item()
        if isinstance(item, float):
            return dumper.represent_float(item)
        elif isinstance(item, int):
            return dumper.represent_int(item)
        else:
            raise RuntimeError
    else:
        return dumper.represent_list(array.tolist())


def sequence_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(tuple, sequence_representer)
yaml.add_representer(list, sequence_representer)
yaml.add_representer(np.ndarray, ndarray_representer)
yaml.add_representer(np.float32, lambda dumper, x: dumper.represent_float(float(x)))
yaml.add_representer(np.float64, lambda dumper, x: dumper.represent_float(float(x)))
yaml.add_representer(np.integer, lambda dumper, x: dumper.represent_int(int(x)))


def write_yaml(filename, dct):
    """Save a dict as yaml.

    Formatting is done as follows:

    Dicts are NOT expressed in flowstyle (newlines for dictionary keys),
    but tuples and lists are done in flowstyle (inline).

    Args:
        filename: Path to file.
        dct: Dict to save.

    """

    with open(filename, "w") as outfile:
        yaml.dump(dct, outfile, default_flow_style=False)


def read_yaml(filename):
    with open(filename) as stream:
        dct = yaml.safe_load(stream)

    return dct
