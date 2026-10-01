import numpy as np

from collections.abc import Sequence

import flax.linen as nn
import pytest
from flax.core import FrozenDict

from marathon.io import from_dict, read_yaml, to_dict, write_yaml
from marathon.io.dicts import coerce


class Model(nn.Module):
    features: tuple = (8, 8)
    cutoff: float = 5.0
    keys: tuple | None = ("energy", "forces")
    widths: Sequence[int] = (1, 2)
    pairs: tuple[tuple[int, int], ...] = ((0, 1),)
    args: FrozenDict = FrozenDict({"a": 1})
    properties: dict = None
    num_layers: int = 2

    def __call__(self, x):
        return x


def assert_same(a, b):
    assert type(a) is type(b), (a, b)
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        assert a == b


def test_yaml_roundtrip(tmp_path):
    model = Model(
        features=(4, 4, 4),
        cutoff=3.0,
        widths=(3, 4),
        pairs=((0, 1), (2, 3)),
        args=FrozenDict({"b": 2.0}),
    )

    write_yaml(tmp_path / "model.yaml", to_dict(model))
    restored = from_dict(read_yaml(tmp_path / "model.yaml"))

    assert_same(to_dict(restored), to_dict(model))
    assert isinstance(restored.args, FrozenDict)
    assert hash(restored) == hash(model)


def test_from_dict_coerces_user_input():
    spec = {
        "test_io.Model": {
            "features": [2],
            "cutoff": 4,
            "keys": ["energy"],
            "widths": [5, 6.0],
            "args": {"c": 3},
            "properties": None,
            "num_layers": "3",
        }
    }
    model = from_dict(spec)

    assert_same(model.features, (2,))
    assert_same(model.cutoff, 4.0)
    assert_same(model.keys, ("energy",))
    assert_same(model.widths, (5, 6))
    assert model.args == FrozenDict({"c": 3})
    assert model.properties is None
    assert_same(model.num_layers, 3)


@pytest.mark.parametrize(
    "value, hint, expected",
    [
        ([1, 2], tuple, (1, 2)),
        ([1, 2], tuple[float, ...], (1.0, 2.0)),
        ([[1, 2]], tuple[tuple[int, int], ...], ((1, 2),)),
        ((1, 2), list[int], [1, 2]),
        ("1e-3", float, 0.001),
        (np.float32(0.5), float, 0.5),
        (np.int64(3), int, 3),
        (64.0, int, 64),
        (5, str, "5"),
        ("5", int | str, "5"),
        (5, int | str, 5),
        (None, tuple | None, None),
        ([1], tuple | None, (1,)),
        ({"a": [1]}, dict[str, tuple], {"a": (1,)}),
        ("anything", callable, "anything"),
        ([1], object, [1]),
        ({"features": [1]}, Model, {"features": [1]}),
    ],
)
def test_coerce(value, hint, expected):
    assert_same(coerce(value, hint), expected)


@pytest.mark.parametrize(
    "value, hint",
    [
        (3.5, int),
        ("5 A", float),
        (True, int),
        (1, bool),
        (False, str),
        ([1, 2, 3], tuple[int, int]),
        ("ab", tuple),
        ([1], dict),
        (None, int),
    ],
)
def test_coerce_fails(value, hint):
    with pytest.raises(TypeError):
        coerce(value, hint)


def test_from_dict_error_names_field():
    with pytest.raises(TypeError, match=r"test_io.Model.pairs\[1\]\[0\]"):
        from_dict({"test_io.Model": {"pairs": [[0, 1], ["a", 2]]}})
