DEFAULT_PROPERTIES = {
    "energy": {"shape": (1,), "storage": "atoms.calc"},
    "forces": {"shape": ("atom", 3), "storage": "atoms.calc"},
    "stress": {"shape": (3, 3), "storage": "atoms.calc"},
}


def deduce_shape(num_structures, num_atoms, shape):
    # batch context: resolves shape spec and prepends num_structures for non-atom properties
    # see also: marathon.extra.hermes.data_source.properties.deduce_shape (same logic, per-sample context)
    # scalar per structure: no dummy dimension
    if shape == (1,):
        shape = ()

    # replace "atom" -> num_atoms, keep (optional) trailing
    if (len(shape) > 0) and (shape[0] == "atom"):
        shape = (num_atoms,) + shape[1:]
    else:
        shape = (num_structures,) + shape

    return shape


def is_per_atom(shape):
    return (len(shape) > 0) and (shape[0] == "atom")


# -- test --


def test_properties():
    # deduce_shape: scalar per structure -> shape (num_structures,)
    # (1,) removes dummy dim, then prepends num_structures
    assert deduce_shape(5, 10, (1,)) == (5,)

    # deduce_shape: per-atom property
    assert deduce_shape(5, 10, ("atom", 3)) == (10, 3)
    assert deduce_shape(5, 10, ("atom",)) == (10,)

    # deduce_shape: per-structure tensor property
    assert deduce_shape(5, 10, (3, 3)) == (5, 3, 3)
    assert deduce_shape(5, 10, (6,)) == (5, 6)

    # is_per_atom
    assert is_per_atom(("atom", 3)) is True
    assert is_per_atom(("atom",)) is True
    assert is_per_atom((3, 3)) is False
    assert is_per_atom((1,)) is False
    assert is_per_atom(()) is False


test_properties()
