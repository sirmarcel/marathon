# Property specs define how to extract/store properties from ase.Atoms.
#
# Properties are a dict-of-dicts:
#   {"name": {"shape": tuple, "storage": str}, ...}
#
# Shape: tuple like (1,), ("atom", 3), (3, 3)
#   - "atom" placeholder gets replaced with len(atoms)
#   - (1,) for scalar per structure
#
# Storage: "atoms.calc" | "atoms.arrays" | "atoms.info"

DEFAULT_PROPERTIES = {
    "energy": {"shape": (1,), "storage": "atoms.calc"},
    "forces": {"shape": ("atom", 3), "storage": "atoms.calc"},
    "stress": {"shape": (3, 3), "storage": "atoms.calc"},
}


def normalize_properties(properties):
    # yaml returns lists; convert to tuples
    return {
        name: {**spec, "shape": tuple(spec["shape"])} for name, spec in properties.items()
    }


def store_in_atoms(atoms, data, properties):
    # updates in place
    from ase.calculators.singlepoint import SinglePointCalculator

    offset = 0
    results = {}
    for name in sorted(properties.keys()):
        spec = properties[name]
        shape = spec["shape"]
        storage = spec["storage"]

        resolved = deduce_shape(atoms, shape)
        size = deduce_size(resolved)

        d = data[offset : offset + size]
        d = reshape(atoms, d, shape)

        if storage == "atoms.info":
            atoms.info[name] = d
        elif storage == "atoms.arrays":
            atoms.arrays[name] = d
        elif storage == "atoms.calc":
            results[name] = d
        else:
            raise ValueError(f"Unknown storage: {storage}")

        offset += size

    calc = SinglePointCalculator(atoms, **results)
    atoms.calc = calc

    return atoms


def extract_from_atoms(atoms, properties):
    import numpy as np

    data = []

    for name in sorted(properties.keys()):
        spec = properties[name]
        storage = spec["storage"]
        shape = spec["shape"]

        resolved = deduce_shape(atoms, shape)

        try:
            if storage == "atoms.info":
                d = atoms.info[name]
            elif storage == "atoms.arrays":
                d = atoms.arrays[name]
            elif storage == "atoms.calc":
                d = atoms.calc.results[name]
            else:
                raise ValueError(f"Unknown storage: {storage}")

            actual_shape = np.asarray(d).shape
            if actual_shape != resolved:
                if name == "stress":
                    d = convert_stress(d, resolved)
                else:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected {resolved}, got {actual_shape}"
                    )
        except KeyError:
            d = np.full(resolved, np.nan)

        # handle scalar values (e.g. energy)
        d = np.atleast_1d(d).flatten()

        data.append(d)

    data = np.concatenate(data)

    return data


# -- helpers --


def reshape(atoms, data, shape):
    resolved = deduce_shape(atoms, shape)
    if resolved == ():
        return data[0]
    return data.reshape(*resolved)


def deduce_shape(atoms, shape):
    # shape: tuple like (1,), ("atom", 3), (3, 3)
    # replaces "atom" placeholder with len(atoms)
    # (1,) means scalar per structure -> ()
    # see also: marathon.data.properties.deduce_shape (same logic, batch context)
    if shape == (1,):
        return ()
    if len(shape) == 0:
        return shape
    if shape[0] == "atom":
        return (len(atoms),) + shape[1:]
    return shape


def deduce_size(shape):
    from math import prod

    return prod(shape)


def convert_stress(stress, target_shape):
    from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress

    if stress.shape == (3, 3) and target_shape == (6,):
        return full_3x3_to_voigt_6_stress(stress)
    elif stress.shape == (6,) and target_shape == (3, 3):
        return voigt_6_to_full_3x3_stress(stress)
    else:
        raise ValueError(f"convert stress {stress.shape} -> {target_shape} unsupported")


# -- test --


def test_properties():
    import numpy as np

    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    # -- test deduce_shape --
    atoms3 = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [1, 0, 0]])  # 3 atoms
    atoms5 = Atoms("C5", positions=[[i, 0, 0] for i in range(5)])  # 5 atoms

    assert deduce_shape(atoms3, (1,)) == ()
    assert deduce_shape(atoms3, ("atom", 3)) == (3, 3)
    assert deduce_shape(atoms3, (3, 3)) == (3, 3)
    assert deduce_shape(atoms5, ("atom", 9, 6)) == (5, 9, 6)
    assert deduce_shape(atoms5, ("atom",)) == (5,)

    # -- test deduce_size --
    assert deduce_size((1,)) == 1
    assert deduce_size((3, 3)) == 9
    assert deduce_size((5, 9, 6)) == 270

    # -- test reshape --
    scalar_data = np.array([5.0])
    reshaped = reshape(atoms3, scalar_data, (1,))
    assert reshaped == 5.0
    assert not isinstance(reshaped, np.ndarray)

    forces_flat = np.arange(9.0)
    reshaped_forces = reshape(atoms3, forces_flat, ("atom", 3))
    assert reshaped_forces.shape == (3, 3)
    assert np.allclose(reshaped_forces[0], [0, 1, 2])

    # -- test extract_from_atoms with all storage types --
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [1, 0, 0]])
    atoms.set_pbc([True, False, True])
    atoms.set_cell(np.eye(3) * 2)

    energy = 10.0
    forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    bec = np.arange(18.0).reshape(3, 6)
    vibe = np.array([42.0, 13.37, 3.14])

    calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    atoms.calc = calc
    atoms.arrays["bec"] = bec
    atoms.info["vibe"] = vibe

    properties = {
        "energy": {"shape": (1,), "storage": "atoms.calc"},
        "forces": {"shape": ("atom", 3), "storage": "atoms.calc"},
        "bec": {"shape": ("atom", 6), "storage": "atoms.arrays"},
        "vibe": {"shape": (3,), "storage": "atoms.info"},
    }

    extracted = extract_from_atoms(atoms, properties)

    # verify total size: bec (18) + energy (1) + forces (9) + vibe (3) = 31 (sorted order)
    assert extracted.shape == (31,)

    # -- test missing data fills with NaN --
    specs_missing = {
        "energy": {"shape": (1,), "storage": "atoms.calc"},
        "nonexistent": {"shape": ("atom", 3), "storage": "atoms.arrays"},
    }
    extracted_missing = extract_from_atoms(atoms, specs_missing)
    # sorted order: energy (1), nonexistent (9)
    assert extracted_missing[0] == energy
    assert np.all(np.isnan(extracted_missing[1:10]))

    # -- test store_in_atoms --
    atoms_target = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [1, 0, 0]])

    store_in_atoms(atoms_target, extracted, properties)

    # verify calc results
    assert np.isclose(atoms_target.calc.results["energy"], energy)
    assert np.allclose(atoms_target.calc.results["forces"], forces)

    # verify arrays
    assert np.allclose(atoms_target.arrays["bec"], bec)

    # verify info
    assert np.allclose(atoms_target.info["vibe"], vibe)


test_properties()
