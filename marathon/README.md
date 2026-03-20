# Technical readme

Welcome to the technical `marathon` readme! This is where we keep high-level conventions and so on. While the idea is that the subpackages are pretty independent, there are some implicit contracts that bind them together which we explain here. Individual modules make more targeted assumptions where needed; those are documented in their own `README.md` files.

## Data model

The most basic internal data object is a `Sample`, which is a `namedtuple` of `(structure, labels)`. The `structure` dict describes the atomic geometry and its neighbor graph:

```python
structure = {
    "positions": ...,      # (num_atoms, 3)
    "atomic_numbers": ..., # (num_atoms,) -- Z values
    "cell": ...,           # (3, 3)
    "charges": ...,        # (num_atoms,)
    "centers": ...,        # (num_pairs,) -- i indices of neighbor pairs
    "others": ...,         # (num_pairs,) -- j indices of neighbor pairs
    "displacements": ...,  # (num_pairs, 3) -- R_ij vectors
    "cell_shifts": ...,    # (num_pairs, 3) -- periodic image shifts
    "pbc": ...,            # (3,) -- periodic boundary conditions
    "max_neighbors": ...,  # int -- max neighbors per atom
}
```

The `labels` dict contains target properties:

```python
labels = {
    "energy": ...,    # scalar -- total energy (eV)
    "forces": ...,    # (num_atoms, 3) -- eV/Å
    "stress": ...,    # (3, 3) -- dU/dε (eV, see below)
    "num_atoms": ..., # int
}
```

The neighbor list is computed by [`vesin`](https://github.com/Luthaf/vesin) from positions and a cutoff radius. The `displacements` array contains the actual $R_{ij} = R_j - R_i + S \cdot \text{cell}$ vectors. In the typical case, these are the quantities that the model operates on and that we differentiate through to obtain forces (see `evaluate/README.md`).

The next step is a `Batch`, which is another `namedtuple` that collates one or more `Sample`s, padded and index-offset so that an MPNN can process them as one big disconnected graph:

```python
Batch(
    atomic_numbers,      # (num_atoms,) -- Z_i
    displacements,       # (num_pairs, 3) -- R_ij
    centers,             # (num_pairs,) -- i indices, offset per structure
    others,              # (num_pairs,) -- j indices, offset per structure
    atom_to_structure,   # (num_atoms,) -- which structure each atom belongs to
    pair_to_structure,   # (num_pairs,) -- which structure each pair belongs to
    structure_mask,      # (num_structures,) -- False for padding
    atom_mask,           # (num_atoms,) -- False for padding
    pair_mask,           # (num_pairs,) -- False for padding
    labels,              # dict of batched labels + masks
)
```

It is expected that models that require something less off-the-shelf implement their own `Batch` class and related infrastructure. We try, as much as possible, to be agnostic to the internals of the batch. Only parts of the code that *must* explicitly interact with it care about internals, for example some parts of `marathon.evaluate`.

## Padding and masking

Padding adds a single extra "padding structure" at the end of the batch. Its atoms fill the remaining allocated slots; its pairs point both endpoints to a single padding atom (creating self-loops that don't participate in message passing). Displacements for padding pairs are zero.

**Downstream code must handle zero displacements** ($R_{ij} = 0$). Division by $|R_{ij}|$ will blow up if you don't mask.

Three boolean masks distinguish real from padding: `structure_mask`, `atom_mask`, `pair_mask`. Additionally, each label `key` in `batch.labels` has a corresponding `key + "_mask"` (e.g. `energy_mask`, `forces_mask`) that is `True` where the label is valid and `False` where it's missing or padding. This handles the case where some structures have, say, energy labels but not stress labels. NaN values in input labels are treated as missing data and get mask `False`.

Depending on the model architecture, it may be better to always have `Batch`es with the exact same shape (so they can be stacked and processed with `vmap`/`scan`), or it may be better to batch dynamically to varying shapes. The former avoids recompilation; the latter reduces wasted computation. `marathon.grain` provides both strategies (`ToFixedShapeBatch` and `ToFixedLengthBatch`).


## Label conventions

All *per-structure* properties are stored as genuinely per-structure quantities -- not per atom and not per volume:

- **Energy** is total energy (eV), not energy/atom.
- **Stress** is the ASE stress multiplied by volume, i.e. $dU/d\varepsilon$ in energy units (eV), not pressure. If you need stress in force/area units, divide by volume.
- **Forces** are per-atom by nature (eV/Å).

The properties/normalisation system (see below) can be used to change display units and normalise by atom count for loss computation, but it implicitly relies on inputs following this convention. The pipelines process `ase.Atoms` as inputs, and expect the energy there to be total energy and the stress the standard `ase` stress definition; we handle the conversion.

Everything up to generating a `Batch` is done in pure `numpy` in configurable precision (default: float64/int64). It is up to downstream code to convert to `jax` arrays and potentially cast into other `dtypes`. Some of the infrastructure is therefore also compatible with `pytorch`.


## Properties System

Marathon supports custom properties beyond the standard `energy`, `forces`, `stress`. The key idea is that users define **one** `PROPERTIES` dict and **one** `NORMALIZATION` dict, then pass them to all relevant infrastructure--each module extracts what it needs.

### Property definition

A property has these fields (not all are required everywhere):

```python
PROPERTIES = {
    "energy": {
        "shape": (1,),                    # for data loading/batching
        "storage": "atoms.calc",          # for DataSource: where in ase.Atoms
        "report_unit": (1000, "meV"),     # for emit: (scale, unit)
        "symbol": "E",                    # for emit: short symbol
    },
    "forces": {
        "shape": ("atom", 3),
        "storage": "atoms.calc",
        "report_unit": (1000, "meV/Å"),
        "symbol": "F",
    },
    # ...
}
```

- `shape`: tuple where `"atom"` placeholder gets replaced with num_atoms. **Only leading `"atom"` dimension is supported** (e.g., `("atom", 3)` works, but `(3, "atom")` does not).
- `storage`: where to read/write from ase.Atoms: `"atoms.calc"`, `"atoms.arrays"`, `"atoms.info"`
- `report_unit`: `(scale_factor, base_unit)` for converting internal units (eV) to display units (meV)
- `symbol`: short symbol for console/log output

**Note on storage:** `atoms.calc` is reserved for `energy`, `forces`, and `stress` (these are what ASE calculators provide). Custom properties should use `atoms.arrays` (per-atom data) or `atoms.info` (per-structure data). We don't guarantee that other properties will work correctly through calculators.

### Normalization

The normalization dict specifies which properties are divided by atom count:

```python
NORMALIZATION = {
    "energy": "atom",
    "stress": "atom",
}
```

- `"atom"` means divide by num_atoms during loss/metrics and append `/atom` to units
- Properties not listed (like forces) are not modified. Setting normalization to a quantity that's of dimension `[atom,...]` will yield undefined behaviour or crash.

### Where configs are used

| Module | Uses from PROPERTIES | Uses NORMALIZATION |
|--------|---------------------|-------------------|
| `data/` batching | `shape` | - |
| `grain/DataSource` | `shape`, `storage` | - |
| `evaluate/` loss | - | yes |
| `evaluate/` metrics | - | yes |
| `emit/` logging | `report_unit`, `symbol` | yes (for unit suffix) |
| `emit/` plotting | `report_unit` | yes (for unit suffix) |

### Custom properties example

To add dipole moment as a custom property:

```python
# One unified PROPERTIES dict
PROPERTIES = {
    "energy": {
        "shape": (1,), "storage": "atoms.calc",
        "report_unit": (1000, "meV"), "symbol": "E",
    },
    "forces": {
        "shape": ("atom", 3), "storage": "atoms.calc",
        "report_unit": (1000, "meV/Å"), "symbol": "F",
    },
    "dipole": {
        "shape": (3,), "storage": "atoms.info",  # per-structure, not from calc
        "report_unit": (1, "Debye"), "symbol": "μ",
    },
}

# One unified NORMALIZATION dict
NORMALIZATION = {
    "energy": "atom",
    # dipole not listed -> no /atom suffix, no per-atom normalization
}

# Pass to all relevant places:
from marathon.emit import Txt
from marathon.emit.pretty import format_metrics
from marathon.evaluate import get_loss_fn

logger = Txt(keys=["energy", "dipole"], properties=PROPERTIES, normalization=NORMALIZATION)
loss_fn = get_loss_fn(predict_fn, weights={"energy": 1.0, "dipole": 1.0}, normalization=NORMALIZATION)
msg = format_metrics(metrics, properties=PROPERTIES, normalization=NORMALIZATION)
```

The defaults are split across modules for convenience (`data/properties.py`, `evaluate/properties.py`, `emit/properties.py`), but users can combine them into a single dict.


## Serialization conventions

Model architectures are stored as YAML "spec dicts" -- a single-entry dict mapping a fully-qualified class path to its `__init__` kwargs:

```yaml
my_package.models.MACE:
  num_features: 128
  num_layers: 3
```

`marathon.io.to_dict(module)` serialises any dataclass (including `flax.nn.Module`); `marathon.io.from_dict(spec)` reconstructs it by dynamically importing the class. `parent` and `name` fields are excluded (these are Flax-internal).

Numerical data (weights, optimizer state) uses `msgpack` via Flax's serialization. Checkpoints combine both: the architecture lives in `model.yaml`, the weights in `model.msgpack`, training state in `state.msgpack`, and metrics in `metrics.yaml`. See `emit/README.md` for details.


## Code style

Formatting is handled by `ruff` with a line length of 92. The line length is enforced by the formatter only, not the linter — so lines that can't be shortened automatically are left alone. We suppress rules that get in the way of research code: short variable names (`E741`), lambdas (`E731`), non-top-level imports (`E402`), and line length warnings (`E501`).

Import ordering groups `numpy` and `jax` before other third-party packages (configured via `isort` sections in `pyproject.toml`). Lazy imports are used for optional dependencies (`grain`, `matplotlib`, `wandb`, `numba`) and occasionally to avoid circular imports or heavy startup costs.

The code is deliberately kept **copy-and-pasteable** — individual files should be easy to pull into a script or notebook for quick hacking. This is part of the reason for inline testing, avoiding deep relative import chains, and keeping modules self-contained where possible.

More specifically:

- **Descriptive names, minimal docstrings.** Docstrings only where behaviour isn't obvious from the signature and context. Comments explain *why*, not *what*.
- **Lambdas and comprehensions** are used freely. Short closures are preferred over named helper functions when the logic is local and throwaway.
- **`namedtuple`** for data objects (`Sample`, `Batch`). Plain dicts for labels and properties configs. Dataclasses (via `flax.nn.Module` or the `frozen` helper in `utils.py`) for model components.
- **Factory functions** follow the `get_*_fn` pattern (e.g. `get_loss_fn`, `get_predict_fn`, `get_metrics_fn`). They return closures that capture configuration, keeping the hot path clean.
- **Inline tests** at the bottom of many modules, separated from production code by a `# -- test --` comment. Always use exactly this marker. These tests run on import and serve as both documentation and a basic smoke test. They're not a substitute for `pytest` but catch regressions immediately.
- **`__all__`** in `__init__.py` files defines the public API of each subpackage.
- **Private functions** are prefixed with `_` (e.g. `_huber`). No `__double_underscore` mangling.
