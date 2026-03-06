## Conventions, etc.

The most basic internal data object is a `Sample`, which is simply a `namedtuple` of `(structure, labels)`, where `structure` is a dict containing information about the geometry/composition and `labels` is a `dict` with the keys `"energy"`, `"forces"`, `"stress"` (and some others).

The next step is a `Batch`, which is another `namedtuple` that contains one (or more) `samples` batched together and padded/offset such that an MPNN can deal with everything all at once. I.e. a `Batch` is a big graph that consists of disconnected subgraphs each corresponding to one original `Sample`.

Padding is accomplished by adding a fake additional graph that only connects to itself and is otherwise zeroed out or copies node labels from the first example. Downstream code needs to handle zero displacements, i.e., $r_{ij} = 0$!

Depending on the dataset and the model architecture, it may be better to always have `Batch`es with the exact same shape (so they can be stacked and processed with `vmap`), or it may be better to batch dynamically to some set of shapes. This induces recompiles, but may be acceptable to reduce wasted computation.

On the `labels` side, all *per structure* properties are expected to be actually *per structure* -- not per atom and not per volume. This means that, internally, energy is total energy and stress is `d U / d strain` with no normalisation. The properties/normalization system (see below) can be used to change this, but it implicitly relies on inputs following this convention. The pipelines process `ase.Atoms` as inputs, and expect the energy there to be total energy and the stress the "normal" `ase` stress definition; we handle the converting.

Note: We do everything up to generating a `Batch` in pure `numpy` in (configurable) precision. It is up to downstream code to convert to `jax` arrays and potentially cast into other `dtypes`. Some of the infrastructure is therefore also compatible with `pytorch`!


## Properties System

Marathon supports custom properties beyond the standard `energy`, `forces`, `stress`. The key idea is that users define **one** `PROPERTIES` dict and **one** `NORMALIZATION` dict, then pass them to all relevant infrastructure—each module extracts what it needs.

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
- Properties not listed (like forces) are per-atom by nature

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
