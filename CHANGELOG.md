# Changelog

## Unreleased

### Added

- Model inputs through the properties system. `keys` selects labels, the new `inputs` argument selects properties the model reads; both refer to the same `properties` dict, and the role is decided at the call site, never in `properties.yaml`. Inputs land in `sample.structure` and in a new `batch.inputs` dict, padded and masked like labels. `to_sample`, `batch_samples` (plain and edge-to-edge), `ToSample`, `ToFixedLengthBatch`, `ToFixedShapeBatch`, and `ToEdgeToEdgeBatch` take `inputs=()`.
- `to_sample` and `grain.ToSample` take `structure_fn(atoms, cutoff) -> dict` to replace `to_structure` with a model-specific geometry builder, keeping the inputs merge and label reading in one place.
- `data.read_properties` and `data.batch_properties`: the role-agnostic building blocks for custom samplers and batchers. `to_labels` and `batch_labels` are now thin wrappers around them with unchanged signatures.

### Changed

- `to_sample`, `to_labels`, and `grain.ToSample` lost the `energy`/`forces`/`stress` convenience flags. `keys` now defaults to `("energy", "forces")`; pass `keys=()` for no labels, `keys=("energy", "forces", "stress")` for stress.
- `Batch` (in `data.batching` and `extra.edge_to_edge.batching`) gained a required trailing `inputs` field. Code that builds a `Batch` by hand must pass `inputs={}`.
- `to_structure` no longer reads `atoms.get_initial_charges()` into `structure["charges"]`. Nothing consumed it; declare `initial_charges` as a property with `storage: atoms.arrays` and pass it via `inputs` instead.

### Fixed

- `grain.DataSource`: an `info.yaml` now updates `atoms.info` instead of replacing it, so properties stored in `atoms.info` survive.

## v0.3.1 (2026-09-24)

### Fixed

- `grain.transforms.RandomRotation`: import the Voigt stress helpers from `ase.stress` (they left `ase.constraints` in ASE 3.29), and copy `calc.results` before rotating so the input atoms are no longer mutated in place. (#5, #6)
- `evaluate.metrics.get_stats`: labels missing from a sample are treated like NaN labels, and keys with no valid labels at all are omitted from the result with a `comms.warn` instead of crashing. `metrics_fn` computes R² only for keys that have stats; `emit.pretty.format_metrics` and `emit.SummedMetric` skip R² when absent. (#2, #8)

### Changed

- Pin `grain<0.2.17`: newer versions read an unparsed absl flag in worker processes on Linux and break `DataLoader` with `worker_count > 0` outside `absl.app`.

### Internal

- CI: `tests.yml` runs lint, inline tests, pytest, and all examples on pull requests and pushes to `main`. (#7)
- Dropped the undeclared `jaxtyping` import from `tests/lj.py` and the example models; shapes are now comments.
- Exclude `*.md` from ruff, which as of 0.16 would reformat Python blocks in the READMEs.
- Example run outputs are gitignored.

## v0.3.0 (2026-05-07)

### Changed

- **`marathon.emit.plot`**: `plot()` is now unit-agnostic. It takes values at face value and writes a `metrics.yaml` computed from what it plots; the caller owns scaling and normalization. Signature change: removed `metrics`, `properties`, `normalization`; added `units` (optional `{key: str}` for axis labels). `simple_scatterplot` lost its `metrics=` kwarg and the corresponding assertion.

### Fixed

- `plot()` previously asserted hand-rolled RMSE/MAE/R² against the metrics dict from training, but the units never matched (predictions/labels were scaled while the metrics were not, and stress wasn't being per-atom-normalized in the example collator either). The assertion failed any time it actually ran. Resolved by the unit-agnostic refactor above plus an example-side fix.

### Internal

- Moved both VRAM-allocating inline tests from `marathon/evaluate/metrics.py` into `tests/test_metrics.py`. Importing `marathon.evaluate.metrics` no longer initializes the JAX backend (which would preallocate ~75% of GPU memory and OOM under parallel grain workers). Added a rule to `CLAUDE.md`: inline `# -- test --` blocks must not dispatch any JAX op.
- `examples/train_plain/run.py:predict_and_collate` rewritten to be generic, driven by a single `properties` dict merged from `marathon.data.properties.DEFAULT_PROPERTIES` and `marathon.emit.properties.DEFAULT_PROPERTIES`. Designed to be copy-pasted into custom pipelines.

## v0.2.2 (2026-03-20)

Fix logo URLs for PyPI (use absolute URLs).

## v0.2.1 (2026-03-20)

Add README to PyPI package metadata.

## v0.2.0 (2026-03-20)

First PyPI release. Major update porting from marathon-dev.

### Added
- **Properties system**: extensible property definitions (`shape`, `storage`, `report_unit`, normalization) threading through the full pipeline
- **marathon.grain**: scalable data pipelines — memory-mapped datasets, configurable batching (`ToFixedLengthBatch`, `ToFixedShapeBatch`), filters, and augmentation (`RandomRotation`)
- **marathon.extra.edge_to_edge**: PET-style rectangular neighborlists with reverse indices, backed by numba
- **Huber loss** option and per-structure metric tracking
- **emit**: console formatting (`pretty`), unit-aware logging, diagnostic scatterplots
- Four end-to-end examples (train_plain, train_grain, inference, calculator)
- Docstrings and READMEs for all subpackages
- CI release workflow via GitHub Actions + trusted publishing

### Changed
- Data model refactored: `Sample(structure, labels)` with explicit neighbor graph; `Batch` fields renamed (e.g. `node_mask` → `atom_mask`)
- `comms` dependency replaced by `opsis` (published separately on PyPI)
- `hermes` renamed to `marathon.grain` (deprecated re-exports at `marathon.extra.hermes`)
- `io.from_dict` gains `allow_stubs` and `default_namespace` parameters

### Removed
- `ensemble` module (dead code, zero downstream usage)
