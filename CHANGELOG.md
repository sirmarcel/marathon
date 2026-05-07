# Changelog

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
