# Changelog

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
