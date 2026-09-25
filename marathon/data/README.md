# `marathon.data`: Samples, batching, and data splits

Converts `ase.Atoms` objects into the internal `Sample` and `Batch` representations described in the top-level README. This module is pure numpy and does not depend on JAX (except `splits.py`, which uses `jax.random` for reproducible shuffling).

## `sample`: `ase.Atoms` → `Sample`

`to_structure(atoms, cutoff)` computes the neighbor list (via `vesin`) and packs geometry into a structure dict. `read_properties(atoms, keys, ...)` reads properties using the `storage` field from the properties config; it is role-agnostic. `to_labels` wraps it and adds `num_atoms`. `to_sample` combines both: `keys` become labels, `inputs` are merged into the structure dict.

For standard properties (`energy`, `forces`, `stress`), extraction from the ASE calculator handles the stress convention (multiply by volume to get $dU/d\varepsilon$). Custom properties are read from `atoms.info` or `atoms.arrays` as configured.

## `batching`: `[Sample]` → `Batch`

`batch_samples` collates a list of `Sample`s into a single `Batch` namedtuple with padding. Index arrays (`centers`, `others`) are offset per structure so the batch looks like one disconnected graph. `batch_properties` stacks named properties from a list of dicts into padded arrays with per-key masks (NaN → `False`); it serves both `batch.labels` (via `batch_labels`, which adds `num_atoms`) and `batch.inputs` (read from the structure dicts).

## `sizes`: Buffer allocation

`determine_max_sizes` scans samples to find worst-case atom and pair counts for a given batch size, then rounds up via `next_size` to a JIT-friendly number.

Typically not needed and superseded by more complex `grain` machinery -- but this is nice for very small all-GPU training runs.

## `splits`: Train/valid/test splits

`get_splits` draws non-overlapping index sets using `jax.random` for reproducibility.

## `properties`: Shape and storage specs

`DEFAULT_PROPERTIES` defines `shape` and `storage` for the standard properties. `deduce_shape` resolves the `"atom"` placeholder at batch time.
