## `hermes`: *fast* training at *scale*

**Note:** The canonical import path is now `marathon.grain`. Importing from `marathon.extra.hermes` still works but emits a deprecation warning.

This subpackage contains infrastructure for building training pipelines with [`grain`](https://github.com/google/grain), which takes care of preparing batches in parallel in a scalable way. The aim is to train with datasets that don't fit into VRAM, and possibly not even RAM.

### Structure

- `data_source/`: mmap-based `DataSource` providing fast random access to flattened `ase.Atoms`
- `transforms/`: grain-compatible `Transform`s for filtering, sampling, and batching
- `pain.py`: re-exports from `grain.python` with slightly different defaults

### Exports

**Data Source**
- `DataSource`: mmap-based random access to datasets
- `prepare`: convert `ase.Atoms` collections to mmap format

**Data Loading**
- `DataLoader`: grain's data loader with multiprocessing
- `IndexSampler`: simple index-based sampler supporting shuffling and epochs
- `prefetch_to_device`: prefetch iterator to GPU memory

**Transforms (Filters)**
- `FilterEmpty`: remove samples with no edges
- `FilterTooSmall`: remove samples below a size threshold
- `FilterAboveNumAtoms`: remove samples above an atom count
- `FilterMixedPBC`: remove samples with inconsistent periodic boundaries
- `FilterNoop`: passthrough filter

**Transforms (Batching)**
- `ToSample`: convert `Atoms` to `Sample` (graph with neighborlist)
- `ToFixedShapeBatch`: fixed shape, varying number of samples per batch
- `ToFixedLengthBatch`: varying shape, fixed number of samples per batch
- `ToEdgeToEdgeBatch`: for edge-to-edge models (see `marathon.extra.edge_to_edge`)
- `ToStack`: stack multiple batches into a chunk for `jax.lax.scan`

**Transforms (Augmentation)**
- `RandomRotation`: apply random O(3) rotation to structures

### Usage

Please ensure `marathon[hermes]` is installed (see `pyproject.toml`).

For a complete training example, see `examples/train_grain/`.
