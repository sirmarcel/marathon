## `edge_to_edge`: PET-style transformer support

This subpackage provides utilities for models that use edge-to-edge attention, such as PET (Positional Embedding Transformer). These models require a specific neighborlist format:

1. **Rectangular neighborlists**: neighborlists that can be reshaped to `[num_atoms, num_neighbors]`
2. **Reverse indices**: an index array mapping each pair `(i,j)` to its inverse `(j,i)`

### Why is this needed?

Standard neighborlists store pairs `(i,j)` as flat arrays. PET-style transformers need:
- Fixed number of neighbors per atom (for efficient attention)
- Fast lookup of the reverse direction (for message passing)

This is non-trivial because (a) we need to pad neighborhoods to fixed size, and (b) finding reverses requires handling periodic boundary conditions (cell shifts).

### Exports

- `batch_samples(samples, num_structures, num_atoms, num_neighbors, keys)`: Create padded batches suitable for edge-to-edge models
- `update_batch(samples, batch, num_atoms, num_neighbors)`: Convert an existing marathon batch to edge-to-edge format
- `get_neighborlist(centers, others, pair_mask, num_atoms, num_neighbors, cell_shifts)`: Low-level neighborlist transformation

### Batch format

The returned `Batch` namedtuple contains:
- `atomic_numbers`: atomic numbers per atom
- `displacements`: displacement vectors `R_ij`
- `centers`, `others`: reshaped to `[num_atoms * num_neighbors]`
- `reverse`: index mapping `ij → ji`
- `atom_to_structure`, `pair_to_structure`: mapping to original structures
- `structure_mask`, `atom_mask`, `pair_mask`: padding masks
- `labels`: target values

### Integration with grain

Use `ToEdgeToEdgeBatch` from `marathon.grain` for grain-based pipelines.
