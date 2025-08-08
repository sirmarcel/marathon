## `hermes`: *fast* training at *scale*

This subpackage contains infrastructure for building training pipelines with [`grain`](https://github.com/google/grain), which takes care of preparing batches in parallel in a scalable way. The aim of this is to be able to train with datasets that don't fit into VRAM, and maybe not even RAM. We provide the following:

- A `DataSource` that is based on `mmap`-ed arrays representing flattened `ase.Atoms`, which provides reasonably fast random access
- Various `Transform`s that handle turning `Atoms` into `Sample` and then batches. We support different batching strategies (fixed shape, dynamic shape, etc.)
- Support for PET-style edge transformers which require (a) rectangular neighborlists and (b) neighborlist inverses

The subpackage is organised as:

- `data/`: versions of `marathon.data` functionality that is custom (mainly, we put much more information into `Sample`)
- `data_source/`: implementation of the `mmap`-based data source
- `transforms/`: the different transforms
- `pain.py`: this just imports functionality from `grain.python` w/ some slightly different defaults

We have not yet declared the dependencies for this subpackage as it is in development, currently we require `grain`, `matscipy` (for mixed `pbc` only), and `mmap_ninja` (for the `DataSource`), as well as `numba` (for the dense neighborlist infrastructure).
