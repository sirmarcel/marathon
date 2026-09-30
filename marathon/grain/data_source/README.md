## `data_source`

One requirement for having a performant pipeline (at least in the `grain` lifestyle) is having fast random access to samples. The abstraction for this is `DataSource`. Here, we implement all the stuff needed to have a `DataSource` that yields `Atoms` objects reasonably fast. The main problem to solve is storage: `.xyz` files are impossible to read fast in random access fashion (unless you like very many files, which is slow).

We take a simple solution: We `flatten` our `Atoms` objects into a big `mmap`-ed array. The book-keeping is managed by `mmap-ninja`. While writing, `prepare` collects the energies for the composition baseline and fits it once at the end (`fit_baseline` does the same for an existing folder). The result is a folder with the baseline and the mmap. `prepare(..., num_workers=N)` flattens shards of an indexable dataset in a pool of spawned workers and concatenates the shard mmaps in order; each worker receives a pickled copy, so the dataset should be a list (sliced per shard) or a reader that holds only paths and opens files lazily, like `marathon.grain.AseDB` for `ase.db` files.

This folder is then consumed by the `DataSource` which implements the required `grain` interface.

Note: `DataSource` guarantees that all properties are returned and filled with `nan` in case they were not present in the data.
