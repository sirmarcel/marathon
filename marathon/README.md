## Conventions, etc.

The most basic internal data object is a `Sample`, which is simply a `namedtuple` of `(Graph, labels)`, where `Graph` is what you'd expect and `labels` is a `dict` with the keys `"energy"`, `"forces"`, `"stress"` (optional).

The next step is a `Batch`, which is another `namedtuple` that contains one (or more) `samples` batched together and padded/offset such that an MPNN can deal with everything all at once. I.e. a `Batch` is a big graph that consists of disconnected subgraphs each corresponding to one original `Sample`.

Padding is accomplished by adding a fake additional graph that only connects to itself and is otherwise zeroed out or copies node labels from the first example. Downstream code needs to handle zero edges, i.e., $r_{ij} = 0$!

Depending on the dataset and the model architecture, it may be better to always have `Batch`es with the exact same shape (so they can be stacked and processed with `vmap`), or it may be better to batch dynamically to some set of shapes. This induces recompiles, but may be acceptable to reduce wasted computation.

On the `labels` side, we expect a scalar `energy`, `[N, 3]` forces and `[3, 3]` *extensive* stress (i.e. multiplied with `V`, and defined such that it is simply `d U / d eps`). While the code does not really care about units, the `marathon.emit` infrastructure expects everything to be in eV and Ångstrom.

Note: We do everything up to generating a `Batch` in pure `numpy` in double precision. It is up to downstream code to convert to `jax` arrays and potentially cast into other `dtypes`. Some of the infrastructure is therefore also compatible with `pytorch`!
