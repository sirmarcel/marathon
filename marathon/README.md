## Conventions, etc.

The most basic internal data object is a `Sample`, which is simply a `namedtuple` of `(Graph, labels)`, where `Graph` is what you'd expect and `labels` is a `dict` with the keys `"energy"`, `"forces"`, `"stress"` (optional).

The next step is a `Batch`, which is another horrible `namedtuple` that contains one (or more) `samples` batched together and padded/offset such that an MPNN can deal with everything all at once. I.e. a `Batch` is a big "supergraph" (or whatever) that consists of disconnected subgraphs.

For now, we assume (crucially) that every `Batch` belonging to the same split of the data has the exact same shape at all `pytree` leaves.

Padding is accomplished by adding a fake additional graph that only connects to itself and is otherwise zeroed out or copies node labels from the first example. Downstream code needs to handle zero edges!

It's still not clear whether it's better to make `Batches` with size 1 and then shuffle on the fly or put things together on the fly. Likely depends on the diversity of the dataset!

On the `labels` side, we expect a scalar `energy`, `[N, 3]` forces and `[3, 3]` *extensive* stress (i.e. multiplied with `V`, and defined such that it is simply `d U / d eps`). Units don't exist, we just take what comes (some printing may be done in `meV` etc.)

We do everything up to generating a `Batch` in pure `numpy`. It is up to downstream code to convert to `jax` arrays and potentially cast into other `dtypes`.
