## `transforms`

The main building block of `grain` pipelines are various kinds of transforms that are applied to the data. In our case, we mainly care about: (a) turning `Atoms` into our internal `Sample` tuples and (b) batching those tuples into `Batch`. This is also where we have data augmentation.
