### `data`

Normally, the idea of `marathon` would be that you write a custom `to_sample` function if anything beyond the bare minimum is needed downstream. However, in `grain`, there's a higher cost for writing custom `to_sample` functions, since we have to wrap it into a class, etc cetera. We therefore define some more inclusive functions here, which stuff the maximum amount of information into samples, and trust on downstream batching logic, which will probably *have* to be custom, to throw away stuff we don't need.

Currently, we implement this by hacking an extra field into `Graph`, `info`, which is just a free-form `dict` with stuff that may be relevant.
