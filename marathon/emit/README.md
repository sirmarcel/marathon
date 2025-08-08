# `marathon.emit`: Outputs

## `checkpoint`: Checkpointing

Since `marathon` is aimed at small-scale training runs, we have declined to rely on `orbax` and similarly "industrial-grade" solutions. Instead, we store checkpoints as a combination of `.yaml` files (for model architecture) and `.msgpack` (for weights/state).

The infrastructure for reading and writing things can be found in `marathon.io`; here we implement the checkpointing logic itself.