# Examples

| Example | Description |
|---------|-------------|
| `train_plain/` | Training on small datasets with fixed-size padding and GPU-side shuffling |
| `train_grain/` | Scalable training with `marathon.grain` (mmap data, parallel loading) |
| `inference/` | Batch inference on toy data with the prediction pipeline |
| `calculator/` | ASE calculator wrapping a marathon model |

All examples use a Lennard-Jones toy model for demonstration. Each has a `run.sh` that runs the example end-to-end.

**Note:** `train_plain` and `train_grain` run extremely slowly on CPU/Mac. Use a GPU for reasonable performance.
