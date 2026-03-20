# `marathon.emit`: Outputs

## `checkpoint`: Checkpointing

Since `marathon` is aimed at small-scale training runs, we have declined to rely on `orbax` and similarly "industrial-grade" solutions. Instead, we store checkpoints as a combination of `.yaml` files (for model architecture) and `.msgpack` (for weights/state).

The infrastructure for reading and writing things can be found in `marathon.io`; here we implement the checkpointing logic itself.

### Checkpoint folder layout

```
run/checkpoints/latest/
  ├── model/
  │   ├── model.yaml       # architecture (spec dict)
  │   ├── model.msgpack     # weights
  │   └── baseline.yaml     # per-element energy baseline
  ├── state.msgpack          # training state (step counter, etc.)
  ├── metrics.yaml           # evaluation metrics
  ├── config.yaml            # training config (optional)
  └── info.txt               # human-readable save info
```

### Checkpointers

Checkpointers are registered callables that decide *when* to save. Each one is called on every step with `(step, metrics)` and returns `(should_save, (folder_name, info_string))`.

- **`Latest`**: saves every N steps
- **`SummedMetric`**: saves when a summed metric (e.g. energy MAE + forces MAE) improves

Checkpointers can carry state across resumptions via `.state_dict` / `.restore()`.

## `log`: Logging

Two logger implementations with a shared interface:

- **`Txt`**: writes to text files (`logs/train.txt`, `logs/valid.txt`) with formatted columns
- **`WandB`**: logs to Weights & Biases with correct metric summaries (min for MAE/RMSE, max for R²)

Both accept `(step, train_loss, train_metrics, val_loss, val_metrics)` and use the properties/normalization system for unit scaling (eV → meV, etc.).

## `plot`: Diagnostic plots

`plot()` generates one scatterplot per property (ground truth vs. prediction) with embedded MAE/RMSE/R² text.

## `pretty`: Console formatting

`format_metrics()` produces compact multi-line summaries for console output, using the properties system for symbols and units.
