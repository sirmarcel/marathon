# Example: train a model using `marathon.grain`

This example demonstrates how to use the `grain`-based infrastructure to train models at scale with datasets that may not fit in memory.

## Quickstart

This example includes a Lennard-Jones toy dataset for testing. To run end-to-end:

```bash
cd examples/train_grain

# 1. Prepare mmap data from LJ trajectory
python prepare_data.py

# 2. Run training
python run.py
```

This will:
- Generate 240 LJ trajectory frames and split into train/valid sets
- Convert to mmap format in `data_train/` and `data_valid/`
- Train a simple LJ model for 300 epochs with validation every epoch
- Save checkpoints to `run/`

## Overview

The training script (`run.py`) shows a complete workflow:

1. **Data Loading**: Uses `DataSource` for mmap-based random access to training/validation data
2. **Transform Pipeline**: `Atoms → ToSample → Filters → Batching`
3. **Training Loop**: JIT-compiled loss function with metric accumulation
4. **Checkpointing**: Save best models based on validation metrics (`SummedMetric`)
5. **Logging**: Text-based logging (and optionally W&B)

## Adapting for your own data

1. Prepare your data in mmap format using `marathon.grain.prepare`:
   ```python
   from marathon.grain import prepare
   prepare(your_atoms_list, folder="data_train")
   ```
2. Update `model.yaml` with your model architecture
3. Update paths and settings at the top of `run.py`
4. Run: `python run.py`

## Configuration

Key settings in `run.py`:
- `data_train`, `data_valid`: paths to mmap datasets
- `batch_style`: `"batch_shape"` (fixed shape) or `"batch_length"` (fixed sample count)
- `loss_weights`: relative weights for energy/forces/stress
- `max_epochs`, `valid_every_epoch`: training schedule
- `worker_count`, `worker_buffer_size`: grain parallelism settings

## Cleanup

To remove generated files after running the example:

```bash
rm -rf data_train data_valid run
```

## Requirements

Ensure `marathon[grain]` is installed (see `pyproject.toml`).
