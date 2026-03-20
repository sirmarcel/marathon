# `marathon.evaluate`: Prediction, loss, and metrics

## Overview

This module turns a model into a training-ready loss function and provides metrics for monitoring. The design separates the model-specific part (how predictions are made) from the generic part (how errors are computed and tracked).

Only `predict.py` makes specific assumptions about the batch structure and how forces emerge from the model. The loss and metrics machinery is largely independent -- `loss.py` just needs a callable that maps `(params, batch) -> dict` plus `batch.labels`, and `metrics.py` only works with aggregated residual statistics, never touching the batch at all.

In practice, models typically implement their own `predict_fn` (tailored to their architecture and batch format) and then slot seamlessly into the loss and metrics infrastructure.

## `predict`: Forces via autodiff

The default `predict_fn` provided here assumes that **models predict per-atom energy contributions**, and obtains forces and stress via JAX's autodiff on the batch's displacement vectors:

1. The model receives `displacements` ($R_{ij}$) and returns per-atom energies.
2. `jax.value_and_grad` differentiates total energy w.r.t. `batch.displacements`.
3. **Forces** are reconstructed by aggregating displacement gradients:
   $$dR_{ij} = \partial E / \partial R_{ij}$$
   $$F_i = \sum_j dR_{ij}(i \to j) - \sum_j dR_{ij}(j \to i)$$
   (Two `segment_sum` operations, one over `centers`, one over `others`.)
4. **Stress** (optional) is the virial tensor, aggregated per structure:
   $$\sigma = \sum_{\text{pairs}} R_{ij} \otimes dR_{ij}$$

This is why `displacements` -- not positions -- are the quantity we differentiate through. The model never sees absolute positions.

### The default model contract

`get_predict_fn(apply_fn)` expects `apply_fn` to have this exact positional signature:

```python
per_atom_energies = apply_fn(
    params,
    batch.displacements,       # (num_pairs, 3)
    batch.centers,             # (num_pairs,)
    batch.others,              # (num_pairs,)
    batch.atomic_numbers,      # (num_atoms,)
    batch.pair_mask,           # (num_pairs,)
    batch.atom_mask,           # (num_atoms,)
)
# returns: (num_atoms,) -- one energy contribution per atom
```

Per-atom energies are summed per structure via `segment_sum` with `atom_to_structure`.

Alternatively, pass a custom `energy_fn(params, batch) -> (total_energy, per_atom_energies)` to bypass the default wrapping. Or, as most real models do, implement your own `predict_fn(params, batch) -> dict` entirely and pass it directly to `get_loss_fn`.

## `loss`: Weighted loss and residual tracking

`get_loss_fn(predict_fn, weights, ...)` returns `loss_fn(params, batch) -> (scalar, aux)`.

The loss function calls `predict_fn`, computes residuals against `batch.labels`, and applies weighted MSE or Huber loss. Each atom contributes equally to the forces loss. Energy and stress residuals are normalised by atom count before the loss computation.

The `aux` dict carries sufficient statistics for computing metrics downstream without a second forward pass:
- `{key}_abs`: summed |residuals| (for MAE)
- `{key}_sq`: summed residuals² (for RMSE)
- `{key}_n`: count of valid samples

For atom-normalised properties, additional `{key}_per_structure_*` entries track unnormalised residuals.

The assumption is that `loss_fn` can be `vmap`'d or `scan`'d across a stack of batches, which gives the leading dimension expected by metrics.

## `metrics`: MAE, RMSE, R²

`get_metrics_fn(...)` returns a function that consumes an aux dict (with a leading batch dimension, stacked in the training loop) and computes final metrics. R² requires reference statistics (ground truth variance); pre-compute these with `get_stats()` or pass them directly.

## Files

- `predict.py`: wraps model into `predict_fn`; the only file that assumes batch internals
- `loss.py`: weighted loss with MSE or Huber; produces aux dict for metrics
- `metrics.py`: MAE, RMSE, R² from aggregated residuals; reference statistics
- `properties.py`: default normalization config (`energy` and `stress` per atom)
