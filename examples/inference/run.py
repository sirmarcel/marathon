"""
Inference example: load a model, predict on toy data, print results.

This uses the LJ toy model directly (no checkpoint loading) to demonstrate
the prediction and metrics pipeline.
"""

import numpy as np
import jax

from lj_data import steps
from model import LennardJones

key = jax.random.key(1)
key, init_key = jax.random.split(key)

# -- model setup --

model = LennardJones(
    initial_sigma=2.0,
    initial_epsilon=1.5,
    cutoff=6.0,
    onset=5.0,
)
cutoff = model.cutoff
params = model.init(init_key, *model.dummy_inputs())

# -- data --

from marathon.data import batch_samples, determine_max_sizes, to_sample

keys = ["energy", "forces"]
data = steps[:10]  # just a few structures

samples = [to_sample(a, cutoff, stress=False) for a in data]

num_nodes, num_edges = determine_max_sizes(samples, 1)
batches = [batch_samples([s], num_nodes, num_edges, keys) for s in samples]

# -- predict --

from marathon.evaluate import get_predict_fn

pred_fn = get_predict_fn(apply_fn=model.apply, stress=False)
pred_fn = jax.jit(pred_fn)


def predict_and_collate(params, batches):
    predictions = {k: [] for k in keys}
    labels = {k: [] for k in keys}

    for batch in batches:
        preds = pred_fn(params, batch)

        for key in keys:
            mask = batch.labels[key + "_mask"]
            if mask.any():
                predictions[key].append(preds[key][mask])
                labels[key].append(batch.labels[key][mask])

    final_predictions = {}
    final_labels = {}

    for key in keys:
        if key == "energy":
            final_predictions[key] = np.array(predictions[key]).flatten()
            final_labels[key] = np.array(labels[key]).flatten()
        elif key == "forces":
            final_predictions[key] = np.concatenate(predictions[key]).reshape(-1, 3)
            final_labels[key] = np.concatenate(labels[key]).reshape(-1, 3)

    return final_labels, final_predictions


labels, predictions = predict_and_collate(params, batches)

# -- report --

for key in keys:
    l = labels[key].flatten()
    p = predictions[key].flatten()
    mae = np.mean(np.abs(l - p))
    print(f"{key}: MAE = {mae:.6f}")

print("done!")
