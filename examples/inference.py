from pathlib import Path

checkpoint = Path("")
uq = True
keys = ["energy", "forces", "stress"]

data = [...]

if "stress" in keys:
    use_stress = True
else:
    use_stress = False


import numpy as np
import jax

key = jax.random.key(1)
key, init_key = jax.random.split(key)


from myrto.engine import from_dict, read_yaml

model = from_dict(read_yaml(checkpoint / "model/model.yaml"))
cutoff = model.cutoff

params = model.init(init_key, *model.dummy_inputs())

baseline = read_yaml(checkpoint / "model/baseline.yaml")
species_to_weight = baseline["elemental"]

from marathon.elemental import get_energy_fn

elemental_energy_fn = get_energy_fn(species_to_weight)


from marathon.emit.checkpoint import read_msgpack

params = read_msgpack(checkpoint / "model/model.msgpack")

if uq:
    from marathon.ensemble import get_predict_fn

    pred_fn = get_predict_fn(
        model.apply,
        stress=use_stress,
        derivative_variance=True,
    )
    pred_fn = jax.jit(pred_fn)
else:
    raise ValueError  # not yet supported


def predict_and_collate(params, batches):
    # to avoid running out of VRAM, we iterate one
    # structure at a time, and use the chance to also
    # collect the correct labels, dropping masked items

    predictions = {k: [] for k in keys}
    labels = {k: [] for k in keys}

    if uq:
        for k in keys:
            predictions[k + "_var"] = []

    for batch in batches:
        preds = pred_fn(params, batch)

        for key in keys:
            mask = batch.labels[key + "_mask"]
            kv = key + "_var"
            if mask.any():
                predictions[key].append(preds[key][mask])
                labels[key].append(batch.labels[key][mask])

                if kv in preds:
                    predictions[kv].append(preds[kv][mask])

    final_predictions = {}
    final_labels = {}

    for key in predictions.keys():
        if "energy" in key:
            final_predictions[key] = np.array(predictions[key]).flatten()
        if "forces" in key:
            final_predictions[key] = np.array(predictions[key]).reshape(-1, 3)
        if "stress" in key:
            final_predictions[key] = np.array(predictions[key]).reshape(-1, 3, 3)

    for key in keys:
        if key == "energy":
            final_labels[key] = np.array(labels[key]).flatten()
        if key == "forces":
            final_labels[key] = np.array(labels[key]).reshape(-1, 3)
        if key == "stress":
            final_labels[key] = np.array(labels[key]).reshape(-1, 3, 3)

    return final_labels, final_predictions


def predict(data):
    from marathon.data import determine_sizes, get_batch, to_sample

    samples = [to_sample(a, cutoff, stress=use_stress) for a in data]

    for sample in samples:
        sample.labels["energy"] -= elemental_energy_fn(sample.graph)

    num_nodes, num_edges = determine_sizes(samples, 1)
    batches = [get_batch([s], num_nodes, num_edges, keys) for s in samples]

    l, p = predict_and_collate(params, batches)

    return l, p
