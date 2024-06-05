import numpy as np

import jax
import jax.numpy as jnp

from pathlib import Path

from marathon import comms


# -- settings --

workdir = "run"

train_batch_size = 2
valid_batch_size = 10

loss_weights = {"energy": 1.0, "forces": 1.0, "stress": 1.0}

start_learning_rate = 1e-3
min_learning_rate = 1e-6

lr_decay_patience = 50
lr_decay_factor = 0.75
lr_decay_rtol = 1e-4
lr_decay_atol = 1e-8
lr_decay_cooldown = 10
lr_accumulation_epochs = 4

seed = 0

max_epochs = 1000

valid_every = 2
chunk_size = 5

scale_by_variance = True

print_model_summary = True

# -- housekeeping based on settings --

keys = list(loss_weights.keys())
use_stress = "stress" in keys

# -- emissions --

from marathon.emit import Latest, SummedMetric, Txt

checkpointers = (
    Latest(10),
    SummedMetric("best_E+F+S_R2", "r2", keys=["energy", "forces", "stress"]),
    SummedMetric("best_E_R2", "r2", keys=["energy"]),
    SummedMetric("best_F_R2", "r2", keys=["forces"]),
)
loggers = (Txt(keys=keys),)


# -- are we starting from scratch? --

workdir = Path(workdir)
if workdir.is_dir():
    comms.warn(f"found working directory {workdir}, will recover from there")
    try_restore = True
else:
    workdir.mkdir()
    try_restore = False

# -- let's go --

reporter = comms.reporter()
reporter.start("run")

reporter.step("load data")
from data import get_data

data_train, data_valid, _ = get_data()
n_train = len(data_train)
n_valid = len(data_valid)

reporter.step("setup")

# -- randomness --

key = jax.random.key(seed)
key, init_key = jax.random.split(key)

# -- model --
from myrto.engine import read_yaml, from_dict

model = from_dict(read_yaml("model.yaml"))
cutoff = model.cutoff

params = model.init(init_key, *model.dummy_inputs())

if print_model_summary:
    from flax import linen as nn
    msg = nn.tabulate(model, init_key)(*model.dummy_inputs())
    comms.state(msg.split("\n"), title="Model Summary")

# -- optimizer --
import optax
from optax.contrib import reduce_on_plateau

optimizer = optax.chain(
    optax.adam(start_learning_rate),
    reduce_on_plateau(
        factor=lr_decay_factor,
        patience=lr_decay_patience,
        rtol=lr_decay_rtol,
        atol=lr_decay_atol,
        cooldown=lr_decay_cooldown,
        accumulation_size=lr_accumulation_epochs * (n_train // train_batch_size),
    ),
)
opt_state = optimizer.init(params)

# -- data pre-processing --
from marathon.data import to_sample

reporter.step("processing data")

train_samples = [to_sample(a, cutoff, stress=use_stress) for a in data_train]
valid_samples = [to_sample(a, cutoff, stress=use_stress) for a in data_valid]

# -- remove per-element contributions --
from marathon.elemental import elemental

species_to_weight, elemental_energy_fn = elemental(train_samples)
baseline = {"elemental": species_to_weight}

msg = []
for s, w in species_to_weight.items():
    msg.append(f"{s}: {w:.3f}")
comms.state(msg, title="per-atom contributions (by species)")

for sample in train_samples:
    sample.labels["energy"] -= elemental_energy_fn(sample.graph)
for sample in valid_samples:
    sample.labels["energy"] -= elemental_energy_fn(sample.graph)

# -- assemble all the states/restore --

state = {"epoch": 0, "checkpointers": checkpointers, "opt_state": opt_state, "key": key}

if try_restore:
    reporter.step("restoring")
    from marathon.emit import get_latest

    items = get_latest(workdir, state)

    if items is None:
        comms.warn(f"failed to find checkpoints in {workdir}, proceeding anyway")
    else:
        params, state, new_model, baseline = items

        comms.talk(f"restored epoch {state['epoch']} from folder {workdir}", full=True)

        # try to catch the most obvious error: editing the model config between restarts
        from myrto.engine.serialize import to_dict

        assert to_dict(new_model) == to_dict(model)

# -- get data ready for action --
reporter.step("data preparation")


def tree_stack(trees):
    return jax.tree_util.tree_map(lambda *x: jnp.stack(x), *trees)


def tree_split_first_dim(tree, leading):
    def fn(x):
        old_shape = x.shape
        if len(old_shape) > 1:
            new_shape = (leading, int(old_shape[0] / leading), *old_shape[1:])
        else:
            new_shape = (leading, int(old_shape[0] / leading))
        return x.reshape(*new_shape)

    return jax.tree_util.tree_map(fn, tree)


from marathon.data import get_batch, determine_sizes

# we make fake batches with one structure, then shuffle+reshape them into batches later.
# this has the advantage of being jittable and happening entirely on the GPU, so
# it avoids data transfer and the juggling of assembling batches fast enough to
# keep the GPU fed, but it assumes that all samples are similar. maybe revisit later
train_num_nodes, train_num_edges = determine_sizes(train_samples, 1)
valid_num_nodes, valid_num_edges = determine_sizes(valid_samples, 1)

train_pre_batches = [
    get_batch([s], train_num_nodes, train_num_edges, keys) for s in train_samples
]
valid_pre_batches = [
    get_batch([s], valid_num_nodes, valid_num_edges, keys) for s in valid_samples
]

assert n_train % train_batch_size == 0
train_n_batches = n_train // train_batch_size

assert n_valid % valid_batch_size == 0
valid_n_batches = n_valid // valid_batch_size

valid_batches = tree_stack(valid_pre_batches)
valid_batches = tree_split_first_dim(valid_batches, valid_n_batches)

train_data = tree_stack(train_pre_batches)
# train_batches are assembled in the training loop


if scale_by_variance:
    from marathon.evaluate.metrics import get_stats

    stats = get_stats(train_samples, keys=keys)

    old_loss_weights = loss_weights

    loss_weights = {k: v / stats[k]["var"] for k, v in loss_weights.items()}

    msg = []
    for k, v in loss_weights.items():
        msg.append(f"{k}: {old_loss_weights[k]:.3f} -> {v:.3f}")
    comms.state(msg, title="scaled loss weights")

# -- let's fucking go --
reporter.step("setup training loop")
from marathon.evaluate import get_predict_fn, get_loss_fn, get_metrics_fn

pred_fn = get_predict_fn(model.apply, stress=use_stress)
loss_fn = get_loss_fn(pred_fn, weights=loss_weights)

train_metrics_fn = get_metrics_fn(train_samples, keys=keys)
valid_metrics_fn = get_metrics_fn(valid_samples, keys=keys)

from time import monotonic

from marathon.utils import s_to_string
from marathon.emit import save_checkpoints


def report_on_lr(opt_state):
    lr = start_learning_rate * opt_state[1].scale
    best = opt_state[1].best_value
    return f"LR decay: lr {lr:.3e}, best loss {best:.3e}"


class Manager:
    def __init__(self, state, interval, loggers, workdir, model, baseline, max_epochs):
        self.state = state
        self.interval = interval
        self.loggers = loggers
        self.workdir = workdir
        self.model = model
        self.baseline = baseline

        self.max_epochs = max_epochs

        self.start_epoch = state["epoch"]
        self.start_time = monotonic()

    @property
    def done(self):
        return self.epoch >= self.max_epochs

    @property
    def epoch(self):
        return self.state["epoch"]

    @property
    def elapsed(self):
        return monotonic() - self.start_time

    @property
    def time_per_epoch(self):
        return self.elapsed / (self.epoch - self.start_epoch)

    @property
    def eta(self):
        return (self.max_epochs - self.epoch) * self.time_per_epoch

    def report(
        self, key, params, opt_state, train_loss, train_metrics, valid_loss, valid_metrics
    ):
        self.state["epoch"] += self.interval
        self.state["opt_state"] = opt_state
        self.state["key"] = key

        for logger in self.loggers:
            logger(
                self.state["epoch"], train_loss, train_metrics, valid_loss, valid_metrics
            )

        metrics = {"train": train_metrics, "valid": valid_metrics}
        metrics = jax.tree_util.tree_map(lambda x: np.array(x), metrics)

        save_checkpoints(
            metrics, params, self.state, self.model, self.baseline, self.workdir
        )

        title = f"state at epoch: {self.epoch}"
        msg = []

        msg.append(f"train loss: {train_loss:.5e}")
        msg.append(f"valid loss: {valid_loss:.5e}")

        msg.append(report_on_lr(opt_state))

        msg.append(f"validation errors:")
        msg.append(f". E")
        msg.append(f".. R2  : {metrics['valid']['energy']['r2']:.3f} %")
        msg.append(f".. MAE : {metrics['valid']['energy']['mae']:.3e} meV/atom")
        msg.append(f".. RMSE: {metrics['valid']['energy']['rmse']:.3e} meV/atom")

        msg.append(f". F")
        msg.append(f".. R2  : {metrics['valid']['forces']['r2']:.3f} %")
        msg.append(f".. MAE : {metrics['valid']['forces']['mae']:.3e} meV/Å")
        msg.append(f".. RMSE: {metrics['valid']['forces']['rmse']:.3e} meV/Å")

        if use_stress:
            msg.append(f". σ")
            msg.append(f".. R2  : {metrics['valid']['stress']['r2']:.3f} %")
            msg.append(f".. MAE : {metrics['valid']['stress']['mae']:.3e} meV")
            msg.append(f".. RMSE: {metrics['valid']['stress']['rmse']:.3e} meV")

        msg.append("")
        msg.append(f"elapsed: {s_to_string(self.elapsed, 's')}")
        msg.append(
            f"timing: {s_to_string(self.time_per_epoch)}/epoch, {s_to_string(self.eta, 'm')} ETA"
        )

        msg.append("")
        comms.state(msg, title=title)

        return ()


manager = Manager(state, valid_every, loggers, workdir, model, baseline, max_epochs)


def compute_loss(params, batch):
    loss, aux = jax.vmap(lambda x: loss_fn(params, x))(batch)
    loss = jax.tree_util.tree_map(lambda x: x.mean(axis=0), loss)
    aux = jax.tree_util.tree_map(lambda x: x.sum(axis=0), aux)

    return loss, aux


def do_batch(carry, batch):
    params, opt_state = carry

    loss_and_aux, grads = jax.value_and_grad(compute_loss, argnums=0, has_aux=True)(
        params, batch
    )
    loss, aux = loss_and_aux
    updates, opt_state = optimizer.update(grads, opt_state, params, value=loss)

    params = optax.apply_updates(params, updates)

    return (params, opt_state), (loss, aux)


@jax.jit
def main_loop(
    start_epoch, key, params, opt_state, train_data, valid_batches, chunk_size, valid_every
):
    # this is the actual training loop: we loop over a bunch of epochs in one big JIT,
    # reporting back to the CPU every `valid_every`-th epoch through a host callback

    def do_epoch(epoch, val):
        key, params, opt_state = val

        key, shuffling_key = jax.random.split(key)
        shuffled_idx = jax.random.permutation(shuffling_key, jnp.arange(n_train, dtype=int))
        shuffled = jax.tree_util.tree_map(lambda x: x[shuffled_idx], train_data)
        batches = tree_split_first_dim(shuffled, train_n_batches)

        carry, res = jax.lax.scan(do_batch, (params, opt_state), batches)
        params, opt_state = carry
        loss, aux = res

        train_loss = jnp.mean(loss)
        train_metrics = train_metrics_fn(aux)

        # XLA crashes if we do this inside the lax.cond statement...
        #   ... so we hope this gets optimised out when not needed... :clown:
        valid_loss, valid_aux = jax.vmap(lambda x: compute_loss(params, x))(valid_batches)
        valid_loss = jnp.mean(valid_loss)

        valid_metrics = valid_metrics_fn(valid_aux)

        def report(
            key, params, opt_state, train_loss, train_metrics, valid_loss, valid_metrics
        ):
            jax.experimental.io_callback(
                manager.report,
                (),
                key,
                params,
                opt_state,
                train_loss,
                train_metrics,
                valid_loss,
                valid_metrics,
                ordered=True,
            )

            return None

        def nop(
            key, params, opt_state, train_loss, train_metrics, valid_loss, valid_metrics
        ):
            return None

        jax.lax.cond(
            epoch % valid_every == 0,
            report,
            nop,
            key,
            params,
            opt_state,
            train_loss,
            train_metrics,
            valid_loss,
            valid_metrics,
        )

        return (key, params, opt_state)

    return jax.lax.fori_loop(
        start_epoch,
        start_epoch + valid_every * chunk_size,
        do_epoch,
        (key, params, opt_state),
    )


reporter.step("train")

epoch = manager.start_epoch
while not manager.done:
    key, params, opt_state = main_loop(
        epoch,
        key,
        params,
        opt_state,
        train_data,
        valid_batches,
        chunk_size,
        valid_every,
    )
    epoch += valid_every * chunk_size

# -- wrap up --
reporter.step("wrapup")

from marathon.emit import plot, get_all


@jax.jit
def do_eval(params, valid_batches):
    predictions = jax.vmap(jax.vmap(lambda x: pred_fn(params, x)))(valid_batches)
    loss, aux = jax.vmap(lambda x: compute_loss(params, x))(valid_batches)

    return predictions, loss, aux


def collate_labels(labels):
    out = {}
    keys = list(labels.keys())

    for key in keys:
        if key == "energy":
            energy = np.array(labels[key])
            energy = energy[valid_batches.graph_mask]
            out[key] = energy.flatten()

        if key == "forces":
            forces = np.array(labels[key])
            forces = forces[valid_batches.node_mask]
            out[key] = forces.reshape(-1, 3)

        if key == "stress":
            stress = np.array(labels[key])
            stress = stress[valid_batches.graph_mask]
            out[key] = stress.reshape(-1, 3, 3)

    return out


labels = collate_labels(valid_batches.labels)

for f, items in get_all(workdir, state):
    if f.suffix == ".backup":
        continue

    comms.talk(f"working on {f}")

    params, state, model, baseline = items
    predictions, loss, aux = do_eval(params, valid_batches)

    metrics = valid_metrics_fn(aux)

    preds = collate_labels(predictions)

    out = f / "plot/valid"
    out.mkdir(parents=True)

    plot(out, preds, labels, metrics=metrics)


reporter.done()
