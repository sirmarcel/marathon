# -- settings --


train_batch_size = 2
valid_batch_size = 10

loss_weights = {"energy": 1.0, "forces": 1.0, "stress": 1.0}
scale_by_variance = False

remove_baseline = False  # remove constant term based on composition

start_learning_rate = 1e-3
min_learning_rate = 1e-6

lr_decay_patience = 10
lr_decay_factor = 0.75
lr_decay_rtol = 1e-4
lr_decay_atol = 1e-8
lr_decay_cooldown = 10
lr_accumulation_epochs = 4

max_epochs = 300


valid_every = 1
chunk_size = 5


seed = 0
print_model_summary = True
workdir = "run"

use_wandb = False
# used for wandb -- if None, we use folder names
wandb_project = "test-plain"
wandb_name = "myrun"

default_matmul_precision = "default"

# -- imports & startup --

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", default_matmul_precision)

from pathlib import Path

from marathon import comms

reporter = comms.reporter()
reporter.start("run")
reporter.step("startup")


# -- housekeeping based on settings --
keys = list(loss_weights.keys())
use_stress = "stress" in keys

workdir = Path(workdir)


# -- randomness --
key = jax.random.key(seed)
key, init_key = jax.random.split(key)

# -- model --
from marathon.io import from_dict, read_yaml

model = from_dict(read_yaml("model.yaml"))
cutoff = model.cutoff

params = model.init(init_key, *model.dummy_inputs())

if print_model_summary:
    from flax import linen as nn

    msg = nn.tabulate(model, init_key)(*model.dummy_inputs())
    comms.state(msg.split("\n"), title="Model Summary")

num_parameters = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
comms.state(f"Parameter count: {num_parameters}")


# -- checkpointers --
from marathon.emit import SummedMetric

checkpointers = []

name = "R2_" + "+".join([k[0].upper() for k in keys])
checkpointers.append(SummedMetric(name, "r2", keys=keys))

checkpointers = tuple(checkpointers)


# -- data loading --
from data import get_data

reporter.step("load data")

data_train, data_valid, _ = get_data(seed=seed)
n_train = len(data_train)
n_valid = len(data_valid)


# -- data pre-processing --
from marathon.data import to_sample

reporter.step("processing data")

train_samples = [to_sample(a, cutoff, stress=use_stress) for a in data_train]
valid_samples = [to_sample(a, cutoff, stress=use_stress) for a in data_valid]

# -- remove per-element contributions --
from marathon.elemental import get_energy_fn, get_weights

if remove_baseline:
    species_to_weight = get_weights(train_samples)
    baseline = {"elemental": species_to_weight}

    elemental_energy_fn = get_energy_fn(species_to_weight)

    msg = []
    for s, w in species_to_weight.items():
        msg.append(f"{s}: {w:.3f}")
    comms.state(msg, title="per-atom contributions (by species)")

    for sample in train_samples:
        sample.labels["energy"] -= elemental_energy_fn(sample.structure)
    for sample in valid_samples:
        sample.labels["energy"] -= elemental_energy_fn(sample.structure)

else:
    baseline = {}


# -- assemble (pre-)batches --
from marathon.data import batch_samples, determine_max_sizes
from marathon.utils import tree_split_first_dim, tree_stack

reporter.step("data preparation")

# we make fake batches with one structure, then shuffle+reshape them into batches later.
# this has the advantage of being jittable and happening entirely on the GPU, so
# it avoids data transfer and the juggling of assembling batches fast enough to
# keep the GPU fed, but it assumes that all samples are similar. maybe revisit later
train_num_nodes, train_num_edges = determine_max_sizes(train_samples, 1)
valid_num_nodes, valid_num_edges = determine_max_sizes(valid_samples, 1)

train_pre_batches = [
    batch_samples([s], train_num_nodes, train_num_edges, keys) for s in train_samples
]
valid_pre_batches = [
    batch_samples([s], valid_num_nodes, valid_num_edges, keys) for s in valid_samples
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
        msg.append(f"{k}: {old_loss_weights[k]:.3f} -> {v:.3e}")
    comms.state(msg, title="variance scaled loss weights")


# -- optimizer --
import optax
from optax.contrib import reduce_on_plateau

reporter.step("setup optimizer")

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
initial_opt_state = optimizer.init(params)


# -- assemble state / handle restore --

state = {
    "step": 0,
    "checkpointers": checkpointers,
    "opt_state": initial_opt_state,
    "key": key,
}

if workdir.is_dir():
    from marathon.emit import get_latest

    comms.warn(
        f"found working directory {workdir}, will restore (only) model and optimisation state!"
    )
    reporter.step("restoring")

    items = get_latest(workdir, state)

    if items is None:
        comms.warn(f"failed to find checkpoints in workdir {workdir}, ignoring")
    else:
        params, state, new_model, _, _, _ = items

        comms.talk(f"restored epoch {state['step']}")

        # try to catch the most obvious error: editing the model config between restarts
        from marathon.io import to_dict

        assert to_dict(new_model) == to_dict(model)
else:
    workdir.mkdir()

opt_state = state["opt_state"]


# -- loggers --
from marathon.emit import Txt
from marathon.io import to_dict

reporter.step("setup loggers")

config = {
    "n_train": n_train,
    "n_valid": n_valid,
    "loss_weights": loss_weights,
    "scale_by_variance": scale_by_variance,
    "max_epochs": max_epochs,
    "start_learning_rate": start_learning_rate,
    "min_learning_rate": min_learning_rate,
    "lr_decay_patience": lr_decay_patience,
    "lr_decay_factor": lr_decay_factor,
    "lr_decay_rtol": lr_decay_rtol,
    "lr_decay_atol": lr_decay_atol,
    "lr_decay_cooldown": lr_decay_cooldown,
    "lr_accumulation_epochs": lr_accumulation_epochs,
    "train_batch_size": train_batch_size,
    "valid_batch_size": valid_batch_size,
    "valid_every": valid_every,
    "chunk_size": chunk_size,
    "model": to_dict(model),
    "num_parameters": num_parameters,
}


metrics = {key: ["r2", "mae", "rmse"] for key in keys}

loggers = [Txt(metrics=metrics)]


if use_wandb:
    import wandb

    from marathon.emit import WandB

    this_folder = Path(__file__).parent

    if wandb_project is None:
        wandb_project = f"{this_folder.parent.parent.stem}.{this_folder.parent.stem}"

    if wandb_name is None:
        wandb_name = f"{this_folder.stem}"

    run = wandb.init(config=config, name=wandb_name, project=wandb_project)

    config["wandb_id"] = run.id

    loggers.append(WandB(run, metrics=metrics))


# -- setup actual training loop --
from time import monotonic

from marathon.emit import save_checkpoints
from marathon.evaluate import get_loss_fn, get_metrics_fn, get_predict_fn
from marathon.utils import seconds_to_string as s2s

reporter.step("setup training loop")

pred_fn = get_predict_fn(
    model.apply,
    stress=use_stress,
)
loss_fn = get_loss_fn(pred_fn, weights=loss_weights)

train_metrics_fn = get_metrics_fn(train_samples, keys=keys)
valid_metrics_fn = get_metrics_fn(valid_samples, keys=keys)

# ... manager preamble


def get_lr(opt_state):
    return float(start_learning_rate * opt_state[1].scale)


def report_on_lr(opt_state):
    lr = get_lr(opt_state)
    best = opt_state[1].best_value
    return f"LR decay: lr {lr:.3e}, best loss {best:.3e}"


from marathon.emit.pretty import format_metrics


class Manager:
    def __init__(self, state, interval, loggers, workdir, model, baseline, max_epochs):
        self.state = state
        self.interval = interval
        self.loggers = loggers
        self.workdir = workdir
        self.model = model
        self.baseline = baseline

        self.max_epochs = max_epochs

        self.start_epoch = state["step"]
        self.start_time = monotonic()

        self.cancel = False

    @property
    def done(self):
        return self.epoch >= self.max_epochs or self.cancel

    @property
    def epoch(self):
        return self.state["step"]

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
        self.state["step"] += self.interval
        self.state["opt_state"] = opt_state
        self.state["key"] = key

        if jnp.isnan(train_loss):
            comms.warn(f"loss became NaN at epoch={self.epoch}, canceling training")
            self.cancel = True

        if get_lr(opt_state) < min_learning_rate:
            comms.talk(
                f"learning rate has reached minimum at epoch={self.epoch}, canceling",
                full=True,
            )
            self.cancel = True

        # bail out here -- this may get called a few times after finishing as the GPU wraps up,
        # this causes problems for logging, so we just ignore it
        if self.done:
            return

        info = {"lr": get_lr(opt_state), "time_per_epoch": self.time_per_epoch}

        for logger in self.loggers:
            logger(
                self.state["step"],
                train_loss,
                train_metrics,
                valid_loss,
                valid_metrics,
                other=info,
            )

        metrics = {"train": train_metrics, "valid": valid_metrics}
        metrics = jax.tree_util.tree_map(lambda x: np.array(x), metrics)

        save_checkpoints(
            metrics,
            params,
            self.state,
            self.model,
            self.baseline,
            self.workdir,
            config=config,
        )

        title = f"state at epoch: {self.epoch}"
        msg = []

        msg.append(f"train loss: {train_loss:.5e}")
        msg.append(f"valid loss: {valid_loss:.5e}")

        msg.append(report_on_lr(opt_state))

        msg.append("validation errors:")
        msg += format_metrics(metrics["valid"], keys=keys)

        msg.append("")
        msg.append(f"elapsed: {s2s(self.elapsed, 's')}")
        msg.append(f"timing: {s2s(self.time_per_epoch)}/epoch, {s2s(self.eta, 'm')} ETA")

        msg.append("")
        comms.state(msg, title=title)


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


# -- train! --
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
from marathon.emit import get_all, plot

reporter.step("wrapup")

pred_fn = jax.jit(pred_fn)


def predict_and_collate(params, batches):
    # to avoid running out of VRAM, we iterate one
    # structure at a time, and use the chance to also
    # collect the correct labels, dropping masked items

    predictions = {k: [] for k in keys}
    labels = {k: [] for k in keys}
    n_atoms = []

    for batch in batches:
        n_atoms.append(batch.node_mask.sum())
        preds = pred_fn(params, batch)

        for key in keys:
            mask = batch.labels[key + "_mask"]
            if mask.any():
                predictions[key].append(preds[key][mask])
                labels[key].append(batch.labels[key][mask])

    n_atoms = np.array(n_atoms)

    final_predictions = {}
    final_labels = {}

    for key in predictions.keys():
        if "energy" in key:
            final_predictions[key] = np.array(predictions[key]).flatten() / n_atoms
        if "forces" in key:
            final_predictions[key] = np.array(predictions[key]).reshape(-1, 3)
        if "stress" in key:
            final_predictions[key] = np.array(predictions[key]).reshape(-1, 3, 3)

    for key in keys:
        if key == "energy":
            final_labels[key] = np.array(labels[key]).flatten() / n_atoms
        if key == "forces":
            final_labels[key] = np.array(labels[key]).reshape(-1, 3)
        if key == "stress":
            final_labels[key] = np.array(labels[key]).reshape(-1, 3, 3)

    return final_labels, final_predictions


for f, items in get_all(workdir, state):
    if f.suffix == ".backup":
        continue

    comms.talk(f"working on {f}")

    params, _, _, _, metrics, _ = items

    for batches, split in [[valid_pre_batches, "valid"]]:
        labels, predictions = predict_and_collate(params, batches)

        out = f / f"plot/{split}"
        out.mkdir(parents=True, exist_ok=True)

        plot(out, predictions, labels, metrics=metrics[split])


reporter.done()
if use_wandb:
    run.finish()

comms.talk("cleaning up")
import shutil

if use_wandb:
    wandb_dir = Path("wandb")
    if wandb_dir.is_dir():
        shutil.rmtree(wandb_dir)

for f, items in get_all(workdir, state):
    if f.suffix == ".backup":
        shutil.rmtree(f)

comms.state("done!")
