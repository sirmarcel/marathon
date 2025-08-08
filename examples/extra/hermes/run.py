if __name__ == "__main__":
    # -- settings --

    from marathon.data import datasets

    data_train = datasets / ...
    data_valid = datasets / ...

    # settings for how to construct batches:
    # ..  batch_shape: fixed shape, varied number of samples
    # ..  batch_length: varying shape, fixed number of samples
    # ..  batch_dense: dense NLs, fixed nodes + graphs
    #                             edges configurable

    batch_style = "batch_length"

    num_graphs = 2
    num_nodes = 2**9
    num_edges = 2**17
    num_neighbors = None
    num_neighbors_multiple = 4

    # if chunk_size > 1, multiple batches will be stacked
    # into chunks and scanned over. requires constant batch size
    chunk_size = 1

    loss_weights = {"energy": 0.001, "forces": 0.999}
    scale_by_variance = True

    start_learning_rate = 2e-3
    min_learning_rate = 1e-6

    # measured in *epochs*
    max_epochs = 1000
    valid_every_epoch = 2

    # lr decay
    decay_style = "exponential"
    start_decay_after = 2
    stop_decay_after = max_epochs  # ignored for exponential

    seed = 0
    print_model_summary = True
    benchmark_pipeline = True
    workdir = "run"

    use_wandb = False
    # used for wandb -- use folder names by default
    wandb_project = None
    wandb_name = None

    default_matmul_precision = "default"
    debug_nans = False  # may cause slowdown?

    # settings for grain
    worker_count = 2
    worker_buffer_size = 2

    # -- imports & startup --

    import numpy as np
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_default_matmul_precision", default_matmul_precision)
    jax.config.update("jax_debug_nans", debug_nans)

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

    name = "MAE_" + "+".join([k[0].upper() for k in ["forces"]])
    checkpointers.append(SummedMetric(name, "mae", keys=["forces"]))

    checkpointers = tuple(checkpointers)

    # -- data loading --
    from marathon.evaluate.metrics import get_stats
    from marathon.extra.hermes import (
        DataLoader,
        DataSource,
        FilterEmpty,
        IndexSampler,
        ToDenseBatch,
        ToFixedLengthBatch,
        ToFixedShapeBatch,
        ToSample,
        ToStack,
        prefetch_to_device,
    )
    from marathon.extra.hermes.pain import Record, RecordMetadata

    to_sample = ToSample(cutoff=cutoff, energy=True, forces=True, stress=use_stress)

    source_train = DataSource(data_train)
    species_to_weight = source_train.species_to_weight
    source_valid = DataSource(data_valid, species_to_weight=species_to_weight)
    baseline = {"elemental": source_valid.species_to_weight}
    n_train = len(source_train)
    n_valid = len(source_valid)

    max_steps = max_epochs * n_train
    valid_every = valid_every_epoch * n_train
    comms.talk(f"run for {max_epochs} epochs, {max_steps} steps", full=True)
    comms.talk(
        f"validate every {valid_every_epoch} epochs, every {valid_every} steps", full=True
    )

    reporter.step("loading validation set")

    # for now we assume that validation set fits into RAM easily
    valid_samples = []

    if batch_style == "batch_shape":
        batcher = ToFixedShapeBatch(
            num_graphs=num_graphs, num_edges=num_edges, num_nodes=num_nodes
        )
    elif batch_style == "batch_length":
        batcher = ToFixedLengthBatch(
            batch_size=num_graphs,
        )
    elif batch_style == "batch_dense":
        batcher = ToDenseBatch(
            num_nodes=num_nodes,
            num_neighbors=num_neighbors,
            num_graphs=num_graphs,
            num_neighbors_multiple=num_neighbors_multiple,
        )
    else:
        raise ValueError

    if chunk_size > 1:
        if batch_style == "batch_shape":
            pass
        elif batch_style == "dense" and num_neighbors is not None:
            pass
        else:
            raise ValueError("chunk_size > 1 but variable-shape batcher")

    def valid_iterator():
        filterer = FilterEmpty()
        for i in range(n_valid):
            sample = to_sample.map(source_valid[i])
            if filterer.filter(sample):
                valid_samples.append(sample)
                yield Record(data=sample, metadata=RecordMetadata(index=i, record_key=i))

    data_valid = [b.data for b in batcher(valid_iterator())]
    valid_stats = get_stats(valid_samples, keys=keys)

    valid_batch_sizes = np.array([batch.graph_mask.sum() for batch in data_valid])
    median_valid_batch_size = int(np.median(valid_batch_sizes))

    if scale_by_variance:
        old_loss_weights = loss_weights

        loss_weights = {k: v / valid_stats[k]["var"] for k, v in loss_weights.items()}

        msg = []
        for k, v in loss_weights.items():
            msg.append(f"{k}: {old_loss_weights[k]:.3f} -> {v:.3f}")
        comms.state(msg, title="variance scaled loss weights")

    del valid_samples

    reporter.step("setup training pipeline")

    def get_training_iterator(num_epochs):
        if batch_style == "batch_shape":
            batchers = [
                ToFixedShapeBatch(
                    num_graphs=num_graphs, num_edges=num_edges, num_nodes=num_nodes
                )
            ]
        elif batch_style == "batch_length":
            batchers = [
                ToFixedLengthBatch(
                    batch_size=num_graphs,
                )
            ]
        elif batch_style == "batch_dense":
            batchers = [
                ToDenseBatch(
                    num_nodes=num_nodes,
                    num_neighbors=num_neighbors,
                    num_graphs=num_graphs,
                    num_neighbors_multiple=num_neighbors_multiple,
                )
            ]
        else:
            raise ValueError

        if chunk_size > 1:
            batchers.append(ToStack(batch_size=chunk_size, drop_remainder=True))

        return iter(
            DataLoader(
                data_source=source_train,
                sampler=IndexSampler(
                    n_train,
                    num_epochs=num_epochs,
                ),
                operations=[
                    to_sample,
                    FilterEmpty(),
                    *batchers,
                ],
                worker_count=worker_count,
                worker_buffer_size=worker_buffer_size,
            )
        )

    if benchmark_pipeline:
        from time import monotonic

        reporter.step("benchmark training pipeline", spin=False)

        @jax.jit
        def test_fn(batch):
            if chunk_size == 1:
                return (
                    batch.edge_mask.sum(),
                    batch.node_mask.sum(),
                    batch.graph_mask.sum(),
                    batch.edge_mask.shape[0],
                    batch.node_mask.shape[0],
                    batch.graph_mask.shape[0],
                )
            else:
                return (
                    batch.edge_mask.sum(),
                    batch.node_mask.sum(),
                    batch.graph_mask.sum(),
                    batch.edge_mask.shape[0] * batch.edge_mask.shape[1],
                    batch.node_mask.shape[0] * batch.node_mask.shape[1],
                    batch.graph_mask.shape[0] * batch.graph_mask.shape[1],
                )

        # trigger jit
        test_fn(next(get_training_iterator(1)))

        test_iter = prefetch_to_device(get_training_iterator(1), 2)

        results = []
        start = monotonic()
        for i, batch in enumerate(test_iter):
            reporter.tick(f"chunk {i}")
            results.append(test_fn(batch))
            del batch
        results = np.array(results)
        duration = monotonic() - start

        real_samples = results[:, 2].sum()
        util_edges = 100 * results[:, 0] / results[:, 3]
        util_nodes = 100 * results[:, 1] / results[:, 4]
        util_samples = 100 * results[:, 2] / results[:, 5]
        pipeline_speed = duration / real_samples

        unique_edges = np.unique(results[:, 3]).shape[0]
        unique_nodes = np.unique(results[:, 4]).shape[0]
        unique_samples = np.unique(results[:, 4]).shape[0]

        num_chunks = i + 1
        num_batches = num_chunks * chunk_size

        msg = []
        msg.append(f"speed       : {1e6*pipeline_speed:.0f}µs/sample")
        msg.append(f"              {worker_count} workers, buffer {worker_buffer_size}")
        msg.append(
            f"edges  : {np.mean(util_edges):.2f}% / {np.mean(results[:, 0]):.0f} mean"
        )
        msg.append(
            f"nodes  : {np.mean(util_nodes):.2f}% / {np.mean(results[:, 1]):.0f} mean"
        )
        msg.append(
            f"samples: {np.mean(util_samples):.2f}% / {np.mean(results[:, 2]):.0f} mean"
        )

        msg.append("")
        msg.append(
            f"unique shapes: {unique_edges} edges, {unique_nodes} nodes, {unique_samples} samples"
        )
        msg.append(
            f"... -> expecting {unique_edges*unique_nodes*unique_samples} compilations"
        )
        msg.append("")
        if chunk_size > 1:
            msg.append(f"num chunks: {num_chunks} containing {chunk_size} batches")
        msg.append(
            f"num batches: {num_batches} ({real_samples/num_batches:.0f} samples/batch)"
        )

        comms.state(msg, title="Training Pipeline Statistics")

        if np.mean(util_edges) < 50 or np.mean(util_nodes) < 50:
            comms.warn("Ratio of real to padded edges or nodes is TOO LOW (<50%). No!")
            comms.warn("I SHOULD REFUSE TO CONTINUE WITH THIS SICK JOB ...")

        median_train_batch_size = int(np.median(results[:, 2]) / chunk_size)

        median_batch_size = median_train_batch_size
        batches_per_epoch = num_batches
    else:
        pipeline_speed = 0.0
        median_batch_size = median_valid_batch_size
        batches_per_epoch = int(len(source_train) / median_batch_size)

    comms.talk(f"estimated samples/batch: {median_batch_size}")
    comms.talk(f"estimated batches/epoch: {batches_per_epoch}")

    iter_train = get_training_iterator(max_epochs)

    # -- optimizer --
    import optax

    reporter.step("setup optimizer")

    if decay_style == "linear":
        transition_steps = stop_decay_after * batches_per_epoch
        initial_steps = start_decay_after * batches_per_epoch
        scheduler = optax.schedules.linear_schedule(
            init_value=start_learning_rate,
            end_value=min_learning_rate,
            transition_begin=initial_steps,
            transition_steps=transition_steps - initial_steps,
        )

    elif decay_style == "exponential":
        transition_steps = max_epochs * batches_per_epoch
        initial_steps = start_decay_after * batches_per_epoch
        decay_rate = min_learning_rate / start_learning_rate
        scheduler = optax.schedules.exponential_decay(
            init_value=start_learning_rate,
            transition_steps=transition_steps - initial_steps,
            transition_begin=initial_steps,
            decay_rate=decay_rate,
            end_value=min_learning_rate,
        )

    @optax.inject_hyperparams
    def optimizer(learning_rate):
        return optax.lamb(learning_rate)

    optimizer = optimizer(scheduler)

    initial_opt_state = optimizer.init(params)

    # -- assemble state / handle restore --

    state = {
        "step": 0,
        "checkpointers": checkpointers,
        "opt_state": initial_opt_state,
        # "key": key,
        "iter_train": iter_train.get_state(),
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

            comms.talk(f"restored step {state['step']}")

            # try to catch the most obvious error: editing the model config between restarts
            from marathon.io import to_dict

            assert to_dict(new_model) == to_dict(model)

            iter_train.set_state(state["iter_train"])
    else:
        workdir.mkdir()

    opt_state = state["opt_state"]

    # -- loggers --
    from marathon.io import to_dict

    from marathon.emit import Txt

    reporter.step("setup loggers")

    if batch_style == "batch_shape":
        training_pipeline = {
            "style": "shape",
            "num_graphs": num_graphs,
            "num_edges": num_edges,
            "num_nodes": num_nodes,
        }
    elif batch_style == "batch_length":
        training_pipeline = {
            "style": "length",
            "num_graphs": num_graphs,
        }
    elif batch_style == "batch_dense":
        training_pipeline = {
            "style": "dense",
            "num_graphs": num_graphs,
            "num_neighbors": num_neighbors,
            "num_nodes": num_nodes,
            "num_neighbors_multiple": num_neighbors_multiple,
        }
    else:
        raise ValueError

    if decay_style == "linear":
        lr_decay = {
            "style": "linear",
            "start_decay_after": start_decay_after,
            "stop_decay_after": stop_decay_after,
        }
    elif decay_style == "exponential":
        lr_decay = {"style": "exponential", "start_decay_after": start_decay_after}
    else:
        raise ValueError

    config = {
        "n_train": n_train,
        "n_valid": n_valid,
        "loss_weights": loss_weights,
        "max_steps": max_steps,
        "start_learning_rate": start_learning_rate,
        "min_learning_rate": min_learning_rate,
        "lr_decay": lr_decay,
        "chunk_size": chunk_size,
        "training_pipeline": training_pipeline,
        "valid_every": valid_every,
        "model": to_dict(model),
        "num_parameters": num_parameters,
        "worker_count": worker_count,
        "worker_buffer_size": worker_buffer_size,
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
    from marathon.utils import s_to_string, tree_concatenate, tree_stack

    reporter.step("setup training loop")

    if hasattr(model, "energy"):
        comms.talk("using custom energy function for model")
        pred_fn = get_predict_fn(
            energy_fn=model.energy,
            stress=use_stress,
        )
    else:
        pred_fn = get_predict_fn(
            apply_fn=model.apply,
            stress=use_stress,
        )

    loss_fn = get_loss_fn(pred_fn, weights=loss_weights)
    loss_fn = jax.jit(loss_fn)

    train_metrics_fn = get_metrics_fn(keys=keys)  # no stats
    valid_metrics_fn = get_metrics_fn(keys=keys, stats=valid_stats)

    # ... manager preamble

    def get_lr(opt_state):
        return float(opt_state.hyperparams["learning_rate"])

    def report_on_lr(opt_state):
        lr = get_lr(opt_state)
        return f"LR: {lr:.3e}"

    def format_metrics(metrics, keys=["energy", "forces"]):
        key_to_unit = {"energy": "meV/atom", "forces": "meV/Å", "stress": "meV"}
        key_to_name = {"energy": "E", "forces": "F", "stress": "σ"}
        msg = []

        for key in keys:
            m = metrics[key]

            msg.append(f". {key_to_name[key]}")
            if "r2" in m:
                msg.append(f".. R2  : {m['r2']:.3f} %")
            msg.append(f".. MAE : {m['mae']:.3e} {key_to_unit[key]}")
            msg.append(f".. RMSE: {m['rmse']:.3e} {key_to_unit[key]}")

        return msg

    class Manager:
        def __init__(self, state, interval, loggers, workdir, model, baseline, max_steps):
            self.state = state
            self.interval = interval
            self.loggers = loggers
            self.workdir = workdir
            self.model = model
            self.baseline = baseline

            self.max_steps = max_steps

            self.start_step = state["step"]
            self.start_time = monotonic()

            self.cancel = False

        @property
        def done(self):
            return self.step >= self.max_steps or self.cancel

        @property
        def step(self):
            return self.state["step"]

        @property
        def elapsed(self):
            return monotonic() - self.start_time

        @property
        def time_per_step(self):
            return self.elapsed / (self.step - self.start_step)

        @property
        def compute_time_per_step(self):
            return self.time_per_step - pipeline_speed

        @property
        def eta(self):
            return (self.max_steps - self.step) * self.time_per_step

        def should_validate(self, step):
            return step >= self.step + self.interval

        def report(
            self,
            step,
            params,
            opt_state,
            train_state,
            train_loss,
            train_metrics,
            valid_loss,
            valid_metrics,
            info={},
        ):
            assert step > self.step  # always forward

            self.state["step"] = step
            self.state["opt_state"] = opt_state
            self.state["iter_train"] = train_state

            if jnp.isnan(train_loss):
                comms.warn(f"loss became NaN at step={self.step}, canceling training")
                self.cancel = True

            if get_lr(opt_state) < min_learning_rate:
                # sometimes we stop decay before max_steps, in that case don't break
                if stop_decay_after == max_epochs:
                    comms.talk(
                        f"learning rate has reached minimum at steps={self.step}, canceling"
                    )
                    self.cancel = True

            info = {
                "lr": get_lr(opt_state),
                "time_per_step": self.time_per_step,
                "compute_time_per_step": self.compute_time_per_step,
                **info,
            }

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

            title = f"state at step: {self.step}"
            msg = []

            msg.append(f"train loss: {train_loss:.5e}")
            msg.append(f"valid loss: {valid_loss:.5e}")

            msg.append(report_on_lr(opt_state))

            msg.append("validation errors:")
            msg += format_metrics(metrics["valid"], keys=keys)

            msg.append("")
            msg.append(f"elapsed: {s_to_string(self.elapsed, 's')}")
            msg.append(
                f"timing: {s_to_string(self.time_per_step)}/step, {s_to_string(self.eta, 'm')} ETA"
            )

            msg.append("")
            comms.state(msg, title=title)

    manager = Manager(state, valid_every, loggers, workdir, model, baseline, max_steps)

    @jax.jit
    def _do_batch(carry, batch):
        params, opt_state = carry

        loss_and_aux, grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
            params, batch
        )
        loss, aux = loss_and_aux
        updates, opt_state = optimizer.update(grads, opt_state, params, value=loss)

        params = optax.apply_updates(params, updates)

        return (params, opt_state), (loss, aux)

    if chunk_size == 1:
        do_batch = _do_batch
        aggregate_loss = np.mean
        aggregate_aux = tree_stack
    else:

        @jax.jit
        def do_batch(carry, batches):
            return jax.lax.scan(_do_batch, carry, batches)

        aggregate_loss = np.mean
        aggregate_aux = tree_concatenate

    # -- train! --
    import itertools

    reporter.step("🚄", spin=False)

    start = monotonic()

    iter_train_with_prefetch = prefetch_to_device(iter_train, 2)
    iter_valid_with_prefetch = prefetch_to_device(itertools.cycle(data_valid), 2)

    ran_steps = 0
    train_aux = []
    train_loss = []
    report = None
    cache_size = 1
    while True:
        try:
            batch = next(iter_train_with_prefetch)
        except StopIteration:
            comms.talk("exhausted training iterator")
            # break
            manager.cancel = True

        if not manager.done:
            ran_steps += batch.graph_mask.sum()
            (params, opt_state), (loss, aux) = do_batch((params, opt_state), batch)
            current_step = manager.step + ran_steps
            del batch
            reporter.tick(f"{current_step}")

            if do_batch._cache_size() > cache_size:
                cache_size = do_batch._cache_size()
                comms.talk(f"recompiled at step={current_step} ({cache_size})")

            train_aux.append(aux)
            train_loss.append(loss)

        if report is not None:
            manager.report(*report)
            report = None

        if manager.done:
            break

        if manager.should_validate(manager.step + ran_steps):
            valid_aux = []
            valid_loss = []
            for i in range(len(data_valid)):
                batch = next(iter_valid_with_prefetch)
                reporter.tick(f"{current_step} (valid {i})")
                loss, aux = loss_fn(params, batch)
                valid_aux.append(aux)
                valid_loss.append(loss)

            train_aux = aggregate_aux(train_aux)
            train_metrics = train_metrics_fn(train_aux)

            valid_aux = tree_stack(valid_aux)
            valid_metrics = valid_metrics_fn(valid_aux)

            train_loss = aggregate_loss(train_loss)
            valid_loss = np.mean(valid_loss)

            report = (
                manager.step + ran_steps,
                params,
                opt_state,
                iter_train.get_state(),
                train_loss,
                train_metrics,
                valid_loss,
                valid_metrics,
                {
                    "compiles_do_batch": do_batch._cache_size(),
                    "compiles_loss_fn": loss_fn._cache_size(),
                },
            )

            train_aux = []
            train_loss = []
            ran_steps = 0

    # -- wrap up --
    from marathon.emit import get_all

    reporter.step("wrapup")
    # todo: consider doing wrapup plots here

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
