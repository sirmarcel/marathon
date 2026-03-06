"""
Example training script using marathon.grain for scalable data loading.

This is a simplified version of production training code, demonstrating:
- DataSource for mmap-based data access
- grain-based parallel data loading
- Training loop with validation and checkpointing
"""

if __name__ == "__main__":
    # -- settings --
    # In a real application, load these from a config file

    from pathlib import Path

    data_train = Path("data_train")  # Path to mmap data (created by prepare_data.py)
    data_valid = Path("data_valid")

    # Batching: "batch_shape" = fixed shape, "batch_length" = fixed sample count
    batch_style = "batch_length"

    # For batch_shape:
    num_nodes = 2**9
    num_edges = 2**17
    # For batch_length:
    num_graphs = 2  # matches train_batch_size in train_plain

    loss_weights = {"energy": 1.0, "forces": 1.0, "stress": 1.0}
    scale_by_variance = False
    remove_baseline = False  # remove per-element energy offset (set True for production)

    start_learning_rate = 1e-3
    min_learning_rate = 1e-6

    max_epochs = 300
    valid_every_epoch = 1

    # Learning rate decay
    decay_style = "exponential"  # or "linear", "warmup_cosine"
    start_decay_after = 10
    stop_decay_after = max_epochs

    seed = 0
    print_model_summary = True
    benchmark_pipeline = True
    workdir = "run"

    use_wandb = False

    # grain settings
    worker_count = 4
    worker_buffer_size = 2

    # -- imports & startup --

    import numpy as np
    import jax
    import jax.numpy as jnp

    from pathlib import Path
    from time import monotonic

    from marathon import comms

    reporter = comms.reporter()
    reporter.start("run")
    reporter.step("startup")

    keys = list(loss_weights.keys())
    use_stress = "stress" in keys

    workdir = Path(workdir)

    # -- randomness --
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)

    # -- model --
    from marathon.io import from_dict, read_yaml

    model_config = read_yaml("model.yaml")
    model = from_dict(model_config["model"])
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

    name = "MAE_F"
    checkpointers.append(SummedMetric(name, "mae", keys=["forces"]))

    checkpointers = tuple(checkpointers)

    # -- data loading --
    from marathon.evaluate.metrics import get_stats
    from marathon.grain import (
        DataLoader,
        DataSource,
        FilterEmpty,
        IndexSampler,
        Record,
        RecordMetadata,
        ToFixedLengthBatch,
        ToFixedShapeBatch,
        ToSample,
        prefetch_to_device,
    )
    from marathon.utils import tree_stack

    to_sample = ToSample(cutoff=cutoff, energy=True, forces=True, stress=use_stress)

    source_train = DataSource(data_train, remove_baseline=remove_baseline)
    source_valid = DataSource(data_valid, remove_baseline=remove_baseline)
    baseline = {"elemental": source_train.species_to_weight} if remove_baseline else {}
    n_train = len(source_train)
    n_valid = len(source_valid)

    max_steps = max_epochs * n_train
    valid_every = valid_every_epoch * n_train
    comms.talk(f"run for {max_epochs} epochs, {max_steps} steps", full=True)
    comms.talk(
        f"validate every {valid_every_epoch} epochs, every {valid_every} steps", full=True
    )

    # -- setup batchers --
    if batch_style == "batch_shape":
        batcher = ToFixedShapeBatch(
            num_graphs=num_graphs,
            num_edges=num_edges,
            num_nodes=num_nodes,
            keys=tuple(keys),
        )
    elif batch_style == "batch_length":
        batcher = ToFixedLengthBatch(batch_size=num_graphs, keys=tuple(keys))
    else:
        raise ValueError(f"Unknown batch_style: {batch_style}")

    # -- validation set --
    reporter.step("loading validation set")

    valid_samples = []
    filter_empty = FilterEmpty()

    def valid_iterator():
        for i in range(n_valid):
            sample = to_sample.map(source_valid[i])
            if filter_empty.filter(sample):
                valid_samples.append(sample)
                yield Record(data=sample, metadata=RecordMetadata(index=i, record_key=i))

    data_valid_batches = [b.data for b in batcher(valid_iterator())]
    valid_stats = get_stats(valid_samples, keys=keys)

    valid_batch_sizes = np.array(
        [batch.structure_mask.sum() for batch in data_valid_batches]
    )
    median_valid_batch_size = int(np.median(valid_batch_sizes))

    if scale_by_variance:
        old_loss_weights = loss_weights
        loss_weights = {k: v / valid_stats[k]["var"] for k, v in loss_weights.items()}

        msg = []
        for k, v in loss_weights.items():
            msg.append(f"{k}: {old_loss_weights[k]:.3f} -> {v:.3f}")
        comms.state(msg, title="variance scaled loss weights")

    del valid_samples

    # -- training iterator --
    reporter.step("setup training pipeline")

    def get_training_iterator(num_epochs):
        if batch_style == "batch_shape":
            batchers = [
                ToFixedShapeBatch(
                    num_graphs=num_graphs,
                    num_edges=num_edges,
                    num_nodes=num_nodes,
                    keys=tuple(keys),
                )
            ]
        else:
            batchers = [ToFixedLengthBatch(batch_size=num_graphs, keys=tuple(keys))]

        return iter(
            DataLoader(
                data_source=source_train,
                sampler=IndexSampler(n_train, num_epochs=num_epochs, seed=seed),
                operations=[to_sample, FilterEmpty(), *batchers],
                worker_count=worker_count,
                worker_buffer_size=worker_buffer_size,
            )
        )

    # -- benchmark pipeline (optional) --
    if benchmark_pipeline:
        reporter.step("benchmark training pipeline", spin=False)

        @jax.jit
        def test_fn(batch):
            return (
                batch.pair_mask.sum(),
                batch.atom_mask.sum(),
                batch.structure_mask.sum(),
                batch.pair_mask.shape[0],
                batch.atom_mask.shape[0],
                batch.structure_mask.shape[0],
            )

        test_fn(next(get_training_iterator(1)))  # trigger jit

        test_iter = prefetch_to_device(get_training_iterator(1), 2)
        results = []
        start = monotonic()
        for i, batch in enumerate(test_iter):
            reporter.tick(f"batch {i}")
            results.append(test_fn(batch))
            del batch
        results = np.array(results)
        duration = monotonic() - start

        real_samples = results[:, 2].sum()
        pipeline_speed = duration / real_samples
        num_batches = i + 1

        msg = []
        msg.append(f"speed: {1e6 * pipeline_speed:.0f}µs/sample")
        msg.append(
            f"utilization (pairs): {100 * results[:, 0].sum() / results[:, 3].sum():.1f}%"
        )
        msg.append(
            f"utilization (atoms): {100 * results[:, 1].sum() / results[:, 4].sum():.1f}%"
        )
        msg.append(
            f"num batches: {num_batches} ({real_samples / num_batches:.0f} samples/batch)"
        )
        comms.state(msg, title="Training Pipeline Statistics")

        median_batch_size = int(np.median(results[:, 2]))
        batches_per_epoch = num_batches
    else:
        pipeline_speed = 0.0
        median_batch_size = median_valid_batch_size
        batches_per_epoch = int(n_train / median_batch_size)

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
    elif decay_style == "warmup_cosine":
        warmup_epochs = start_decay_after
        scheduler = optax.schedules.warmup_cosine_decay_schedule(
            init_value=min_learning_rate,
            peak_value=start_learning_rate,
            warmup_steps=warmup_epochs * batches_per_epoch,
            decay_steps=max_epochs * batches_per_epoch,
            end_value=min_learning_rate,
        )
    else:
        raise ValueError(f"Unknown decay_style: {decay_style}")

    @optax.inject_hyperparams
    def optimizer(learning_rate):
        return optax.adam(learning_rate)

    optimizer = optimizer(scheduler)
    initial_opt_state = optimizer.init(params)

    # -- state / restore --
    state = {
        "step": 0,
        "checkpointers": checkpointers,
        "opt_state": initial_opt_state,
        "iter_train": iter_train.get_state(),
    }

    if workdir.is_dir():
        from marathon.emit import get_latest

        comms.warn(f"found working directory {workdir}, will restore!")
        reporter.step("restoring")

        items = get_latest(workdir, state)
        if items is None:
            comms.warn(f"failed to find checkpoints in workdir {workdir}, ignoring")
        else:
            params, state, new_model, _, _, _ = items
            comms.talk(f"restored step {state['step']}")

            from marathon.io import to_dict

            assert to_dict(new_model) == to_dict(model)
            iter_train.set_state(state["iter_train"])
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
        "max_steps": max_steps,
        "max_epochs": max_epochs,
        "start_learning_rate": start_learning_rate,
        "min_learning_rate": min_learning_rate,
        "decay_style": decay_style,
        "batch_style": batch_style,
        "valid_every": valid_every,
        "model": to_dict(model),
        "num_parameters": num_parameters,
    }

    metrics = {key: ["r2", "mae", "rmse"] for key in keys}
    loggers = [Txt(metrics=metrics)]

    if use_wandb:
        import wandb

        from marathon.emit import WandB

        run = wandb.init(config=config)
        config["wandb_id"] = run.id
        loggers.append(WandB(run, metrics=metrics))

    # -- training loop setup --
    from marathon.emit import save_checkpoints
    from marathon.emit.pretty import format_metrics
    from marathon.evaluate import get_loss_fn, get_metrics_fn, get_predict_fn
    from marathon.utils import seconds_to_string as s2s

    reporter.step("setup training loop")

    if hasattr(model, "predict"):
        pred_fn = lambda params, batch: model.predict(params, batch, stress=use_stress)
    elif hasattr(model, "energy"):
        pred_fn = get_predict_fn(energy_fn=model.energy, stress=use_stress)
    else:
        pred_fn = get_predict_fn(apply_fn=model.apply, stress=use_stress)

    loss_fn = get_loss_fn(pred_fn, weights=loss_weights)
    loss_fn = jax.jit(loss_fn)

    train_metrics_fn = get_metrics_fn(keys=keys)
    valid_metrics_fn = get_metrics_fn(keys=keys, stats=valid_stats)

    def get_lr(opt_state):
        return float(opt_state.hyperparams["learning_rate"])

    @jax.jit
    def do_batch(carry, batch):
        params, opt_state = carry
        loss_and_aux, grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
            params, batch
        )
        loss, aux = loss_and_aux
        updates, opt_state = optimizer.update(grads, opt_state, params, value=loss)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (loss, aux)

    # -- train! --
    reporter.step("🚄", spin=False)

    start_time = monotonic()
    start_step = state["step"]

    iter_train_with_prefetch = prefetch_to_device(iter_train, 2)
    iter_valid_with_prefetch = None

    ran_steps = 0
    train_aux = []
    train_loss = []
    done = False

    while not done:
        try:
            batch = next(iter_train_with_prefetch)
        except StopIteration:
            comms.talk("exhausted training iterator")
            done = True
            continue

        ran_steps += batch.structure_mask.sum()
        (params, opt_state), (loss, aux) = do_batch((params, opt_state), batch)
        current_step = state["step"] + ran_steps
        del batch
        reporter.tick(f"{current_step}")

        train_aux.append(aux)
        train_loss.append(loss)

        if current_step >= max_steps:
            done = True

        if jnp.isnan(loss):
            comms.warn(f"loss became NaN at step={current_step}, stopping")
            done = True

        # Validation
        if ran_steps >= valid_every or done:
            import itertools

            iter_valid_with_prefetch = prefetch_to_device(
                itertools.cycle(data_valid_batches), 2
            )

            valid_aux = []
            valid_loss = []
            for i in range(len(data_valid_batches)):
                batch = next(iter_valid_with_prefetch)
                reporter.tick(f"{current_step} (valid {i})")
                v_loss, v_aux = loss_fn(params, batch)
                valid_aux.append(v_aux)
                valid_loss.append(v_loss)

            train_aux = tree_stack(train_aux)
            train_metrics = train_metrics_fn(train_aux)

            valid_aux = tree_stack(valid_aux)
            valid_metrics = valid_metrics_fn(valid_aux)

            train_loss_avg = np.mean(train_loss)
            valid_loss_avg = np.mean(valid_loss)

            # Update state
            state["step"] = current_step
            state["opt_state"] = opt_state
            state["iter_train"] = iter_train.get_state()

            # Log
            elapsed = monotonic() - start_time
            time_per_step = (
                elapsed / (current_step - start_step) if current_step > start_step else 0
            )
            eta = (max_steps - current_step) * time_per_step

            info = {"lr": get_lr(opt_state), "time_per_step": time_per_step}

            for logger in loggers:
                logger(
                    state["step"],
                    train_loss_avg,
                    train_metrics,
                    valid_loss_avg,
                    valid_metrics,
                    other=info,
                )

            # Checkpoint
            metrics_all = {"train": train_metrics, "valid": valid_metrics}
            metrics_all = jax.tree_util.tree_map(lambda x: np.array(x), metrics_all)

            save_checkpoints(
                metrics_all,
                params,
                state,
                model,
                baseline,
                workdir,
                config=config,
            )

            # Report
            msg = []
            msg.append(f"step: {state['step']}")
            msg.append(f"train loss: {train_loss_avg:.5e}")
            msg.append(f"valid loss: {valid_loss_avg:.5e}")
            msg.append(f"LR: {get_lr(opt_state):.3e}")
            msg.append("validation errors:")
            msg += format_metrics(valid_metrics, keys=keys)
            msg.append(f"elapsed: {s2s(elapsed, 's')}, ETA: {s2s(eta, 'm')}")
            comms.state(msg, title=f"Validation at step {state['step']}")

            # Reset accumulators
            train_aux = []
            train_loss = []
            ran_steps = 0

    # -- wrap up --
    reporter.step("wrapup")
    reporter.done()

    if use_wandb:
        run.finish()

    comms.state("done!")
