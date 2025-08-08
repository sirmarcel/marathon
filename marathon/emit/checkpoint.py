import shutil
from pathlib import Path

from marathon.io import (
    from_dict,
    read_msgpack,
    read_yaml,
    register,
    to_dict,
    write_msgpack,
    write_yaml,
)


def save_checkpoints(
    metrics,
    params,
    state,
    model,
    baseline,
    workdir=Path("run"),
    config=None,
):
    workdir = workdir / "checkpoints"
    workdir.mkdir(exist_ok=True)  # will fail if parent doesn't exist yet

    step = state["step"]

    evaluations = [checkpointer(step, metrics) for checkpointer in state["checkpointers"]]

    for need_saving, name_and_info in evaluations:
        if need_saving:
            name, info = name_and_info
            folder = workdir / name
            save_with_backup(
                folder, params, state, model, baseline, metrics, info=info, config=config
            )


def get_latest(folder, empty_state):
    assert folder.is_dir()

    latest = 0
    items = None
    for f in folder.glob("checkpoints/*/"):
        new_items = restore(f, empty_state)
        state = new_items[1]

        if state["step"] > latest:
            items = new_items
            latest = state["step"]

    return items


def get_all(folder, empty_state):
    assert folder.is_dir()

    for f in folder.glob("checkpoints/*/"):
        items = restore(f, empty_state)
        yield f, items


# -- checkpointers --


class Latest:
    def __init__(self, every):
        self.every = every

        self.state_dict = {}

    def __call__(self, step, metrics):
        if step % self.every == 0:
            info = f"saving checkpoints every {self.every} steps\n"
            info += f"-> step={step}"
            return True, ("latest", info)
        else:
            return False, (None, None)

    def restore(self, dct):
        return self


class SummedMetric:
    def __init__(self, name, metric, keys=["energy", "forces"], split="valid"):
        self.best = float("inf")

        self.name = name
        self.metric = metric
        self.keys = keys
        self.split = split

        if metric == "r2":
            self.factor = -1.0
        else:
            self.factor = 1.0

    def __call__(self, step, metrics):
        target = 0.0
        for key in self.keys:
            target += self.factor * metrics[self.split][key][self.metric]

        if target < self.best:
            self.best = float(target)
            info = f"model with best summed {self.metric} of {' + '.join(self.keys)}"
            info += f" on split {self.split} at step={step}:\n"
            info += f"-> loss={self.best:.10e}"
            return True, (self.name, info)

        else:
            return False, (None, None)

    @property
    def state_dict(self):
        return {"best": self.best}

    def restore(self, dct):
        self.best = dct["best"]
        return self


register(Latest)
register(SummedMetric)


# -- i/o --


def save_with_backup(folder, params, state, model, baseline, metrics, info="", config=None):
    if folder.is_dir():
        backup = folder.with_suffix(".backup")
        if backup.is_dir():
            shutil.rmtree(backup)
        folder.rename(backup)

    save(folder, params, state, model, baseline, metrics, info=info, config=config)


def save(folder, params, state, model, baseline, metrics, info="", config=None):
    # todo: consider saving "safely" by writing to TMP first

    folder.mkdir()  # fail here if exists

    model_dir = folder / "model"
    model_dir.mkdir()

    write_yaml(model_dir / "model.yaml", to_dict(model))
    write_msgpack(model_dir / "model.msgpack", params)
    write_yaml(model_dir / "baseline.yaml", baseline)

    write_msgpack(folder / "state.msgpack", state)

    write_yaml(folder / "metrics.yaml", metrics)

    if info != "":
        with open(folder / "info.txt", "w") as f:
            f.write(info)

    if config is not None:
        write_yaml(folder / "config.yaml", config)


def restore(folder, empty_state):
    model_dict = read_yaml(folder / "model/model.yaml")
    model = from_dict(model_dict)

    params = read_msgpack(folder / "model/model.msgpack")

    baseline = read_yaml(folder / "model/baseline.yaml")

    # todo: why do we need an empty state here?
    state = read_msgpack(folder / "state.msgpack", target=empty_state)

    if (folder / "metrics.yaml").is_file():
        metrics = read_yaml(folder / "metrics.yaml")
    else:
        metrics = None

    if (folder / "config.yaml").is_file():
        config = read_yaml(folder / "config.yaml")
    else:
        config = None

    return params, state, model, baseline, metrics, config
