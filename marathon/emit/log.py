from pathlib import Path


class WandB:
    def __init__(self, run, keys=["energy", "forces"], metrics=None):
        if metrics is None:
            metrics = {key: ["r2", "mae", "rmse"] for key in keys}
        if keys is None:
            assert metrics is not None

        self.metrics = metrics

        self.run = run

        for key, metrics in self.metrics.items():
            for metric in metrics:
                if "r2" == metric:
                    summary = "max"
                else:
                    summary = "min"

            self.run.define_metric(f"train/{key} {metric}", summary=summary)
            self.run.define_metric(f"val/{key} {metric}", summary=summary)

    def __call__(self, epoch, train_loss, train_metrics, val_loss, val_metrics, other=None):
        import numpy as np

        data = {}
        data["train/loss"] = train_loss
        data["val/loss"] = val_loss

        if np.isnan(train_loss):
            # we don't log NaNs
            return

        for key, metrics in self.metrics.items():
            for metric in metrics:
                data[f"train/{key} {metric}"] = train_metrics[key][metric]
                data[f"val/{key} {metric}"] = val_metrics[key][metric]

        if other is not None:
            if "lr" in other:
                data["learning_rate"] = other["lr"]
            if "time_per_epoch" in other:
                data["time_per_epoch"] = other["time_per_epoch"]

        self.run.log(step=epoch, data=data, commit=True)


class Txt:
    def __init__(self, keys=["energy", "forces"], metrics=None, workdir=Path("run/")):
        if metrics is None:
            metrics = {key: ["r2", "mae", "rmse"] for key in keys}
        if keys is None:
            assert metrics is not None

        self.metrics = metrics
        self.folder = workdir / "logs"

        self.is_set_up = False

    def setup(self):
        metric_min_widths = []
        metric_desc = []
        metric_formatters = []
        for key, ms in self.metrics.items():
            for m in ms:
                metric_min_widths.append(get_width(m))
                metric_desc.append(f"{get_name(key)} {m.upper()}")
                metric_formatters.append(get_formatter(m))

        min_widths = [
            8,  # up to 10000000
            9,  # 1.23e-10
            *metric_min_widths,
        ]

        titles = [
            "Epoch",
            "Loss",
            *metric_desc,
        ]

        self.formatters = [
            lambda x: f"{int(x)}",
            lambda x: f"{x:.2e}",
            *metric_formatters,
        ]

        self.widths = [max(mw, len(title)) for mw, title in zip(min_widths, titles)]

        # if the folder exists, we simply append and don't write a title
        if not self.folder.is_dir():
            self.folder.mkdir()

            with open(self.folder / "train.txt", "w") as f:
                f.write("Training set losses. Units: meV (U), meV/Å (F), meV (σ), % (R2)\n")
                f.write(self.row_to_str(titles))

            with open(self.folder / "valid.txt", "w") as f:
                f.write("Training set losses. Units: meV (U), meV/Å (F), meV (σ), % (R2)\n")
                f.write(self.row_to_str(titles))

    def row_to_str(self, entries):
        return (
            " | ".join([f"{s:>{width}}" for s, width in zip(entries, self.widths)]) + "\n"
        )

    def __call__(self, epoch, train_loss, train_metrics, val_loss, val_metrics, other=None):
        if not self.is_set_up:
            self.setup()

        row = []

        row.append(epoch)
        row.append(train_loss)

        for key, metrics in self.metrics.items():
            for metric in metrics:
                row.append(train_metrics[key][metric])

        formatted = [f(x) for x, f in zip(row, self.formatters)]

        with open(self.folder / "train.txt", "a") as f:
            f.write(self.row_to_str(formatted))

        row = []
        row.append(epoch)
        row.append(val_loss)

        for key, metrics in self.metrics.items():
            for metric in metrics:
                row.append(train_metrics[key][metric])

        formatted = [f(x) for x, f in zip(row, self.formatters)]

        with open(self.folder / "valid.txt", "a") as f:
            f.write(self.row_to_str(formatted))


def get_width(metric):
    if "r2" in metric:
        return 6  # 99.999
    else:
        return 9


def get_formatter(metric):
    if "r2" in metric:
        return lambda x: f"{x:.3f}"
    else:
        return lambda x: f"{x:.2e}"


def get_name(key):
    names = {
        "energy": "U",
        "forces": "F",
        "stress": "σ",
    }

    return names[key]


def get_unit(key, metric):
    if "r2" in metric:
        return "%"

    if key == "energy":
        return "meV/atom"
    if key == "forces":
        return "meV/Å"
    if key == "stress":
        return "meV"


# -- test --

# import shutil

# logger = Txt()
# logger(
#     100,
#     1e-2,
#     {
#         "energy": {"r2": 99.9, "mae": 1e-3, "rmse": 1e-2},
#         "forces": {"r2": 99.9, "mae": 1e-3, "rmse": 1e-2},
#         "stress": {"r2": 99.9, "mae": 1e-3, "rmse": 1e-2},
#     },
#     1e-3,
#     {
#         "energy": {"r2": 99.9, "mae": 1e-3, "rmse": 1e-2},
#         "forces": {"r2": 99.9, "mae": 1e-3, "rmse": 1e-2},
#         "stress": {"r2": 99.9, "mae": 1e-3, "rmse": 1e-2},
#     },
# )

# logger = Txt()
# logger(
#     100,
#     1e-2,
#     {
#         "energy": {"r2": 91.9, "mae": 1e-3, "rmse": 1e-2},
#         "forces": {"r2": 99.9, "mae": 1e-3, "rmse": 1e-2},
#         "stress": {"r2": 93.9, "mae": 1e-3, "rmse": 1e-2},
#     },
#     1e-3,
#     {
#         "energy": {"r2": 99.9, "mae": 1e-3, "rmse": 1e-2},
#         "forces": {"r2": 19.9, "mae": 1e-3, "rmse": 1e-2},
#         "stress": {"r2": 99.9, "mae": 1e-3, "rmse": 1e-2},
#     },
# )

# shutil.rmtree("logs/")
