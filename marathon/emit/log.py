from pathlib import Path

from marathon.emit.properties import (
    DEFAULT_PROPERTIES,
    get_full_unit,
    get_scale,
    get_symbol,
)
from marathon.evaluate.properties import DEFAULT_NORMALIZATION


class WandB:
    def __init__(
        self,
        run,
        keys=["energy", "forces"],
        metrics=None,
        properties=DEFAULT_PROPERTIES,
        normalization=DEFAULT_NORMALIZATION,
    ):
        if metrics is None:
            metrics = {key: ["r2", "mae", "rmse"] for key in keys}
        if keys is None:
            assert metrics is not None

        self.metrics = metrics
        self.properties = properties
        self.normalization = normalization

        self.run = run

        for key, ms in self.metrics.items():
            for metric in ms:
                if "r2" == metric:
                    summary = "max"
                    unit = "%"
                else:
                    summary = "min"
                    unit = get_full_unit(key, properties, normalization)

                self.run.define_metric(f"train/{key} {metric} ({unit})", summary=summary)
                self.run.define_metric(f"val/{key} {metric} ({unit})", summary=summary)

    def __call__(self, step, train_loss, train_metrics, val_loss, val_metrics, other=None):
        import numpy as np

        data = {}
        data["train/loss"] = train_loss
        data["val/loss"] = val_loss

        if np.isnan(train_loss):
            # we don't log NaNs
            return

        for key, ms in self.metrics.items():
            scale = get_scale(key, self.properties)
            unit = get_full_unit(key, self.properties, self.normalization)

            for metric in ms:
                if metric == "r2":
                    unit_str = "%"
                    value_scale = 1
                else:
                    unit_str = unit
                    value_scale = scale

                if key in train_metrics and metric in train_metrics[key]:
                    data[f"train/{key} {metric} ({unit_str})"] = (
                        train_metrics[key][metric] * value_scale
                    )
                if key in val_metrics and metric in val_metrics[key]:
                    data[f"val/{key} {metric} ({unit_str})"] = (
                        val_metrics[key][metric] * value_scale
                    )

        if other is not None:
            for k, v in other.items():
                data[k] = v

        self.run.log(step=step, data=data, commit=True)


class Txt:
    def __init__(
        self,
        keys=["energy", "forces"],
        metrics=None,
        workdir=Path("run/"),
        properties=DEFAULT_PROPERTIES,
        normalization=DEFAULT_NORMALIZATION,
    ):
        if metrics is None:
            metrics = {key: ["r2", "mae", "rmse"] for key in keys}
        if keys is None:
            assert metrics is not None

        self.metrics = metrics
        self.folder = workdir / "logs"
        self.properties = properties
        self.normalization = normalization

        self.is_set_up = False

    def setup(self):
        metric_min_widths = []
        metric_desc = []
        metric_formatters = []
        for key, ms in self.metrics.items():
            symbol = get_symbol(key, self.properties)
            for m in ms:
                metric_min_widths.append(get_width(m))
                metric_desc.append(f"{symbol} {m.upper()}")
                metric_formatters.append(get_formatter(m))

        min_widths = [
            8,  # up to 10000000
            9,  # 1.23e-10
            *metric_min_widths,
        ]

        titles = [
            "Step",
            "Loss",
            *metric_desc,
        ]

        self.formatters = [
            lambda x: f"{int(x)}",
            lambda x: f"{x:.2e}",
            *metric_formatters,
        ]

        self.widths = [max(mw, len(title)) for mw, title in zip(min_widths, titles)]

        # Build dynamic units header
        units_parts = []
        for key in self.metrics.keys():
            unit = get_full_unit(key, self.properties, self.normalization)
            symbol = get_symbol(key, self.properties)
            units_parts.append(f"{unit} ({symbol})")
        units_str = ", ".join(units_parts) + ", % (R2)"

        # if the folder exists, we simply append and don't write a title
        if not self.folder.is_dir():
            self.folder.mkdir()

            with open(self.folder / "train.txt", "w") as f:
                f.write(f"Training set losses. Units: {units_str}\n")
                f.write(self.row_to_str(titles))

            with open(self.folder / "valid.txt", "w") as f:
                f.write(f"Validation set losses. Units: {units_str}\n")
                f.write(self.row_to_str(titles))

    def row_to_str(self, entries):
        return (
            " | ".join([f"{s:>{width}}" for s, width in zip(entries, self.widths)]) + "\n"
        )

    def __call__(self, step, train_loss, train_metrics, val_loss, val_metrics, other=None):
        if not self.is_set_up:
            self.setup()
            self.is_set_up = True

        row = []

        row.append(step)
        row.append(train_loss)

        for key, metrics in self.metrics.items():
            scale = get_scale(key, self.properties)
            for metric in metrics:
                value = train_metrics[key].get(metric)
                if value is not None and metric != "r2":
                    value = value * scale
                row.append(value)

        formatted = [f(x) for x, f in zip(row, self.formatters)]

        with open(self.folder / "train.txt", "a") as f:
            f.write(self.row_to_str(formatted))

        row = []
        row.append(step)
        row.append(val_loss)

        for key, metrics in self.metrics.items():
            scale = get_scale(key, self.properties)
            for metric in metrics:
                value = val_metrics[key].get(metric)
                if value is not None and metric != "r2":
                    value = value * scale
                row.append(value)

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

        def formatter(x):
            if x is None:
                return "--"
            else:
                return f"{x:.3f}"

    else:

        def formatter(x):
            if x is None:
                return "--"
            else:
                return f"{x:.2e}"

    return formatter
