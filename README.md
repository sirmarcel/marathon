# `marathon`: Μαραθώνιος

*pheidippides would be a great name for a message-passing neural network*

This is one half of my personal `jax` tooling for prototyping machine-learning interatomic potentials. This half deals with training models, the other half, [`myrto`](https://github.com/sirmarcel/myrto-dev) with implementing them.

The main concept is that the library only collects tooling required to write a training loop. You are expected and encouraged to put together your own training scripts for your particular experiments.

You are encouraged to fork and adapt `marathon` for your personal experiments. Please only upstream things that are general and reasonably aesthetically pleasing.

***

The general functionality of `marathon` is:

- `marathon.data`: Logic to turn `ase.Atoms` into `Samples` and then `Batch`es (i.e., padding everything to fixed size)
- `marathon.emit`: Checkpointing, logging (to text and WandB), and basic plots
- `marathon.evaluate`: Predicting energy, forces, and stress, computing the loss as well as metrics (MAE, RMSE, R2)
- `marathon.ensemble`: Versions of the above functionality for ensemble models
- `marathon.elemental`: Computing per-element contributions with linear regression (needed to avoid floating point difficulties)

The `examples/` folder contains (partial) scripts to do various things with `marathon`. Adapt to your needs! In particular, a training script that is "good enough" for many cases is given in `examples/run.py`. It expects to live in a folder with a `model.yaml` (which can be turned into a model via `myrto.engine`) and a `data.py` that does the job of loading datasets.

## Installation and dependencies

You'll need `jax`, probably via `pip install "jax[cuda12]"`, as well as [`myrto`](https://github.com/sirmarcel/myrto-dev).

Then, you should be able to run `pip install -e .`, which will install all other dependencies.

`marathon` provides a number of extras, all of which are installable via `pip install -e .[all]`. They are required to run some parts of the code. They are not automatically installed to avoid dependency resolution hell. Check the `pyproject.toml` for a list.

For convenience, `marathon` looks for an environment variable named `DATASETS` and turns it, if it exists, into a `Path` at `marathon.data.datasets`.

## Development

Linting and formatting is done by `ruff` (`ruff format . && ruff check --fix .`). We expect a line length of 92, but it is not enforced by the linter, only by the formatter. This is to avoid hassle if lines cannot be shortened automatically.
