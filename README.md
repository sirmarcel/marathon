# `marathon`: modular training infrastructure in `jax`

*pheidippides would be a great name for a message-passing neural network*

`marathon` is an experimental `jax/flax`-oriented framework for prototyping machine-learning interatomic potentials. It does not provide a finished and polished training loop; instead it provides a few composable parts that can be assembled and adapted as needed for experiments. It's therefore not intended as user-facing production code, instead it aims to make experiments faster and more pleasant.

`marathon` provides the following functionality:

- `marathon.data`: Processing `ase.Atoms` objects first into graphs and then into suitably padded batches of graphs
- `marathon.emit`: Checkpointing and logging (text, W&B), diagnostic plots
- `marathon.evaluate`: Predicting energy, forces, and stress, computing the loss as well as metrics (MAE, RMSE, R2)
- `marathon.elemental`: Computing per-element contributions with linear regression (needed to avoid floating point difficulties)
- `marathon.io`: Reading and writing of `msgpack` and `yaml`, as well as a very minimal way to turn `dataclass` instances into `dicts` and vice versa (to instantiate and store `flax.nn.Module`s)

In addition, `marathon.experimental` contains more advanced tooling:

- `marathon.experimental.hermes` provides tools to build `marathon` training pipelines with [`grain`](https://github.com/google/grain) designed to scale to large-ish datasets (up to millions of samples) (**currently in development**)
- `marathon.experimental.ensemble` contains variants of the functionality in `marathon` for ensemble-based uncertainty quantification (**currently abandoned**)

Finally, `examples/` contains almost-finished scripts that implement a full training run that can be easily adapted to particular usecases.

Since the library is aimed at active research and is used and adapted as needed, there is no documentation beyond `README.md` files at each module level explaining terminology, notation, and sometimes the idea behind the design of a subpackage.

You are encouraged to fork and adapt `marathon` for your personal experiments. Very useful functionality can be upstreamed, but this is an essentially personal project, so in many cases it may be more efficient to just maintain a fork.

## Installation and dependencies

You'll need `jax`, probably via `pip install "jax[cuda12]"`.

Then, you should be able to run `pip install -e .`, which will install all other dependencies.

`marathon` provides a number of extras, all of which are installable via `pip install -e .[all]`. They are required to run some parts of the code. They are not automatically installed to avoid dependency resolution hell. Check the `pyproject.toml` for a list.

For convenience, `marathon` looks for an environment variable named `DATASETS` and turns it, if it exists, into a `Path` at `marathon.data.datasets`.

## Development

Linting and formatting is done by `ruff` (`ruff format . && ruff check --fix .`). We expect a line length of 92, but it is not enforced by the linter, only by the formatter. This is to avoid hassle if lines cannot be shortened automatically.
