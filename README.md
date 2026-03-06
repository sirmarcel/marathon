<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo.png">
    <img src="assets/logo-dark.png" alt="marathon" width="300">
  </picture>
</p>

<h1 align="center">marathon</h1>

<p align="center">
  <em>modular training infrastructure for machine-learning interatomic potentials in JAX</em>
</p>

<p align="center">
  <img alt="status: experimental" src="https://img.shields.io/badge/status-experimental-orange">
  <img alt="code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
</p>

<p align="center">
  <em>pheidippides would be a great name for a message-passing neural network</em>
</p>

---

`marathon` is an experimental `jax/flax`-oriented toolkit for prototyping machine-learning interatomic potentials. It does not provide a finished and polished training loop; instead it provides a few composable parts that can be assembled and adapted as needed for experiments. It's therefore not intended as user-facing production code, instead it aims to make experiments faster and more pleasant.

| Module | Description |
|--------|-------------|
| `marathon.data` | Process `ase.Atoms` into padded batches with a flexible properties system |
| `marathon.evaluate` | Predict energy, forces, stress; compute loss (MSE/Huber) and metrics (MAE, RMSE, R2) |
| `marathon.emit` | Checkpointing, logging (text, W&B), diagnostic plots |
| `marathon.io` | Read/write `msgpack` and `yaml`; serialize `flax.nn.Module` instances |
| `marathon.elemental` | Per-element energy baselines via linear regression |
| `marathon.grain` | Scalable data pipelines with [grain](https://github.com/google/grain) for large datasets |
| `marathon.extra.edge_to_edge` | Fixed-size neighborhood batching for PET-style edge transformers |

Since the library is aimed at active research and is used and adapted as needed, there is no documentation beyond `README.md` files at each module level explaining terminology, notation, and sometimes the idea behind the design of a subpackage. This avoids the risk of documentation and code going out of sync -- at the cost of requiring more code reading. (Luckily, the computers can do some of the reading nowadays...)

Anyhow, you are encouraged to fork and adapt `marathon` for your personal experiments. PRs with self-contained and reusable features are welcome.

## Installation

The main dependency is `jax`; detailed installation instructions are [here](https://github.com/jax-ml/jax/?tab=readme-ov-file#installation). Typically, the standard install works fairly well:

```bash
pip install "jax[cuda13]"   # or jax[cpu] for CPU-only
pip install -e .
```

`marathon` provides a number of extras, installable via `pip install -e ".[all]"`. They are required to run some parts of the code but not automatically installed to avoid dependency resolution hell in HPC systems.

```bash
pip install -e ".[grain]"   # grain pipelines: grain, mmap_ninja, numba
pip install -e ".[dev]"     # development: pytest, ruff
pip install -e ".[wandb]"   # Weights & Biases logging
pip install -e ".[plot]"    # plotting: matplotlib, scipy
```

For convenience, `marathon` looks for an environment variable named `DATASETS` and turns it, if it exists, into a `Path` at `marathon.data.datasets`. It is highly recommended to use it!

## Quick start

For datasets that can be fully fit into (GPU) memory (do this ahead of time to fixed size and shuffle on GPU):

```python
from marathon.data import to_sample, batch_samples, determine_max_sizes

samples = [to_sample(atoms, cutoff=5.0) for atoms in my_atoms]
num_atoms, num_pairs = determine_max_sizes(samples, batch_size=4)
batch = batch_samples(samples, num_atoms, num_pairs, keys=["energy", "forces"])
```

For large-scale training with [`grain`](https://github.com/google/grain) pipelines (streaming data from disk through a series of transforms):

```python
from marathon.grain import DataSource, DataLoader, IndexSampler, ToSample, ToFixedLengthBatch

ds = DataSource("path/to/prepared/dataset")
sampler = IndexSampler(len(ds), shuffle=True, seed=0)
loader = DataLoader(
    data_source=ds,
    sampler=sampler,
    operations=[ToSample(cutoff=5.0), ToFixedLengthBatch(batch_size=4)],
)
```

## Development

```bash
pip install -e ".[dev]"
ruff format . && ruff check --fix .
python -m pytest
```

Linting and formatting is done by `ruff`. We use a line length of 92, but it is not enforced by the linter, only by the formatter. This avoids hassle when lines can't be shortened automatically. We also suppress some rules that get in the way of research code: short variable names (`E741`), lambdas (`E731`), and non-top-level imports (`E402`). Import ordering groups `numpy` and `jax` before other third-party packages.

The code itself tends towards concise and functional: descriptive names, minimal docstrings (only where behaviour isn't obvious from context), and liberal use of lambdas and comprehensions. Many modules include inline tests at the bottom that run on import.
