# marathon

JAX-based infrastructure for training MLIPs (machine learning interatomic potentials). Published on PyPI as `marathon-train`, importable as `marathon`.

## Running tests

All three layers must pass before committing.

**Lint and format** (must be clean):
```
ruff check marathon/ && ruff format --check marathon/
```
Auto-fix with `ruff check --fix marathon/ && ruff format marathon/`.

**Inline tests** run on import. Every file with a `# -- test --` marker has tests at the bottom that execute when the module is imported:
```
python -c "import marathon.data.properties"
python -c "import marathon.data.sample"
python -c "import marathon.data.batching"
python -c "import marathon.utils"
python -c "import marathon.emit.pretty"
python -c "import marathon.emit.properties"
python -c "import marathon.extra.edge_to_edge.neighborlist"
python -c "import marathon.grain.data_source.flatten_atoms"
python -c "import marathon.grain.data_source.properties"
```

**Pytest** (from the repo root):
```
cd tests && python -m pytest -x .
```

**Examples** in `examples/` each have a `run.sh` that runs end-to-end:
```
cd examples/inference && bash run.sh
cd examples/calculator && bash run.sh
cd examples/train_plain && bash run.sh   # slow on CPU
cd examples/train_grain && bash run.sh   # slow on CPU, requires grain extras
```

**Integration tests** live in `../playground/` (outside this repo):
```
python ../playground/test_imports.py
python ../playground/test_grain_compat.py
```

## Package structure

```
marathon/
  data/         # ase.Atoms -> Sample -> Batch (pure numpy)
  grain/        # scalable data pipelines via grain (optional dep)
  evaluate/     # predict, loss, metrics (JAX)
  emit/         # checkpointing, logging, plotting
  io/           # serialization (yaml, msgpack, spec dicts)
  extra/
    edge_to_edge/  # PET-style rectangular neighborlists (requires numba)
    hermes/        # deprecated re-exports from grain (backward compat)
```

Each subpackage has its own `README.md` with module-specific conventions. The technical README at `marathon/README.md` covers the shared data model, label conventions, properties system, serialization, and code style.

## Code style

- `ruff` with line length 92. Suppressed rules: `E741`, `E731`, `F722`, `E402`, `E501`.
- Import order: numpy/jax before other third-party (configured via isort sections in `pyproject.toml`).
- Lazy imports for optional deps: `grain`, `matplotlib`, `wandb`, `numba`.
- Inline tests use exactly the `# -- test --` marker. No other variations.
- Inline `# -- test --` blocks must **not dispatch any JAX op** (no `jnp.*` calls, no `jax.devices()`, no calls into helpers that do those things internally). Inline tests run on every import of the module — including in parallel grain data-loader workers — and the first JAX device op preallocates ~75% of GPU memory by default, which OOMs immediately under parallelism. Importing `jax`/`jax.numpy` at module top is fine; backend init is lazy. If a test exercises JAX, it belongs in `tests/`, not inline.
- `__all__` in every `__init__.py` defines the public API.
- Factory functions: `get_*_fn` pattern (returns closures).
- `namedtuple` for data objects, plain dicts for configs, `@frozen` or `@dataclass` for transforms.
- Private functions use `_` prefix. No `__` mangling.
- Docstrings only where behavior isn't obvious from the signature. Comments explain *why*, not *what*.

## Key conventions

- **Energy** is total energy (eV), not per-atom. **Stress** is dU/de (eV), not pressure.
- **Padding** adds one extra structure at the end of each batch. Downstream code must handle zero displacements.
- The properties system (`PROPERTIES` + `NORMALIZATION` dicts) threads through data loading, loss, metrics, and logging. See `marathon/README.md` for details.

## Releasing

Use the `/release` skill (`.claude/commands/release.md`). It handles pre-flight checks, local verification, changelog, version bump, tagging, and PyPI publish.

- Versioning: **EffVer** (effort-based), not semver. Bump levels are `micro`, `meso`, `macro`.
- Version lives in `pyproject.toml` (not derived from tags).
- CI publishes to PyPI on tag push via `.github/workflows/release.yml` (trusted publishing).

## Optional dependencies

Core: `opsis`, `PyYAML`, `vesin`, `flax`, `ase`. Install extras with:
- `pip install marathon-train[grain]` -- grain pipelines (grain, mmap_ninja, numba, scipy)
- `pip install marathon-train[plot]` -- matplotlib plotting
- `pip install marathon-train[wandb]` -- W&B logging
- `pip install marathon-train[all]` -- everything
