import numpy as np

import multiprocessing
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from mmap_ninja import RaggedMmap

from marathon import comms
from marathon.io import write_yaml

from .flatten_atoms import flatten_atoms, unflatten_numbers_and_energy
from .properties import DEFAULT_PROPERTIES


def prepare(
    dataset,
    folder="storage",
    batch_size=100,
    samples_per_composition=25,
    reporter=None,
    properties=DEFAULT_PROPERTIES,
    num_workers=1,
    baseline=True,
):
    """Serialize ase.Atoms to a memory-mapped DataSource folder. No-ops if folder exists.

    With num_workers=1, dataset is any iterable of Atoms. With num_workers > 1, it must
    support len and integer indexing and be cheap to pickle: a list (sliced per worker),
    or a reader that holds paths and opens its files lazily per process, like AseDB.
    With baseline=True, also fits the per-species energy baseline (see fit_baseline).
    """
    folder = Path(folder)
    if folder.exists():
        comms.warn(f"{folder} exists, exiting")
        return None

    mmap = folder / "mmap"
    mmap.mkdir(parents=True)

    if reporter:
        reporter.step("processing", spin=False)

    if num_workers == 1:

        def iterate():
            for i, atoms in enumerate(dataset):
                if reporter:
                    reporter.tick(f"{i}")
                yield flatten_atoms(atoms, properties=properties)

        RaggedMmap.from_generator(
            out_dir=mmap,
            sample_generator=iterate(),
            batch_size=batch_size,
            verbose=False,
        )
    else:
        context = multiprocessing.get_context(
            "fork" if sys.platform == "linux" else "spawn"
        )
        _prepare_parallel(
            dataset, folder, batch_size, properties, num_workers, context, reporter
        )

    if reporter:
        reporter.finish_step()

    write_yaml(folder / "properties.yaml", properties)

    if baseline:
        fit_baseline(folder, samples_per_composition=samples_per_composition)


def fit_baseline(folder, samples_per_composition=25):
    """Fit per-species energy contributions for a prepared folder, write baseline.yaml.

    Uses the first samples_per_composition records of each ordered composition.
    """
    from .data_source import DataSource

    folder = Path(folder)
    source = DataSource(folder, remove_baseline=False)
    offsetter = OffsetHelper(samples_per_composition=samples_per_composition)
    for i in range(len(source)):
        numbers, energy = unflatten_numbers_and_energy(source._mmap[i], source.properties)
        offsetter.add(tuple(numbers.tolist()), energy)

    species_to_weight = offsetter.get_species_weights()
    msg = []
    for s, w in species_to_weight.items():
        msg.append(f"{s}: {w:.3f}")
    comms.state(msg, title="per-atom contributions (by species)")

    write_yaml(folder / "baseline.yaml", species_to_weight)
    return species_to_weight


class OffsetHelper:
    def __init__(self, samples_per_composition=5):
        self.compositions = {}
        self.samples_per_composition = samples_per_composition

    def __call__(self, atoms):
        self.add(tuple(atoms.get_atomic_numbers().tolist()), atoms.get_potential_energy())

    def add(self, composition, energy):
        if composition not in self.compositions:
            self.compositions[composition] = [energy]
        else:
            if len(self.compositions[composition]) < self.samples_per_composition:
                self.compositions[composition].append(energy)

    def get_species_weights(self):
        from marathon.elemental import compute_weights

        compositions, energy = [], []
        for C, Es in self.compositions.items():
            for E in Es:
                compositions.append(C)
                energy.append(E)

        return compute_weights(compositions, np.array(energy))


# each extend re-opens the output memmaps, so small batches make the merge superlinear
_MERGE_BATCH_SIZE = 10_000


def _prepare_parallel(
    dataset, folder, batch_size, properties, num_workers, context, reporter=None
):
    n = len(dataset)
    bounds = np.linspace(0, n, min(num_workers, n) + 1).astype(int)
    shards = [folder / "shards" / f"{k}" for k in range(len(bounds) - 1)]
    shards[0].parent.mkdir()

    if hasattr(dataset, "close"):
        # forked workers must not inherit open handles (lmdb refuses to reopen them)
        dataset.close()

    def chunk(start, stop):
        # workers get only their slice of an in-memory list; readers pickle small
        if isinstance(dataset, (list, tuple)):
            return dataset[start:stop], 0, stop - start
        return dataset, start, stop

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=context) as pool:
        futures = [
            pool.submit(_write_shard, *chunk(start, stop), shard, batch_size, properties)
            for start, stop, shard in zip(bounds[:-1], bounds[1:], shards)
        ]
        for i, future in enumerate(as_completed(futures)):
            future.result()
            if reporter:
                reporter.tick(f"{i + 1}/{len(futures)} shards")

    def iterate():
        for shard in shards:
            records = RaggedMmap(shard)
            for i in range(len(records)):
                yield records[i]

    RaggedMmap.from_generator(
        out_dir=folder / "mmap",
        sample_generator=iterate(),
        batch_size=max(batch_size, _MERGE_BATCH_SIZE),
        verbose=False,
    )
    shutil.rmtree(folder / "shards")


def _write_shard(dataset, start, stop, out_dir, batch_size, properties):
    RaggedMmap.from_generator(
        out_dir=out_dir,
        sample_generator=(
            flatten_atoms(dataset[i], properties=properties)
            for i in range(int(start), int(stop))
        ),
        batch_size=batch_size,
        verbose=False,
    )
