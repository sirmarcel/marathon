import numpy as np

import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from mmap_ninja import RaggedMmap
from mmap_ninja import numpy as mmap_numpy

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
    shard_size=10_000,
):
    """Serialize ase.Atoms to a memory-mapped DataSource folder. No-ops if folder exists.

    With num_workers=1, dataset is any iterable of Atoms. With num_workers > 1, it must
    support len and integer indexing and be cheap to pickle: a list (sliced per shard),
    or a reader that holds paths and opens its files lazily per process, like AseDB.
    Shards of shard_size records go to a pool of spawned workers, so the calling
    script needs an `if __name__ == "__main__"` guard.

    With baseline=True, fits the per-species energy baseline on the first
    samples_per_composition records of each ordered composition, skipping records
    without energy, and writes baseline.yaml. With baseline=False, writes none: open
    the folder with DataSource(folder, remove_baseline=False), or give the model a
    baseline from elsewhere.
    """
    folder = Path(folder)
    if folder.exists():
        comms.warn(f"{folder} exists, exiting")
        return None

    if baseline and "energy" not in properties:
        raise ValueError("baseline=True needs an 'energy' property; pass baseline=False")

    mmap = folder / "mmap"
    mmap.mkdir(parents=True)

    if reporter:
        reporter.step("processing", spin=False)

    offsetter = OffsetHelper(samples_per_composition) if baseline else None

    if num_workers == 1:

        def iterate():
            for i, atoms in enumerate(dataset):
                if reporter:
                    reporter.tick(f"{i}")
                yield _flatten(atoms, properties, offsetter)

        RaggedMmap.from_generator(
            out_dir=mmap,
            sample_generator=iterate(),
            batch_size=batch_size,
            verbose=False,
        )
    else:
        _prepare_parallel(
            dataset,
            folder,
            batch_size,
            properties,
            num_workers,
            shard_size,
            offsetter,
            reporter,
        )

    if reporter:
        reporter.finish_step()

    write_yaml(folder / "properties.yaml", properties)

    if baseline:
        _write_baseline(folder, offsetter)


def fit_baseline(folder, samples_per_composition=25):
    """Fit per-species energy contributions for a prepared folder, write baseline.yaml.

    Uses the first samples_per_composition records of each ordered composition,
    skipping records without energy.
    """
    from .data_source import DataSource

    folder = Path(folder)
    source = DataSource(folder, remove_baseline=False)
    offsetter = OffsetHelper(samples_per_composition)
    for i in range(len(source)):
        numbers, energy = unflatten_numbers_and_energy(source._mmap[i], source.properties)
        offsetter.add(tuple(numbers.tolist()), energy)

    return _write_baseline(folder, offsetter)


def _flatten(atoms, properties, offsetter):
    flattened = flatten_atoms(atoms, properties=properties)
    if offsetter is not None:
        numbers, energy = unflatten_numbers_and_energy(flattened, properties)
        offsetter.add(tuple(numbers.tolist()), energy)
    return flattened


def _write_baseline(folder, offsetter):
    if offsetter.skipped:
        comms.warn(f"baseline: skipped {offsetter.skipped} records without energy")
    if not offsetter.compositions:
        raise ValueError("baseline: no records with energy; pass baseline=False")

    species_to_weight = offsetter.get_species_weights()
    msg = []
    for s, w in species_to_weight.items():
        msg.append(f"{s}: {w:.3f}")
    comms.state(msg, title="per-atom contributions (by species)")

    write_yaml(folder / "baseline.yaml", species_to_weight)
    return species_to_weight


class OffsetHelper:
    """Collects the first samples_per_composition energies of each ordered composition."""

    def __init__(self, samples_per_composition=5):
        self.compositions = {}
        self.samples_per_composition = samples_per_composition
        self.skipped = 0

    def __call__(self, atoms):
        self.add(tuple(atoms.get_atomic_numbers().tolist()), atoms.get_potential_energy())

    def add(self, composition, energy):
        energy = float(energy)
        if np.isnan(energy):
            self.skipped += 1
            return

        energies = self.compositions.setdefault(composition, [])
        if len(energies) < self.samples_per_composition:
            energies.append(energy)

    def merge(self, other):
        # other collected the records that follow ours, so appending keeps "first N"
        for composition, energies in other.compositions.items():
            for energy in energies:
                self.add(composition, energy)
        self.skipped += other.skipped

    def get_species_weights(self):
        from marathon.elemental import compute_weights

        compositions, energy = [], []
        for C, Es in self.compositions.items():
            for E in Es:
                compositions.append(C)
                energy.append(E)

        return compute_weights(compositions, np.array(energy))


def _prepare_parallel(
    dataset, folder, batch_size, properties, num_workers, shard_size, offsetter, reporter
):
    n = len(dataset)
    bounds = list(range(0, n, shard_size)) + [n] if n else [0, 0]
    shards = [folder / "shards" / f"{k}" for k in range(len(bounds) - 1)]
    (folder / "shards").mkdir()

    def chunk(start, stop):
        # workers get only their slice of an in-memory list; readers pickle small
        if isinstance(dataset, (list, tuple)):
            return dataset[start:stop], 0, stop - start
        return dataset, start, stop

    samples = offsetter.samples_per_composition if offsetter is not None else None

    # spawn on every platform: fresh interpreters inherit no handles, locks or threads
    # from the parent (lmdb environments, JAX backends); marathon.grain imports in 0.4 s
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=context) as pool:
        futures = {}
        for k, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
            future = pool.submit(
                _write_shard,
                *chunk(start, stop),
                shards[k],
                batch_size,
                properties,
                samples,
            )
            futures[future] = k

        collected = [None] * len(shards)
        for i, future in enumerate(as_completed(futures)):
            collected[futures[future]] = future.result()
            if reporter:
                reporter.tick(f"{i + 1}/{len(shards)} shards")

    if offsetter is not None:
        for partial in collected:
            offsetter.merge(partial)

    _merge_shards(shards, folder / "mmap")
    shutil.rmtree(folder / "shards")


def _write_shard(dataset, start, stop, out_dir, batch_size, properties, samples):
    offsetter = OffsetHelper(samples) if samples is not None else None
    RaggedMmap.from_generator(
        out_dir=out_dir,
        sample_generator=(
            _flatten(dataset[i], properties, offsetter) for i in range(start, stop)
        ),
        batch_size=batch_size,
        verbose=False,
    )
    return offsetter


def _merge_shards(shards, out_dir):
    # array-level concatenation through mmap_ninja's own extend, no per-record loop:
    # starts and ends index into the data buffer, so later shards shift by the running end
    out_dir.rmdir()
    shutil.move(shards[0], out_dir)
    if len(shards) == 1:
        return

    out = RaggedMmap(out_dir)
    end = int(out.ends[-1])
    for shard in shards[1:]:
        records = RaggedMmap(shard)
        mmap_numpy.extend(out.memmap, records.memmap)
        mmap_numpy.extend(out.starts, records.starts + end)
        mmap_numpy.extend(out.ends, records.ends + end)
        mmap_numpy.extend(out.shapes, records.shapes)
        mmap_numpy.extend(out.flattened_shapes, records.flattened_shapes)
        end += int(records.ends[-1])
