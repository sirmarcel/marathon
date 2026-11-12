import numpy as np

import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from mmap_ninja import RaggedMmap
from mmap_ninja import base as mmap_base
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

    With num_workers > 1, dataset must support len and integer indexing and pickle
    cheaply (a list, or a lazy reader like AseDB); workers are spawned, so the calling
    script needs an `if __name__ == "__main__"` guard.

    Work happens in `<folder>.partial`, renamed to folder once complete.
    """
    target = Path(folder)
    if target.exists():
        comms.warn(f"{target} exists, exiting")
        return None

    if baseline and "energy" not in properties:
        raise ValueError("baseline=True needs an 'energy' property; pass baseline=False")

    # a killed run never reaches the cleanup below, so folder only appears when complete
    folder = target.with_name(target.name + ".partial")
    if folder.exists():
        comms.warn(f"removing {folder}, left by an interrupted run")
        shutil.rmtree(folder)

    try:
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

        folder.rename(target)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise


def fit_baseline(folder, samples_per_composition=25):
    """Fit the per-species energy baseline of a prepared folder, write baseline.yaml."""
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
        if isinstance(dataset, (list, tuple)):
            return dataset[start:stop], 0, stop - start
        return dataset, start, stop

    samples = offsetter.samples_per_composition if offsetter is not None else None

    # not fork: workers must not inherit lmdb handles or JAX state
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
        for partial, _, _ in collected:
            offsetter.merge(partial)

    dtypes = {dtype for _, _, dtype in collected if dtype is not None}
    if len(dtypes) != 1:
        raise ValueError(f"expected records of one dtype, got {sorted(dtypes)}")

    lengths = [lengths for _, lengths, _ in collected]
    _merge_shards(shards, lengths, dtypes.pop(), folder / "mmap", num_workers, reporter)
    shutil.rmtree(folder / "shards")


def _write_shard(dataset, start, stop, path, batch_size, properties, samples):
    # the shard is the raw concatenated records; the parent builds the index from lengths
    offsetter = OffsetHelper(samples) if samples is not None else None
    lengths = np.zeros(stop - start, dtype=np.int64)
    dtype = None
    with open(path, "wb") as file:
        for batch in range(start, stop, batch_size):
            records = [
                _flatten(dataset[i], properties, offsetter)
                for i in range(batch, min(batch + batch_size, stop))
            ]
            if dtype is None:
                dtype = records[0].dtype
            lengths[batch - start : batch - start + len(records)] = [
                len(r) for r in records
            ]
            np.concatenate(records).astype(dtype, copy=False).tofile(file)

    return offsetter, lengths, None if dtype is None else dtype.name


def _merge_shards(shards, lengths, dtype, out_dir, num_threads, reporter=None):
    # writes what RaggedMmap.from_generator would have written for the same records
    dtype = np.dtype(dtype)
    sizes = np.array([l.sum() for l in lengths], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    lengths = np.concatenate(lengths)
    ends = np.cumsum(lengths)

    mmap_numpy.from_ndarray(out_dir / "starts", ends - lengths)
    mmap_numpy.from_ndarray(out_dir / "ends", ends)
    mmap_numpy.from_ndarray(out_dir / "shapes", lengths[:, None])
    mmap_numpy.from_ndarray(out_dir / "flattened_shapes", lengths)
    mmap_base._int_to_file(1, out_dir / "shapes_are_flat.ninja")
    mmap_numpy._save_mmap_kwargs(out_dir, dtype, (int(offsets[-1]),), "C")
    mmap_base._str_to_file("ragged", out_dir / "type.ninja")

    # sizes are known, so shards go to their offsets in any order
    fd = os.open(out_dir / "data.ninja", os.O_WRONLY | os.O_CREAT, 0o666)
    try:
        os.truncate(fd, int(offsets[-1]) * dtype.itemsize)
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [
                pool.submit(_copy_shard, shard, fd, int(offset) * dtype.itemsize)
                for shard, offset in zip(shards, offsets[:-1])
            ]
            for i, future in enumerate(as_completed(futures)):
                future.result()
                if reporter:
                    reporter.tick(f"merged {i + 1}/{len(shards)} shards")
        os.fsync(fd)
    finally:
        os.close(fd)


def _copy_shard(shard, fd, offset, chunk_size=2**26):
    # plain reads: paging a shard in through np.memmap is slow on network filesystems
    with open(shard, "rb", buffering=0) as file:
        while chunk := file.read(chunk_size):
            view = memoryview(chunk)
            while view:
                written = os.pwrite(fd, view, offset)
                view = view[written:]
                offset += written
    shard.unlink()
