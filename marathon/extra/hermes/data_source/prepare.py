import numpy as np

from pathlib import Path

from mmap_ninja import RaggedMmap

from marathon import comms
from marathon.io import write_yaml

from .flatten_atoms import flatten_atoms
from .properties import DEFAULT_PROPERTIES


def prepare(
    dataset,
    folder="storage",
    batch_size=100,
    samples_per_composition=25,
    reporter=None,
    properties=DEFAULT_PROPERTIES,
):
    # consume iterable dataset (of Atoms)
    # users are expected to write their own iterators to correct irregularities

    folder = Path(folder)
    if folder.exists():
        comms.warn(f"{folder} exists, exiting")
        return None

    mmap = folder / "mmap"
    mmap.mkdir(parents=True)

    offsetter = OffsetHelper(samples_per_composition=samples_per_composition)

    if reporter:
        reporter.step("processing", spin=False)

    def iterate():
        for i, atoms in enumerate(dataset):
            if reporter:
                reporter.tick(f"{i}")
            offsetter(atoms)
            yield flatten_atoms(atoms, properties=properties)

    RaggedMmap.from_generator(
        out_dir=mmap,
        sample_generator=iterate(),
        batch_size=batch_size,
        verbose=False,
    )

    if reporter:
        reporter.finish_step()

    species_to_weight = offsetter.get_species_weights()
    msg = []
    for s, w in species_to_weight.items():
        msg.append(f"{s}: {w:.3f}")
    comms.state(msg, title="per-atom contributions (by species)")

    write_yaml(folder / "baseline.yaml", species_to_weight)
    write_yaml(folder / "properties.yaml", properties)


class OffsetHelper:
    def __init__(self, samples_per_composition=5):
        self.compositions = {}
        self.samples_per_composition = samples_per_composition

    def __call__(self, atoms):
        composition = tuple(atoms.get_atomic_numbers().tolist())
        if composition not in self.compositions:
            self.compositions[composition] = [atoms.get_potential_energy()]
        else:
            if len(self.compositions[composition]) < self.samples_per_composition:
                self.compositions[composition].append(atoms.get_potential_energy())

    def get_species_weights(self):
        from marathon.elemental import compute_weights

        compositions, energy = [], []
        for C, Es in self.compositions.items():
            for E in Es:
                compositions.append(C)
                energy.append(E)

        return compute_weights(compositions, np.array(energy))
