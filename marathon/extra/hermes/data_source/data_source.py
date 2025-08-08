import numpy as np

from pathlib import Path

from mmap_ninja import RaggedMmap

from marathon.io import read_yaml

from .flatten_atoms import unflatten_atoms


class DataSource:
    def __init__(self, folder, remove_baseline=True, species_to_weight=None):
        self.folder = Path(folder)
        self._mmap = RaggedMmap(self.folder / "mmap")
        self.species_to_weight = None

        if (self.folder / "info.yaml").is_file():
            self.info = read_yaml(self.folder / "info.yaml")
        else:
            self.info = None

        if remove_baseline:
            if not species_to_weight:
                self.species_to_weight = read_yaml(self.folder / "baseline.yaml")
            else:
                self.species_to_weight = species_to_weight

    def __repr__(self):
        # used by DataLoader to verify checkpoint
        return f"{self.__module__}.{self.__class__.__qualname__} reading {self.folder.resolve()}"

    def __len__(self):
        return len(self._mmap)

    def __getitem__(self, record_key):
        flattened = self._mmap[record_key]
        atoms = unflatten_atoms(flattened)
        if self.species_to_weight:
            offset = sum([self.species_to_weight[Z] for Z in atoms.get_atomic_numbers()])
            atoms.calc.results["energy"] -= offset

        if np.isnan(atoms.calc.results["energy"]):
            del atoms.calc.results["energy"]

        if np.isnan(atoms.calc.results["forces"]).any():
            del atoms.calc.results["forces"]

        if np.isnan(atoms.calc.results["stress"]).any():
            del atoms.calc.results["stress"]

        if self.info is not None:
            atoms.info = self.info[record_key]

        return atoms

    # we can't pickle open files, so we make sure that the mmap gets re-opened
    def __setstate__(self, state):
        self.folder = Path(state["folder"])
        self._mmap = RaggedMmap(self.folder / "mmap")
        self.species_to_weight = state["species_to_weight"]
        self.info = state["info"]

    def __getstate__(self):
        return {
            "folder": self.folder,
            "species_to_weight": self.species_to_weight,
            "info": self.info,
        }
