from pathlib import Path

from mmap_ninja import RaggedMmap

from marathon.io import read_yaml

from .flatten_atoms import unflatten_atoms
from .properties import normalize_properties


class DataSource:
    """Memory-mapped dataset of ase.Atoms, with optional per-species energy baseline removal."""

    def __init__(self, folder, remove_baseline=True, species_to_weight=None):
        self.folder = Path(folder)
        self._mmap = RaggedMmap(self.folder / "mmap")
        self.species_to_weight = None

        if (self.folder / "info.yaml").is_file():
            self.info = read_yaml(self.folder / "info.yaml")
        else:
            self.info = None

        if (self.folder / "properties.yaml").is_file():
            self.properties = normalize_properties(
                read_yaml(self.folder / "properties.yaml")
            )
        else:
            # todo: remove backwards compatibility at some point;
            #       this makes sure that legacy datasets (which stored voigt)
            #       still work for a while longer...
            self.properties = {
                "energy": {"shape": (1,), "storage": "atoms.calc"},
                "forces": {"shape": ("atom", 3), "storage": "atoms.calc"},
                "stress": {"shape": (6,), "storage": "atoms.calc"},
            }

        if remove_baseline:
            if not species_to_weight:
                self.species_to_weight = read_yaml(self.folder / "baseline.yaml")
            else:
                self.species_to_weight = species_to_weight

    def __repr__(self):
        from marathon.data import datasets

        if datasets is not None:
            # we only consider paths relative to the datasets folder; this makes
            # checkpoints somewhat portable between computers
            marathon_dataset_path = str(datasets.resolve())
            this_dataset_path = str(self.folder.resolve())
            this_dataset_path = this_dataset_path.replace(
                marathon_dataset_path, "$DATASETS"
            )
        else:
            this_dataset_path = str(self.folder.resolve())

        # used by DataLoader to verify checkpoint when restoring
        return (
            f"{self.__module__}.{self.__class__.__qualname__} reading {this_dataset_path}"
        )

    def __len__(self):
        return len(self._mmap)

    def __iter__(self):
        return self[:]

    def __getitem__(self, record_key):
        if isinstance(record_key, slice):
            # return a generator instead of list to avoid instantiating many Atoms
            def iterator():
                for idx in range(len(self))[record_key]:
                    yield self.get_atoms(idx)

            return iterator()
        else:
            return self.get_atoms(record_key)

    def get_atoms(self, index):
        flattened = self._mmap[index]

        atoms = unflatten_atoms(flattened, self.properties)
        if self.species_to_weight:
            offset = sum([self.species_to_weight[Z] for Z in atoms.get_atomic_numbers()])
            atoms.calc.results["energy"] -= offset

        if self.info is not None:
            atoms.info = self.info[index]

        return atoms

    # we can't pickle open files, so we make sure that the mmap gets re-opened
    def __setstate__(self, state):
        self.folder = Path(state["folder"])
        self._mmap = RaggedMmap(self.folder / "mmap")
        self.species_to_weight = state["species_to_weight"]
        self.info = state["info"]
        self.properties = state["properties"]

    def __getstate__(self):
        return {
            "folder": self.folder,
            "species_to_weight": self.species_to_weight,
            "info": self.info,
            "properties": self.properties,
        }
