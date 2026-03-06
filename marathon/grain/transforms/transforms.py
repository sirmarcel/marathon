from dataclasses import dataclass

from marathon.data.properties import DEFAULT_PROPERTIES
from grain.python import (
    FilterTransform,
    MapTransform,
    RandomMapTransform,
    Record,
)
from marathon.utils import frozen, next_size


@dataclass(frozen=True)
class FilterEmpty(FilterTransform):
    def filter(self, sample):
        return len(sample.structure["centers"]) > 0


@dataclass(frozen=True)
class FilterNoop(FilterTransform):
    def filter(self, item):
        return True


@dataclass(frozen=True)
class FilterAboveNumAtoms(FilterTransform):
    threshold: int

    def filter(self, atoms):
        return len(atoms) <= self.threshold


@dataclass(frozen=True)
class FilterTooSmall(FilterTransform):
    cutoff: float

    def filter(self, atoms):
        import numpy as np

        if not atoms.pbc.all():
            return True

        cell = atoms.get_cell().array  # [A, a]
        inv = np.linalg.inv(cell).T  # [X, x]
        normals = inv / np.linalg.norm(inv, axis=1)[:, None]  # [X, x]
        heights = (cell * normals).sum(axis=1)

        return heights.min() > self.cutoff * 2


@dataclass(frozen=True)
class FilterMixedPBC(FilterTransform):
    def filter(self, atoms):
        if atoms.pbc.all():
            return True
        elif not atoms.pbc.any():
            return True
        else:
            return False


@frozen
class ToSample(MapTransform):
    cutoff: float
    # TODO: remove energy/forces/stress bools, use keys/properties instead
    energy: bool = True
    forces: bool = True
    stress: bool = False
    keys: tuple = None
    properties: dict = None
    float_dtype: str = "float32"
    int_dtype: str = "int32"

    def map(self, atoms):
        import numpy as np

        from marathon.data import to_sample

        float_dtype = getattr(np, self.float_dtype)
        int_dtype = getattr(np, self.int_dtype)
        properties = self.properties if self.properties is not None else DEFAULT_PROPERTIES

        return to_sample(
            atoms,
            self.cutoff,
            keys=self.keys,
            energy=self.energy,
            forces=self.forces,
            stress=self.stress,
            properties=properties,
            float_dtype=float_dtype,
            int_dtype=int_dtype,
        )


@dataclass(frozen=True)
class RandomRotation(RandomMapTransform):
    keys: tuple = ("forces", "stress")

    def random_map(self, atoms, rng):
        import numpy as np

        from ase.calculators.singlepoint import SinglePointCalculator
        from scipy.spatial.transform import Rotation

        for key in self.keys:
            if key not in ("forces", "stress"):
                raise ValueError(
                    f"RandomRotation only supports forces and stress, got: {key}"
                )

        rotation = Rotation.random(random_state=rng)
        sign = 1 if rng.random() < 0.5 else -1
        R = sign * rotation.as_matrix()

        results = atoms.calc.results
        if "forces" in self.keys and "forces" in results:
            F = results["forces"]
            results["forces"] = np.einsum("ab,ib->ia", R, F)

        if "stress" in self.keys and "stress" in results:
            stress = results["stress"]
            if stress.shape == (6,):
                from ase.constraints import (
                    full_3x3_to_voigt_6_stress,
                    voigt_6_to_full_3x3_stress,
                )

                stress = np.einsum("ab,cd,bd->ac", R, R, voigt_6_to_full_3x3_stress(stress))
                results["stress"] = full_3x3_to_voigt_6_stress(stress)
            elif stress.shape == (3, 3):
                results["stress"] = np.einsum("ab,cd,bd->ac", R, R, stress)
            else:
                raise ValueError(f"found stress, but unknown shape {stress.shape}")

        atoms = atoms.copy()
        pos = atoms.get_positions()
        atoms.set_positions(np.einsum("ab,ib->ia", R, pos))

        cell = atoms.get_cell().array
        atoms.set_cell(np.einsum("ab,Ab->Aa", R, cell))

        calc = SinglePointCalculator(atoms, **results)
        atoms.calc = calc

        return atoms


@frozen
class ToFixedLengthBatch:
    # make batches with fixed number of samples, padding out
    # atomic_numbers and displacements to some reduced set of sizes
    batch_size: int
    keys: tuple = ("energy", "forces")
    properties: dict = None
    drop_remainder: bool = True
    strategy: str = "multiples"

    def __call__(self, input_iterator):
        records_to_batch = []
        last_record_metadata = None
        for input_record in input_iterator:
            last_record_metadata = input_record.metadata
            records_to_batch.append(input_record.data)
            if len(records_to_batch) == self.batch_size:
                batch = self._batch(records_to_batch)
                records_to_batch = []
                yield Record(last_record_metadata.remove_record_key(), batch)

        # we exhausted the iterator but haven't returned a batch
        # ... we either drop the rest or, below, return a smaller batch
        if records_to_batch and not self.drop_remainder:
            yield Record(
                last_record_metadata.remove_record_key(),
                self._batch(records_to_batch),
            )

    def _batch(self, records_to_batch):
        from marathon.data import batch_samples

        num_atoms, num_pairs = get_totals(records_to_batch)

        # determine padded size, making sure there is always some room for padding
        num_atoms = next_size(num_atoms + 1, strategy=self.strategy)
        num_pairs = next_size(num_pairs + 1, strategy=self.strategy)

        properties = self.properties if self.properties is not None else DEFAULT_PROPERTIES
        return batch_samples(
            records_to_batch, num_atoms, num_pairs, self.keys, properties=properties
        )


def get_totals(samples):
    num_atoms = 0
    num_pairs = 0

    for sample in samples:
        num_atoms += sample.structure["positions"].shape[0]
        num_pairs += sample.structure["displacements"].shape[0]

    return num_atoms, num_pairs


@frozen
class ToFixedShapeBatch:
    # make batches with fixed shape, will fail if the shapes
    # don't allow at least one sample to be batched
    # since we need a fixed number of graphs, we also
    # accept batch_size and return at most this many graphs
    # (at least one will be fake)
    num_atoms: int
    num_pairs: int
    num_structures: int
    keys: tuple = ("energy", "forces")
    properties: dict = None

    def __call__(self, input_iterator):
        records_to_batch = []
        num_atoms = 0
        num_pairs = 0
        last_record_metadata = None
        for input_record in input_iterator:
            this_record_metadata = input_record.metadata

            this_data = input_record.data
            this_num_atoms = this_data.structure["atomic_numbers"].shape[0]
            this_num_pairs = this_data.structure["displacements"].shape[0]

            if (
                num_atoms + this_num_atoms + 1 > self.num_atoms
                or num_pairs + this_num_pairs + 1 > self.num_pairs
                or len(records_to_batch) + 1 == self.num_structures
            ):
                batch = self._batch(records_to_batch)
                records_to_batch = []
                num_atoms = 0
                num_pairs = 0
                yield Record(last_record_metadata.remove_record_key(), batch)

            records_to_batch.append(this_data)
            num_atoms += this_num_atoms
            num_pairs += this_num_pairs
            last_record_metadata = this_record_metadata

        # we exhausted the iterator, let's return the rest
        if records_to_batch:
            yield Record(
                last_record_metadata.remove_record_key(),
                self._batch(records_to_batch),
            )

    def _batch(self, records_to_batch):
        from marathon.data import batch_samples

        properties = self.properties if self.properties is not None else DEFAULT_PROPERTIES
        return batch_samples(
            records_to_batch,
            self.num_atoms,
            self.num_pairs,
            self.keys,
            num_structures=self.num_structures,
            properties=properties,
        )


@frozen
class ToEdgeToEdgeBatch:
    # make edge-to-edge transformer batches w/
    # fixed number of structures, but varying number of atoms and neighbors
    # (optionally w/ some guaranteed extra capacity)
    num_structures: int
    num_atoms: int | None = None  # if None, compute dynamically
    num_neighbors: int | None = None  # if None, compute dynamically
    keys: tuple = ("energy", "forces")
    properties: dict = None
    extra_neighbors: int = 1
    strategy: str = "multiples"

    def __call__(self, input_iterator):
        records_to_batch = []
        total_atoms = 0
        max_neighbors = 0
        last_record_metadata = None
        for input_record in input_iterator:
            this_record_metadata = input_record.metadata

            this_data = input_record.data
            this_num_atoms = this_data.structure["atomic_numbers"].shape[0]
            this_max_neighbors = this_data.structure["max_neighbors"]

            # check if adding this sample would exceed limits (before updating counters)
            exceeds_atoms = (
                self.num_atoms is not None
                and total_atoms + this_num_atoms + 1 > self.num_atoms
            )
            exceeds_structures = len(records_to_batch) + 1 == self.num_structures

            if exceeds_atoms or exceeds_structures:
                batch = self._batch(records_to_batch, total_atoms, max_neighbors)
                records_to_batch = []
                total_atoms = 0
                max_neighbors = 0
                yield Record(last_record_metadata.remove_record_key(), batch)

            # now add the sample and update counters
            records_to_batch.append(this_data)
            total_atoms += this_num_atoms
            last_record_metadata = this_record_metadata
            if this_max_neighbors > max_neighbors:
                max_neighbors = this_max_neighbors

        # we exhausted the iterator, let's return the rest
        if records_to_batch:
            yield Record(
                last_record_metadata.remove_record_key(),
                self._batch(records_to_batch, total_atoms, max_neighbors),
            )

    def _batch(self, records_to_batch, total_atoms, max_neighbors):
        from marathon.extra.edge_to_edge import batch_samples

        # determine num_atoms
        if self.num_atoms is not None:
            num_atoms = self.num_atoms
        else:
            num_atoms = next_size(total_atoms + 1, strategy=self.strategy)

        # determine num_neighbors
        if self.num_neighbors is not None:
            num_neighbors = self.num_neighbors + self.extra_neighbors
        else:
            num_neighbors = next_size(
                max_neighbors + self.extra_neighbors, strategy=self.strategy
            )

        properties = self.properties if self.properties is not None else DEFAULT_PROPERTIES
        return batch_samples(
            records_to_batch,
            self.num_structures,
            num_atoms,
            num_neighbors,
            self.keys,
            properties=properties,
        )
