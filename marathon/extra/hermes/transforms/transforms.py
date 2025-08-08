from dataclasses import dataclass

from marathon.extra.hermes.pain import (
    FilterTransform,
    MapTransform,
    RandomMapTransform,
    Record,
)


@dataclass(frozen=True)
class FilterEmpty(FilterTransform):
    def filter(self, sample):
        return len(sample.graph.centers) > 0


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
class ToSample(MapTransform):
    cutoff: float
    energy: bool = True
    forces: bool = True
    stress: bool = False

    def map(self, atoms):
        from marathon.extra.hermes.data import to_sample

        return to_sample(
            atoms, self.cutoff, energy=self.energy, forces=self.forces, stress=self.stress
        )


@dataclass(frozen=True)
class RandomRotation(RandomMapTransform):
    def random_map(self, atoms, rng):
        import numpy as np

        from ase.calculators.singlepoint import SinglePointCalculator
        from scipy.spatial.transform import Rotation

        rotation = Rotation.random(random_state=rng)
        sign = 1 if rng.random() < 0.5 else -1
        R = sign * rotation.as_matrix()

        results = atoms.calc.results
        if "forces" in results:
            F = results["forces"]
            results["forces"] = np.einsum("ab,ib->ia", R, F)
        # todo: stress (we need to transform to voigt and back)
        # if "stress" in results:
        #     s = results["stress"]
        #     results["stress"] = np.einsum("ab,cd,bd->ac", R, R, s)

        atoms = atoms.copy()
        pos = atoms.get_positions()
        atoms.set_positions(np.einsum("ab,ib->ia", R, pos))

        cell = atoms.get_cell().array
        atoms.set_cell(np.einsum("ab,Ab->Aa", R, cell))

        calc = SinglePointCalculator(atoms, **results)
        atoms.calc = calc

        return atoms


@dataclass(frozen=True)
class ToFixedLengthBatch:
    # make batches with fixed number of samples, padding out
    # nodes and edges to some reduced set of sizes
    batch_size: int
    keys: tuple = ("energy", "forces")
    drop_remainder: bool = True

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
        from marathon.data import get_batch

        num_nodes, num_edges = get_totals(records_to_batch)

        # determine padded size, making sure there is always some room for padding
        num_nodes = get_size(num_nodes + 1)
        num_edges = get_size(num_edges + 1)

        return get_batch(records_to_batch, num_nodes, num_edges, self.keys)


def get_totals(samples):
    num_nodes = 0
    num_edges = 0

    for sample in samples:
        num_nodes += sample.graph.nodes.shape[0]
        num_edges += sample.graph.edges.shape[0]

    return num_nodes, num_edges


def get_size(n):
    if n <= 64:
        return next_multiple(n, 16)

    # if n <= 256:
    #     return next_multiple(n, 64)

    if n <= 1024:
        return next_multiple(n, 256)

    if n <= 4096:
        return next_multiple(n, 1024)

    if n <= 32768:
        return next_multiple(n, 4096)

    return next_multiple(n, 16384)


def next_multiple(val, n):
    return n * (1 + int(val // n))


@dataclass(frozen=True)
class ToFixedShapeBatch:
    # make batches with fixed shape, will fail if the shapes
    # don't allow at least one sample to be batched
    # since we need a fixed number of graphs, we also
    # accept batch_size and return at most this many graphs
    # (at least one will be fake)
    num_nodes: int
    num_edges: int
    num_graphs: int
    keys: tuple = ("energy", "forces")

    def __call__(self, input_iterator):
        records_to_batch = []
        num_nodes = 0
        num_edges = 0
        last_record_metadata = None
        for input_record in input_iterator:
            this_record_metadata = input_record.metadata

            this_data = input_record.data
            this_num_nodes = this_data.graph.nodes.shape[0]
            this_num_edges = this_data.graph.edges.shape[0]

            if (
                num_nodes + this_num_nodes + 1 > self.num_nodes
                or num_edges + this_num_edges + 1 > self.num_edges
                or len(records_to_batch) + 1 == self.num_graphs
            ):
                batch = self._batch(records_to_batch)
                records_to_batch = []
                num_nodes = 0
                num_edges = 0
                yield Record(last_record_metadata.remove_record_key(), batch)

            records_to_batch.append(this_data)
            num_nodes += this_num_nodes
            num_edges += this_num_edges
            last_record_metadata = this_record_metadata

        # we exhausted the iterator, let's return the rest
        if records_to_batch:
            yield Record(
                last_record_metadata.remove_record_key(),
                self._batch(records_to_batch),
            )

    def _batch(self, records_to_batch):
        from marathon.data import get_batch

        return get_batch(
            records_to_batch,
            self.num_nodes,
            self.num_edges,
            self.keys,
            num_graphs=self.num_graphs,
        )


@dataclass(frozen=True)
class ToDenseBatch:
    # make batches kind of fixed shape:
    # we guarantee fixed num_nodes and num_graphs,
    # but not num_neighbors
    num_nodes: int
    num_graphs: int
    num_neighbors_multiple: int = 16
    keys: tuple = ("energy", "forces")
    num_neighbors: int | None = None
    extra_neighbors: int = 0

    def __call__(self, input_iterator):
        records_to_batch = []
        num_nodes = 0
        max_neighbors = 0
        last_record_metadata = None
        for input_record in input_iterator:
            this_record_metadata = input_record.metadata

            this_data = input_record.data
            this_num_nodes = this_data.graph.nodes.shape[0]
            this_max_neighbors = this_data.graph.info["max_neighbors"]

            if this_max_neighbors > max_neighbors:
                max_neighbors = this_max_neighbors

            if (
                num_nodes + this_num_nodes + 1 > self.num_nodes
                or len(records_to_batch) + 1 == self.num_graphs
            ):
                batch = self._batch(records_to_batch, max_neighbors)
                records_to_batch = []
                num_nodes = 0
                max_neighbors = 0
                yield Record(last_record_metadata.remove_record_key(), batch)

            records_to_batch.append(this_data)
            num_nodes += this_num_nodes
            last_record_metadata = this_record_metadata

            if this_max_neighbors > max_neighbors:
                max_neighbors = this_max_neighbors

        # we exhausted the iterator, let's return the rest
        if records_to_batch:
            yield Record(
                last_record_metadata.remove_record_key(),
                self._batch(records_to_batch, max_neighbors),
            )

    def _batch(self, records_to_batch, max_neighbors):
        from .dense import get_batch

        if not self.num_neighbors:
            proposed_num_neighbors = next_multiple(
                max_neighbors, self.num_neighbors_multiple
            )
            if proposed_num_neighbors >= max_neighbors + self.extra_neighbors:
                num_neighbors = proposed_num_neighbors
            else:
                num_neighbors = next_multiple(
                    max_neighbors + self.extra_neighbors, self.num_neighbors_multiple
                )

        else:
            num_neighbors = self.num_neighbors + self.extra_neighbors

        return get_batch(
            records_to_batch,
            self.num_nodes,
            self.num_graphs,
            num_neighbors,
            self.keys,
        )
