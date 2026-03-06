from grain.python import Record, RecordMetadata

from marathon.data import to_sample
from marathon.grain import ToEdgeToEdgeBatch, ToFixedLengthBatch, ToFixedShapeBatch

from test_grain_data_source import make_fake_atoms


def make_samples(n=10, seed=42):
    """Create Sample objects for testing batchers."""
    atoms_list = make_fake_atoms(n_structures=n, seed=seed)
    return [to_sample(atoms, cutoff=5.0) for atoms in atoms_list]


def test_to_fixed_length_batch():
    """Test ToFixedLengthBatch accumulates samples and creates padded batches."""
    samples = make_samples(n=10)

    batcher = ToFixedLengthBatch(batch_size=3)
    records = [
        Record(RecordMetadata(index=i, record_key=i), s) for i, s in enumerate(samples)
    ]
    batches = list(batcher(iter(records)))

    # 10 samples / 3 = 3 full batches (drop_remainder=True by default)
    assert len(batches) == 3

    # verify batch structure
    batch = batches[0].data
    assert hasattr(batch, "atomic_numbers")
    assert hasattr(batch, "displacements")
    assert hasattr(batch, "labels")
    assert batch.atom_mask.sum() > 0  # has real atoms


def test_to_fixed_length_batch_keep_remainder():
    """Test drop_remainder=False keeps incomplete final batch."""
    samples = make_samples(n=10)

    batcher = ToFixedLengthBatch(batch_size=3, drop_remainder=False)
    records = [
        Record(RecordMetadata(index=i, record_key=i), s) for i, s in enumerate(samples)
    ]
    batches = list(batcher(iter(records)))

    # 10 samples → 3 + 3 + 3 + 1 = 4 batches
    assert len(batches) == 4


def test_to_fixed_shape_batch():
    """Test ToFixedShapeBatch respects shape limits."""
    samples = make_samples(n=10)

    batcher = ToFixedShapeBatch(
        num_atoms=100,
        num_pairs=500,
        num_structures=5,
    )

    records = [
        Record(RecordMetadata(index=i, record_key=i), s) for i, s in enumerate(samples)
    ]
    batches = list(batcher(iter(records)))

    # verify all batches have exact shapes
    for b in batches:
        assert b.data.atomic_numbers.shape[0] == 100
        assert b.data.displacements.shape[0] == 500


def test_to_edge_to_edge_batch():
    """Test ToEdgeToEdgeBatch produces neighbor-indexed format."""
    samples = make_samples(n=10)

    batcher = ToEdgeToEdgeBatch(
        num_atoms=100,
        num_structures=5,
    )

    records = [
        Record(RecordMetadata(index=i, record_key=i), s) for i, s in enumerate(samples)
    ]
    batches = list(batcher(iter(records)))

    # verify edge-to-edge specific fields
    batch = batches[0].data
    assert hasattr(batch, "reverse")  # ij -> ji mapping
    assert hasattr(batch, "centers")
    assert hasattr(batch, "others")


def test_strategy_powers_of_2():
    """Test strategy parameter produces powers of 2."""
    samples = make_samples(n=5)

    batcher = ToFixedLengthBatch(batch_size=3, strategy="powers_of_2")
    records = [
        Record(RecordMetadata(index=i, record_key=i), s) for i, s in enumerate(samples)
    ]
    batches = list(batcher(iter(records)))

    # check that atom/pair dimensions are powers of 2
    batch = batches[0].data
    num_atoms = batch.atomic_numbers.shape[0]
    num_pairs = batch.displacements.shape[0]

    # power of 2 check: n & (n-1) == 0
    assert num_atoms & (num_atoms - 1) == 0, f"num_atoms={num_atoms} not power of 2"
    assert num_pairs & (num_pairs - 1) == 0, f"num_pairs={num_pairs} not power of 2"


def test_padding_guarantees():
    """Test that all batchers guarantee at least 1 padding slot for atoms, pairs, structures."""
    samples = make_samples(n=10)
    records = [
        Record(RecordMetadata(index=i, record_key=i), s) for i, s in enumerate(samples)
    ]

    # ToFixedLengthBatch
    batcher1 = ToFixedLengthBatch(batch_size=3)
    for batch_record in batcher1(iter(records)):
        batch = batch_record.data
        real_atoms = batch.atom_mask.sum()
        real_pairs = batch.pair_mask.sum()
        real_structures = batch.structure_mask.sum()
        assert real_atoms < len(batch.atom_mask), "no atom padding in ToFixedLengthBatch"
        assert real_pairs < len(batch.pair_mask), "no pair padding in ToFixedLengthBatch"
        assert real_structures < len(batch.structure_mask), (
            "no structure padding in ToFixedLengthBatch"
        )

    # ToFixedShapeBatch
    records = [
        Record(RecordMetadata(index=i, record_key=i), s) for i, s in enumerate(samples)
    ]
    batcher2 = ToFixedShapeBatch(num_atoms=100, num_pairs=500, num_structures=5)
    for batch_record in batcher2(iter(records)):
        batch = batch_record.data
        real_atoms = batch.atom_mask.sum()
        real_pairs = batch.pair_mask.sum()
        real_structures = batch.structure_mask.sum()
        assert real_atoms < len(batch.atom_mask), "no atom padding in ToFixedShapeBatch"
        assert real_pairs < len(batch.pair_mask), "no pair padding in ToFixedShapeBatch"
        assert real_structures < len(batch.structure_mask), (
            "no structure padding in ToFixedShapeBatch"
        )

    # ToEdgeToEdgeBatch
    records = [
        Record(RecordMetadata(index=i, record_key=i), s) for i, s in enumerate(samples)
    ]
    batcher3 = ToEdgeToEdgeBatch(num_atoms=100, num_structures=5)
    for batch_record in batcher3(iter(records)):
        batch = batch_record.data
        real_atoms = batch.atom_mask.sum()
        real_pairs = batch.pair_mask.sum()
        real_structures = batch.structure_mask.sum()
        assert real_atoms < len(batch.atom_mask), "no atom padding in ToEdgeToEdgeBatch"
        assert real_pairs < len(batch.pair_mask), "no pair padding in ToEdgeToEdgeBatch"
        assert real_structures < len(batch.structure_mask), (
            "no structure padding in ToEdgeToEdgeBatch"
        )


if __name__ == "__main__":
    test_to_fixed_length_batch()
    test_to_fixed_length_batch_keep_remainder()
    test_to_fixed_shape_batch()
    test_to_edge_to_edge_batch()
    test_strategy_powers_of_2()
    test_padding_guarantees()
    print("All batcher tests passed!")
