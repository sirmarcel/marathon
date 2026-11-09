import numpy as np


class Mixture:
    """Concatenates sources; an integer weight repeats a source that many times."""

    def __init__(self, sources):
        self.sources, self.weights = map(list, zip(*sources))
        counts = [w * len(s) for s, w in sources]
        self.length = sum(counts)
        self.starts = np.cumsum([0] + counts[:-1])  # first virtual index of each source

    def __len__(self):
        return self.length

    def __iter__(self):
        return self[:]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return (self[i] for i in range(len(self))[idx])
        idx = range(len(self))[idx]  # negative indices, IndexError

        # the source whose segment contains idx
        k = int(np.searchsorted(self.starts, idx, side="right")) - 1
        atoms = self.sources[k][(idx - self.starts[k]) % len(self.sources[k])]
        atoms.info["source_id"] = k
        return atoms

    def __repr__(self):
        parts = ", ".join(f"{w} x {s!r}" for s, w in zip(self.sources, self.weights))
        return f"Mixture({parts})"
