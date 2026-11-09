import numpy as np

from operator import index


class Mixture:
    """Concatenates sources; an integer weight repeats a source that many times."""

    def __init__(self, sources):
        self.sources, self.weights = map(list, zip(*sources))
        counts = [w * len(s) for s, w in sources]
        self.length = sum(counts)
        self.starts = np.cumsum([0] + counts[:-1])  # first virtual index of each source

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = index(idx)
        if not 0 <= idx < len(self):
            raise IndexError(idx)

        # the source whose segment contains idx
        k = int(np.searchsorted(self.starts, idx, side="right")) - 1
        atoms = self.sources[k][(idx - self.starts[k]) % len(self.sources[k])]
        atoms.info["source_id"] = k
        return atoms

    def __repr__(self):
        parts = ", ".join(f"{w} x {s!r}" for s, w in zip(self.sources, self.weights))
        return f"Mixture({parts})"
