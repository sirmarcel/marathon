import numpy as np

import os
from operator import index
from pathlib import Path

DB_SUFFIXES = (".db", ".json", ".aselmdb")


class AseDB:
    """Indexable, picklable view of the rows in ase.db files, ordered by file, then id.

    paths: one file, a folder (its .db/.json/.aselmdb files, sorted), or a list of files.
    to_atoms: row -> Atoms; default is row.toatoms() with key_value_pairs and data
        merged into atoms.info.

    Reads any format ase.db.connect opens (.aselmdb needs ase-db-backends). Files are
    opened lazily, once per process and shared between instances, since lmdb allows
    one environment per file per process. Pickling carries no handles.
    """

    def __init__(self, paths, to_atoms=None):
        self.files = _resolve(paths)
        self.to_atoms = to_atoms or default_to_atoms

        counts = []
        for f in self.files:
            db = _connect(f)
            counts.append(len(_ids(db)))
            _close(db)
        self.offsets = np.concatenate([[0], np.cumsum(counts)]).astype(int)

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, idx):
        idx = index(idx)
        if not 0 <= idx < len(self):
            raise IndexError(idx)

        k = int(np.searchsorted(self.offsets, idx, side="right")) - 1
        db, ids = _open(self.files[k])
        return self.to_atoms(db._get_row(ids[idx - self.offsets[k]]))

    def close(self):
        """Close this process's handles; they reopen on the next access."""
        close_all()

    def __repr__(self):
        return f"AseDB({len(self.files)} files, {len(self)} rows)"


def default_to_atoms(row):
    atoms = row.toatoms()
    atoms.info.update(row.key_value_pairs)
    atoms.info.update(row.data)
    return atoms


# (pid, path) -> (db, ids); handles inherited through a fork belong to the parent
_open_dbs = {}


def _open(path):
    key = (os.getpid(), path)
    if key not in _open_dbs:
        db = _connect(path)
        _open_dbs[key] = (db, _ids(db))
    return _open_dbs[key]


def close_all():
    pid = os.getpid()
    for key in list(_open_dbs):
        db, _ = _open_dbs.pop(key)
        if key[0] == pid:
            _close(db)


def _resolve(paths):
    if isinstance(paths, (str, Path)):
        path = Path(paths)
        if path.is_dir():
            files = sorted(p for p in path.iterdir() if p.suffix in DB_SUFFIXES)
            if not files:
                raise FileNotFoundError(f"no {DB_SUFFIXES} files in {path}")
        else:
            files = [path]
    else:
        files = [Path(p) for p in paths]
    return [str(f) for f in files]


def _connect(path):
    import ase.db

    kwargs = {"readonly": True} if path.endswith(".aselmdb") else {}
    return ase.db.connect(path, use_lock_file=False, **kwargs)


def _ids(db):
    # aselmdb keeps its (sorted) id list; the others need a query
    if hasattr(db, "ids"):
        return db.ids
    return sorted(row.id for row in db.select(columns=["id"], include_data=False))


def _close(db):
    if hasattr(db, "close"):
        db.close()
