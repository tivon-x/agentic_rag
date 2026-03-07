from __future__ import annotations

import os
import pickle
from pathlib import Path

from indexing.bm25_index import BM25Bundle


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save_bm25_bundle(path: str | Path, bundle: BM25Bundle) -> None:
    """Persist BM25 bundle.

    Security note: this uses pickle; only load from trusted files.
    """
    p = Path(path)
    _atomic_write_bytes(p, pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL))


def load_bm25_bundle(path: str | Path) -> BM25Bundle:
    """Load BM25 bundle.

    Security note: this uses pickle; only load from trusted files.
    """
    p = Path(path)
    obj = pickle.loads(p.read_bytes())
    if not isinstance(obj, BM25Bundle):
        raise TypeError(f"Invalid BM25 bundle type: {type(obj)!r}")
    obj.rebuild_index()
    return obj
