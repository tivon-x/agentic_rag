from __future__ import annotations


class SqliteNodeStore:
    """Reserved adapter slot for a future SQLite-backed node store."""

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "SQLite node store is not implemented yet. Set NODE_BACKEND=json."
        )
