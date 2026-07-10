"""SQLite persistence layer.

Cart and order state must survive browser closure, device switching, and
server restarts, so the durable state lives in SQLite (a file on disk),
not in process memory. A single :class:`Database` owns the schema and hands
out short-lived connections.

WAL journalling plus a busy timeout lets many threads read while one writes,
which — together with the version-checked ("optimistic") writes in
:mod:`ecommerce.inventory` — is what prevents overselling under concurrent
purchase attempts.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS stock (
    sku         TEXT PRIMARY KEY,
    on_hand     INTEGER NOT NULL,     -- physical units in the warehouse
    reserved    INTEGER NOT NULL,     -- units held by open reservations
    version     INTEGER NOT NULL      -- bumped on every mutation (optimistic CC)
);

CREATE TABLE IF NOT EXISTS reservations (
    id          TEXT PRIMARY KEY,
    sku         TEXT NOT NULL,
    quantity    INTEGER NOT NULL,
    expires_at  REAL NOT NULL,        -- epoch seconds; hold released after this
    committed   INTEGER NOT NULL DEFAULT 0,
    released    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS carts (
    id          TEXT PRIMARY KEY,
    owner       TEXT,                 -- user id, or NULL for an anonymous cart
    currency    TEXT NOT NULL,
    version     INTEGER NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS cart_items (
    cart_id     TEXT NOT NULL,
    sku         TEXT NOT NULL,
    quantity    INTEGER NOT NULL,
    unit_price  INTEGER NOT NULL,     -- minor units, snapshot at add time
    PRIMARY KEY (cart_id, sku)
);

CREATE TABLE IF NOT EXISTS orders (
    id          TEXT PRIMARY KEY,
    owner       TEXT,
    currency    TEXT NOT NULL,
    state       TEXT NOT NULL,
    total       INTEGER NOT NULL,
    version     INTEGER NOT NULL,
    created_at  REAL NOT NULL,
    payload     TEXT NOT NULL         -- JSON: lines, addresses, totals breakdown
);

CREATE TABLE IF NOT EXISTS order_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id    TEXT NOT NULL,
    from_state  TEXT,
    to_state    TEXT NOT NULL,
    ok          INTEGER NOT NULL,     -- 1 = applied, 0 = rejected transition
    reason      TEXT,
    at          REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS coupon_usage (
    code        TEXT NOT NULL,
    order_id    TEXT NOT NULL,
    PRIMARY KEY (code, order_id)
);

CREATE TABLE IF NOT EXISTS analytics_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    name        TEXT NOT NULL,
    props       TEXT NOT NULL,        -- JSON
    at          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cart_items_cart ON cart_items(cart_id);
CREATE INDEX IF NOT EXISTS idx_order_events_order ON order_events(order_id);
CREATE INDEX IF NOT EXISTS idx_analytics_session ON analytics_events(session_id);
"""


class Database:
    """Owns a SQLite file and vends connections.

    Pass ``":memory:"`` for tests, but note an in-memory database is private
    to one connection — use a temp file when exercising cross-thread
    concurrency so every thread sees the same durable state.
    """

    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._local = threading.local()
        # For a shared in-memory DB we must keep one connection alive.
        self._shared = None
        if path == ":memory:":
            self._shared = self._new_connection()
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def _new_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            timeout=30,
            isolation_level=None,  # autocommit; we manage transactions explicitly
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if self._shared is not None:
            yield self._shared
            return
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._new_connection()
            self._local.conn = conn
        yield conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a BEGIN IMMEDIATE ... COMMIT block; rolls back on error.

        BEGIN IMMEDIATE takes the write lock up front so two concurrent
        transactions serialise cleanly instead of one failing late with
        "database is locked" after doing work.
        """
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
