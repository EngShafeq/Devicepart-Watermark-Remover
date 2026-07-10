"""Inventory with optimistic concurrency control and time-limited holds.

Overselling is prevented with a compare-and-swap on a per-SKU ``version``
column: every mutation reads the current version, then writes with
``WHERE sku = ? AND version = ?``. If a competing transaction slipped in
between, the row count comes back zero and we retry against the fresh state.
Because the *available* check and the version bump happen in the same
conditional UPDATE, two simultaneous buyers of the last unit cannot both
succeed — exactly one commits and the other sees :class:`OutOfStock`.

"Available" = ``on_hand - reserved``. Checkout takes a *soft* reservation
(a hold with a TTL) rather than decrementing stock immediately, so an
abandoned checkout returns the units automatically once the hold expires.
Committing the reservation at payment capture converts the hold into a real
decrement.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable, List, Optional

from .db import Database
from .errors import ConcurrencyConflict, OutOfStock, ReservationExpired

# Injectable clock so tests can advance time without sleeping.
Clock = Callable[[], float]


@dataclass
class Reservation:
    id: str
    sku: str
    quantity: int
    expires_at: float


@dataclass
class StockLevel:
    sku: str
    on_hand: int
    reserved: int
    version: int

    @property
    def available(self) -> int:
        return self.on_hand - self.reserved


class Inventory:
    def __init__(self, db: Database, *, clock: Optional[Clock] = None, max_retries: int = 5):
        self.db = db
        self.max_retries = max_retries
        if clock is None:
            import time

            clock = time.time
        self._clock = clock

    # ---- setup -----------------------------------------------------------
    def set_stock(self, sku: str, on_hand: int) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO stock (sku, on_hand, reserved, version)
                VALUES (?, ?, 0, 1)
                ON CONFLICT(sku) DO UPDATE SET
                    on_hand = excluded.on_hand,
                    version = stock.version + 1
                """,
                (sku, on_hand),
            )

    def level(self, sku: str) -> Optional[StockLevel]:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT sku, on_hand, reserved, version FROM stock WHERE sku = ?",
                (sku,),
            ).fetchone()
        if row is None:
            return None
        return StockLevel(row["sku"], row["on_hand"], row["reserved"], row["version"])

    def available(self, sku: str) -> int:
        # Reclaim any expired holds first so availability reflects reality.
        self._sweep_expired(sku)
        lvl = self.level(sku)
        return lvl.available if lvl else 0

    # ---- reservations (soft holds) --------------------------------------
    def reserve(self, sku: str, quantity: int, *, ttl_seconds: float = 900) -> Reservation:
        """Place a time-limited hold; raises :class:`OutOfStock` if it can't.

        The reservation increments ``reserved`` under a version check, so the
        availability test and the hold are atomic with respect to every other
        buyer. Expired-but-uncommitted holds for the SKU are swept first so
        their units are reusable.
        """
        if quantity <= 0:
            raise ValueError("reservation quantity must be positive")
        for _ in range(self.max_retries):
            self._sweep_expired(sku)
            lvl = self.level(sku)
            if lvl is None:
                raise OutOfStock(sku, quantity, 0)
            if lvl.available < quantity:
                raise OutOfStock(sku, quantity, lvl.available)
            res_id = uuid.uuid4().hex
            expires_at = self._clock() + ttl_seconds
            with self.db.transaction() as conn:
                cur = conn.execute(
                    """
                    UPDATE stock SET reserved = reserved + ?, version = version + 1
                    WHERE sku = ? AND version = ? AND on_hand - reserved >= ?
                    """,
                    (quantity, sku, lvl.version, quantity),
                )
                if cur.rowcount == 1:
                    conn.execute(
                        """
                        INSERT INTO reservations (id, sku, quantity, expires_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (res_id, sku, quantity, expires_at),
                    )
                    return Reservation(res_id, sku, quantity, expires_at)
            # Version moved under us — loop and re-read.
        raise ConcurrencyConflict("stock", sku)

    def commit(self, reservation_id: str) -> None:
        """Convert a hold into a real decrement of on-hand stock.

        Idempotent-ish: committing an already-committed hold is a no-op;
        committing an expired hold raises so callers re-reserve.
        """
        for _ in range(self.max_retries):
            with self.db.connect() as conn:
                res = conn.execute(
                    "SELECT * FROM reservations WHERE id = ?", (reservation_id,)
                ).fetchone()
            if res is None:
                raise KeyError(f"unknown reservation {reservation_id}")
            if res["committed"]:
                return
            if res["released"]:
                raise ReservationExpired(reservation_id)
            if res["expires_at"] < self._clock():
                self._sweep_expired(res["sku"])
                raise ReservationExpired(reservation_id)
            lvl = self.level(res["sku"])
            assert lvl is not None
            with self.db.transaction() as conn:
                cur = conn.execute(
                    """
                    UPDATE stock
                    SET on_hand = on_hand - ?, reserved = reserved - ?,
                        version = version + 1
                    WHERE sku = ? AND version = ?
                    """,
                    (res["quantity"], res["quantity"], res["sku"], lvl.version),
                )
                if cur.rowcount == 1:
                    conn.execute(
                        "UPDATE reservations SET committed = 1 WHERE id = ?",
                        (reservation_id,),
                    )
                    return
        raise ConcurrencyConflict("stock", reservation_id)

    def release(self, reservation_id: str) -> None:
        """Cancel a hold, returning its units to availability."""
        for _ in range(self.max_retries):
            with self.db.connect() as conn:
                res = conn.execute(
                    "SELECT * FROM reservations WHERE id = ?", (reservation_id,)
                ).fetchone()
            if res is None or res["released"] or res["committed"]:
                return
            lvl = self.level(res["sku"])
            assert lvl is not None
            with self.db.transaction() as conn:
                cur = conn.execute(
                    """
                    UPDATE stock SET reserved = reserved - ?, version = version + 1
                    WHERE sku = ? AND version = ?
                    """,
                    (res["quantity"], res["sku"], lvl.version),
                )
                if cur.rowcount == 1:
                    conn.execute(
                        "UPDATE reservations SET released = 1 WHERE id = ?",
                        (reservation_id,),
                    )
                    return
        raise ConcurrencyConflict("stock", reservation_id)

    # ---- direct purchase (no hold) --------------------------------------
    def decrement(self, sku: str, quantity: int) -> None:
        """Atomically decrement on-hand stock, or raise :class:`OutOfStock`.

        This is the version-checked purchase path used by the concurrency
        test: many threads call it at once for a single-unit SKU and exactly
        one succeeds.
        """
        if quantity <= 0:
            raise ValueError("decrement quantity must be positive")
        for _ in range(self.max_retries):
            lvl = self.level(sku)
            if lvl is None:
                raise OutOfStock(sku, quantity, 0)
            if lvl.available < quantity:
                raise OutOfStock(sku, quantity, lvl.available)
            with self.db.transaction() as conn:
                cur = conn.execute(
                    """
                    UPDATE stock SET on_hand = on_hand - ?, version = version + 1
                    WHERE sku = ? AND version = ? AND on_hand - reserved >= ?
                    """,
                    (quantity, sku, lvl.version, quantity),
                )
                if cur.rowcount == 1:
                    return
            # Contended: another writer bumped the version. Re-read and retry.
        # Exhausted retries under heavy contention — re-check to report why.
        lvl = self.level(sku)
        available = lvl.available if lvl else 0
        raise OutOfStock(sku, quantity, available)

    def restock(self, sku: str, quantity: int) -> None:
        """Add units back (e.g. a return was inspected and restocked)."""
        for _ in range(self.max_retries):
            lvl = self.level(sku)
            if lvl is None:
                self.set_stock(sku, quantity)
                return
            with self.db.transaction() as conn:
                cur = conn.execute(
                    """
                    UPDATE stock SET on_hand = on_hand + ?, version = version + 1
                    WHERE sku = ? AND version = ?
                    """,
                    (quantity, sku, lvl.version),
                )
                if cur.rowcount == 1:
                    return
        raise ConcurrencyConflict("stock", sku)

    # ---- housekeeping ----------------------------------------------------
    def _sweep_expired(self, sku: str) -> None:
        """Release any holds on ``sku`` whose TTL has elapsed."""
        now = self._clock()
        with self.db.connect() as conn:
            expired = conn.execute(
                """
                SELECT id, quantity FROM reservations
                WHERE sku = ? AND committed = 0 AND released = 0 AND expires_at < ?
                """,
                (sku, now),
            ).fetchall()
        for res in expired:
            for _ in range(self.max_retries):
                lvl = self.level(sku)
                if lvl is None:
                    break
                with self.db.transaction() as conn:
                    # Guard on released=0 so a concurrent sweeper can't double-free.
                    cur = conn.execute(
                        """
                        UPDATE reservations SET released = 1
                        WHERE id = ? AND released = 0 AND committed = 0
                        """,
                        (res["id"],),
                    )
                    if cur.rowcount == 0:
                        break  # someone else handled it
                    conn.execute(
                        """
                        UPDATE stock SET reserved = reserved - ?, version = version + 1
                        WHERE sku = ?
                        """,
                        (res["quantity"], sku),
                    )
                    break

    def active_reservations(self, sku: str) -> List[Reservation]:
        self._sweep_expired(sku)
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, sku, quantity, expires_at FROM reservations
                WHERE sku = ? AND committed = 0 AND released = 0
                """,
                (sku,),
            ).fetchall()
        return [Reservation(r["id"], r["sku"], r["quantity"], r["expires_at"]) for r in rows]
