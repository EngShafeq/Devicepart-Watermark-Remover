"""Order lifecycle as an explicit, validated state machine.

An order moves through a fixed set of states, and only the transitions
declared in :data:`TRANSITIONS` are legal. Every attempt — successful or
not — is written to the ``order_events`` audit table, so an illegal
transition is *rejected with a clear error and logged* for monitoring rather
than silently ignored.

The persisted order carries a ``version`` column; :meth:`OrderService.transition`
uses it for an optimistic write so two workers can't advance the same order
concurrently.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from .db import Database
from .errors import ConcurrencyConflict, IllegalTransition
from .money import Money


class OrderState(str, Enum):
    PENDING = "pending"
    PAYMENT_AUTHORIZED = "payment_authorized"
    PAYMENT_CAPTURED = "payment_captured"
    FULFILLMENT_PROCESSING = "fulfillment_processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


# Legal forward transitions. Cancellation is allowed from any pre-shipment
# state; once shipped, the return/refund flow (see returns.py) takes over.
TRANSITIONS: Dict[OrderState, Set[OrderState]] = {
    OrderState.PENDING: {OrderState.PAYMENT_AUTHORIZED, OrderState.CANCELLED},
    OrderState.PAYMENT_AUTHORIZED: {OrderState.PAYMENT_CAPTURED, OrderState.CANCELLED},
    OrderState.PAYMENT_CAPTURED: {OrderState.FULFILLMENT_PROCESSING, OrderState.CANCELLED},
    OrderState.FULFILLMENT_PROCESSING: {OrderState.SHIPPED, OrderState.CANCELLED},
    OrderState.SHIPPED: {OrderState.DELIVERED},
    OrderState.DELIVERED: set(),
    OrderState.CANCELLED: set(),
}


def is_legal(frm: OrderState, to: OrderState) -> bool:
    return to in TRANSITIONS.get(frm, set())


@dataclass
class OrderLine:
    sku: str
    quantity: int
    unit_price: Money
    line_total: Money


@dataclass
class Order:
    id: str
    owner: Optional[str]
    currency: str
    state: OrderState
    total: Money
    version: int
    lines: List[OrderLine] = field(default_factory=list)
    payload: Dict = field(default_factory=dict)


@dataclass
class TransitionLog:
    from_state: Optional[str]
    to_state: str
    ok: bool
    reason: Optional[str]
    at: float


class OrderService:
    def __init__(self, db: Database, *, clock=None):
        self.db = db
        if clock is None:
            import time

            clock = time.time
        self._clock = clock

    # ---- creation --------------------------------------------------------
    def create(
        self,
        *,
        owner: Optional[str],
        currency: str,
        lines: List[OrderLine],
        total: Money,
        payload: Optional[Dict] = None,
    ) -> Order:
        order_id = uuid.uuid4().hex
        body = dict(payload or {})
        body["lines"] = [
            {
                "sku": l.sku,
                "quantity": l.quantity,
                "unit_price": l.unit_price.amount,
                "line_total": l.line_total.amount,
            }
            for l in lines
        ]
        now = self._clock()
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO orders (id, owner, currency, state, total, version, created_at, payload)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (order_id, owner, currency, OrderState.PENDING.value, total.amount, now, json.dumps(body)),
            )
            conn.execute(
                "INSERT INTO order_events (order_id, from_state, to_state, ok, reason, at) VALUES (?, ?, ?, 1, ?, ?)",
                (order_id, None, OrderState.PENDING.value, "created", now),
            )
        return Order(order_id, owner, currency, OrderState.PENDING, total, 1, lines, body)

    def get(self, order_id: str) -> Optional[Order]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload"])
        lines = [
            OrderLine(
                l["sku"],
                l["quantity"],
                Money(l["unit_price"], row["currency"]),
                Money(l["line_total"], row["currency"]),
            )
            for l in payload.get("lines", [])
        ]
        return Order(
            row["id"],
            row["owner"],
            row["currency"],
            OrderState(row["state"]),
            Money(row["total"], row["currency"]),
            row["version"],
            lines,
            payload,
        )

    # ---- state machine ---------------------------------------------------
    def transition(self, order_id: str, to: OrderState, *, reason: str = "") -> Order:
        """Move an order to ``to``, or reject and log an illegal attempt.

        On an illegal transition the attempt is recorded with ``ok = 0`` and
        an :class:`IllegalTransition` is raised carrying both states, so the
        caller gets a clear message and monitoring can alert on the audit row.
        """
        order = self.get(order_id)
        if order is None:
            raise KeyError(f"unknown order {order_id}")
        frm = order.state
        if not is_legal(frm, to):
            self._log(order_id, frm, to, ok=False, reason=reason or "illegal transition")
            raise IllegalTransition(frm.value, to.value)
        with self.db.transaction() as conn:
            cur = conn.execute(
                "UPDATE orders SET state = ?, version = version + 1 WHERE id = ? AND version = ?",
                (to.value, order_id, order.version),
            )
            if cur.rowcount != 1:
                raise ConcurrencyConflict("order", order_id)
            conn.execute(
                "INSERT INTO order_events (order_id, from_state, to_state, ok, reason, at) VALUES (?, ?, ?, 1, ?, ?)",
                (order_id, frm.value, to.value, reason or "", self._clock()),
            )
        updated = self.get(order_id)
        assert updated is not None
        return updated

    def can_transition(self, order_id: str, to: OrderState) -> bool:
        order = self.get(order_id)
        return order is not None and is_legal(order.state, to)

    def history(self, order_id: str) -> List[TransitionLog]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT from_state, to_state, ok, reason, at FROM order_events WHERE order_id = ? ORDER BY id",
                (order_id,),
            ).fetchall()
        return [
            TransitionLog(r["from_state"], r["to_state"], bool(r["ok"]), r["reason"], r["at"])
            for r in rows
        ]

    def violations(self, order_id: str) -> List[TransitionLog]:
        """Just the rejected transitions — the monitoring signal."""
        return [e for e in self.history(order_id) if not e.ok]

    def _log(self, order_id, frm, to, *, ok: bool, reason: str) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO order_events (order_id, from_state, to_state, ok, reason, at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    order_id,
                    frm.value if isinstance(frm, OrderState) else frm,
                    to.value if isinstance(to, OrderState) else to,
                    1 if ok else 0,
                    reason,
                    self._clock(),
                ),
            )
