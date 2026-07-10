"""Returns & exchanges: RMA -> label -> inspection -> refund -> restock.

A return authorisation (RMA) is opened against a delivered order for a subset
of its lines. It moves through its own small state machine — requested ->
label_issued -> in_transit -> inspecting -> (approved|rejected) -> refunded /
restocked — mirroring the physical process. On approval the refund amount is
computed from the original order's per-line prices (integer minor units), the
refund is issued through the payment gateway, and accepted items are handed
back to :class:`Inventory` as restock.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .inventory import Inventory
from .money import Money
from .orders import Order, OrderService


class RMAState(str, Enum):
    REQUESTED = "requested"
    LABEL_ISSUED = "label_issued"
    IN_TRANSIT = "in_transit"
    INSPECTING = "inspecting"
    APPROVED = "approved"
    REJECTED = "rejected"
    REFUNDED = "refunded"


RMA_TRANSITIONS: Dict[RMAState, set] = {
    RMAState.REQUESTED: {RMAState.LABEL_ISSUED, RMAState.REJECTED},
    RMAState.LABEL_ISSUED: {RMAState.IN_TRANSIT, RMAState.REJECTED},
    RMAState.IN_TRANSIT: {RMAState.INSPECTING},
    RMAState.INSPECTING: {RMAState.APPROVED, RMAState.REJECTED},
    RMAState.APPROVED: {RMAState.REFUNDED},
    RMAState.REJECTED: set(),
    RMAState.REFUNDED: set(),
}


@dataclass
class ReturnLine:
    sku: str
    quantity: int
    reason: str = ""
    restock: bool = True  # False for damaged/opened items pulled from resale
    accepted: Optional[bool] = None  # set during inspection


@dataclass
class RMA:
    id: str
    order_id: str
    state: RMAState
    lines: List[ReturnLine]
    label_tracking: Optional[str] = None
    refund_amount: Optional[Money] = None
    history: List[str] = field(default_factory=list)


class RefundGateway:
    """Minimal refund side of a payment gateway for the returns flow."""

    def refund(self, capture_id: str, amount: Money) -> str:  # pragma: no cover - trivial
        return f"refund_{capture_id}_{amount.amount}"


class ReturnsService:
    def __init__(self, orders: OrderService, inventory: Inventory, refunds: RefundGateway):
        self.orders = orders
        self.inventory = inventory
        self.refunds = refunds
        self._rmas: Dict[str, RMA] = {}

    # ---- open an RMA -----------------------------------------------------
    def open_rma(self, order_id: str, lines: List[ReturnLine]) -> RMA:
        """Create an RMA for returnable quantities of a delivered order."""
        order = self.orders.get(order_id)
        if order is None:
            raise KeyError(f"unknown order {order_id}")
        if order.state.value not in ("delivered", "shipped"):
            raise ValueError("only shipped/delivered orders can be returned")
        self._validate_quantities(order, lines)
        rma = RMA(uuid.uuid4().hex, order_id, RMAState.REQUESTED, lines)
        rma.history.append("requested")
        self._rmas[rma.id] = rma
        return rma

    def get(self, rma_id: str) -> Optional[RMA]:
        return self._rmas.get(rma_id)

    # ---- state transitions ----------------------------------------------
    def _transition(self, rma: RMA, to: RMAState, note: str = "") -> None:
        if to not in RMA_TRANSITIONS.get(rma.state, set()):
            raise ValueError(f"illegal RMA transition {rma.state.value} -> {to.value}")
        rma.state = to
        rma.history.append(f"{to.value}:{note}" if note else to.value)

    def issue_label(self, rma_id: str) -> str:
        rma = self._require(rma_id)
        tracking = f"RET{rma.id[:10].upper()}"
        rma.label_tracking = tracking
        self._transition(rma, RMAState.LABEL_ISSUED, tracking)
        return tracking

    def mark_in_transit(self, rma_id: str) -> None:
        self._transition(self._require(rma_id), RMAState.IN_TRANSIT)

    def begin_inspection(self, rma_id: str) -> None:
        self._transition(self._require(rma_id), RMAState.INSPECTING)

    def record_inspection(self, rma_id: str, results: Dict[str, bool]) -> None:
        """Mark each returned SKU accepted or rejected during inspection."""
        rma = self._require(rma_id)
        if rma.state is not RMAState.INSPECTING:
            raise ValueError("RMA is not under inspection")
        for line in rma.lines:
            if line.sku in results:
                line.accepted = results[line.sku]

    # ---- settle ----------------------------------------------------------
    def settle(self, rma_id: str) -> Money:
        """Approve, refund accepted lines, and restock resalable units.

        Returns the total refunded amount. Only lines accepted during
        inspection are refunded; only accepted lines flagged ``restock`` are
        returned to inventory (damaged goods stay out of resale).
        """
        rma = self._require(rma_id)
        order = self.orders.get(rma.order_id)
        assert order is not None
        if rma.state is RMAState.INSPECTING:
            # Default any un-inspected line to accepted.
            for line in rma.lines:
                if line.accepted is None:
                    line.accepted = True
            self._transition(rma, RMAState.APPROVED)
        if rma.state is not RMAState.APPROVED:
            raise ValueError("RMA must be approved before settlement")

        unit_prices = {l.sku: l.unit_price for l in order.lines}
        refund = Money.zero(order.currency)
        for line in rma.lines:
            if not line.accepted:
                continue
            unit = unit_prices.get(line.sku)
            if unit is None:
                continue
            refund = refund + unit * line.quantity
            if line.restock:
                self.inventory.restock(line.sku, line.quantity)

        capture_id = order.payload.get("payment", {}).get("capture_id", "unknown")
        self.refunds.refund(capture_id, refund)
        rma.refund_amount = refund
        self._transition(rma, RMAState.REFUNDED, refund.format())
        return refund

    # ---- helpers ---------------------------------------------------------
    def _require(self, rma_id: str) -> RMA:
        rma = self._rmas.get(rma_id)
        if rma is None:
            raise KeyError(f"unknown RMA {rma_id}")
        return rma

    def _validate_quantities(self, order: Order, lines: List[ReturnLine]) -> None:
        ordered = {l.sku: l.quantity for l in order.lines}
        for line in lines:
            if line.sku not in ordered:
                raise ValueError(f"{line.sku} was not on order {order.id}")
            if line.quantity <= 0 or line.quantity > ordered[line.sku]:
                raise ValueError(
                    f"cannot return {line.quantity} of {line.sku}; "
                    f"only {ordered[line.sku]} were purchased"
                )
