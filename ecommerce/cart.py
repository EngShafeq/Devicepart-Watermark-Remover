"""Shopping cart with server-side persistence and anonymous->user merging.

Carts live in SQLite, so a cart survives browser closure, a device switch
(for a signed-in user), and a server restart — the state is on disk, keyed by
cart id, not held in a web session. Each mutation bumps a ``version`` column
so a stale client write can be detected.

Two behaviours matter for correctness:

* **Merge on login.** When an anonymous shopper authenticates, their cart is
  folded into whatever cart the account already had. Quantities for a shared
  SKU are *summed* (then clamped to available stock), not duplicated into two
  lines — see :meth:`CartService.merge`.
* **Self-healing.** :meth:`CartService.reconcile` drops discontinued variants
  and clamps any line whose quantity now exceeds available inventory, so the
  cart shown to the shopper is always purchasable.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .catalog import Catalog, PriceTier
from .db import Database
from .inventory import Inventory
from .money import Money


@dataclass
class CartItem:
    sku: str
    quantity: int
    unit_price: Money


@dataclass
class Cart:
    id: str
    owner: Optional[str]
    currency: str
    version: int
    items: List[CartItem] = field(default_factory=list)

    @property
    def subtotal(self) -> Money:
        total = sum((i.unit_price * i.quantity).amount for i in self.items)
        return Money(total, self.currency)

    def item(self, sku: str) -> Optional[CartItem]:
        return next((i for i in self.items if i.sku == sku), None)

    @property
    def item_count(self) -> int:
        return sum(i.quantity for i in self.items)


@dataclass
class ReconcileReport:
    """What :meth:`CartService.reconcile` changed, for surfacing to the shopper."""

    removed_discontinued: List[str] = field(default_factory=list)
    clamped: Dict[str, int] = field(default_factory=dict)  # sku -> new quantity
    removed_out_of_stock: List[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.removed_discontinued or self.clamped or self.removed_out_of_stock)


class CartService:
    def __init__(
        self,
        db: Database,
        catalog: Catalog,
        inventory: Inventory,
        *,
        clock=None,
        tier: PriceTier = PriceTier.RETAIL,
    ):
        self.db = db
        self.catalog = catalog
        self.inventory = inventory
        self.tier = tier
        if clock is None:
            import time

            clock = time.time
        self._clock = clock

    # ---- lifecycle -------------------------------------------------------
    def create(self, *, owner: Optional[str] = None, currency: str = "USD") -> Cart:
        cart_id = uuid.uuid4().hex
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO carts (id, owner, currency, version, updated_at) VALUES (?, ?, ?, 1, ?)",
                (cart_id, owner, currency, self._clock()),
            )
        return Cart(cart_id, owner, currency, 1, [])

    def get(self, cart_id: str) -> Optional[Cart]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM carts WHERE id = ?", (cart_id,)).fetchone()
            if row is None:
                return None
            items = conn.execute(
                "SELECT sku, quantity, unit_price FROM cart_items WHERE cart_id = ?",
                (cart_id,),
            ).fetchall()
        cart = Cart(row["id"], row["owner"], row["currency"], row["version"])
        cart.items = [
            CartItem(i["sku"], i["quantity"], Money(i["unit_price"], row["currency"]))
            for i in items
        ]
        return cart

    def get_or_create_for_user(self, owner: str, currency: str = "USD") -> Cart:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT id FROM carts WHERE owner = ? ORDER BY updated_at DESC LIMIT 1",
                (owner,),
            ).fetchone()
        if row:
            cart = self.get(row["id"])
            if cart:
                return cart
        return self.create(owner=owner, currency=currency)

    # ---- mutations -------------------------------------------------------
    def add_item(self, cart_id: str, sku: str, quantity: int) -> Cart:
        """Add ``quantity`` of ``sku``, validating against available stock.

        The quantity is capped at what inventory can currently supply, so the
        cart never promises more than exists. Adding an already-present SKU
        increases that one line rather than creating a duplicate.
        """
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        resolved = self.catalog.resolve(sku)
        if resolved is None or not self.catalog.is_sellable(sku):
            raise KeyError(f"{sku} is not a sellable variant")
        _, variant = resolved
        cart = self.get(cart_id)
        if cart is None:
            raise KeyError(f"unknown cart {cart_id}")
        unit_price = variant.price(self.tier, cart.currency)
        existing = cart.item(sku)
        desired = (existing.quantity if existing else 0) + quantity
        available = self.inventory.available(sku)
        final_qty = min(desired, available) if available >= 0 else desired
        if final_qty <= 0:
            raise KeyError(f"{sku} is out of stock")
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO cart_items (cart_id, sku, quantity, unit_price)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cart_id, sku) DO UPDATE SET
                    quantity = excluded.quantity, unit_price = excluded.unit_price
                """,
                (cart_id, sku, final_qty, unit_price.amount),
            )
            self._touch(conn, cart_id)
        return self.get(cart_id)  # type: ignore[return-value]

    def set_quantity(self, cart_id: str, sku: str, quantity: int) -> Cart:
        if quantity < 0:
            raise ValueError("quantity cannot be negative")
        if quantity == 0:
            return self.remove_item(cart_id, sku)
        available = self.inventory.available(sku)
        final_qty = min(quantity, available) if available >= 0 else quantity
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE cart_items SET quantity = ? WHERE cart_id = ? AND sku = ?",
                (final_qty, cart_id, sku),
            )
            self._touch(conn, cart_id)
        return self.get(cart_id)  # type: ignore[return-value]

    def remove_item(self, cart_id: str, sku: str) -> Cart:
        with self.db.transaction() as conn:
            conn.execute(
                "DELETE FROM cart_items WHERE cart_id = ? AND sku = ?", (cart_id, sku)
            )
            self._touch(conn, cart_id)
        return self.get(cart_id)  # type: ignore[return-value]

    # ---- merge & reconcile ----------------------------------------------
    def merge(self, anonymous_cart_id: str, user_cart_id: str) -> Cart:
        """Fold an anonymous cart into the user's cart, summing shared SKUs.

        Runs in a single transaction. For each SKU the merged quantity is the
        sum of both carts, clamped to available stock so the merge can never
        create an unfulfillable line. The anonymous cart is emptied and
        deleted so it can't be merged twice.
        """
        anon = self.get(anonymous_cart_id)
        user = self.get(user_cart_id)
        if anon is None or user is None:
            raise KeyError("both carts must exist to merge")
        merged: Dict[str, int] = {i.sku: i.quantity for i in user.items}
        prices: Dict[str, int] = {i.sku: i.unit_price.amount for i in user.items}
        for item in anon.items:
            merged[item.sku] = merged.get(item.sku, 0) + item.quantity
            prices.setdefault(item.sku, item.unit_price.amount)
        with self.db.transaction() as conn:
            for sku, qty in merged.items():
                if not self.catalog.is_sellable(sku):
                    continue  # drop discontinued SKUs during the merge
                available = self.inventory.available(sku)
                final_qty = min(qty, available) if available >= 0 else qty
                if final_qty <= 0:
                    continue
                conn.execute(
                    """
                    INSERT INTO cart_items (cart_id, sku, quantity, unit_price)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(cart_id, sku) DO UPDATE SET quantity = excluded.quantity
                    """,
                    (user_cart_id, sku, final_qty, prices[sku]),
                )
            conn.execute("DELETE FROM cart_items WHERE cart_id = ?", (anonymous_cart_id,))
            conn.execute("DELETE FROM carts WHERE id = ?", (anonymous_cart_id,))
            self._touch(conn, user_cart_id)
        return self.get(user_cart_id)  # type: ignore[return-value]

    def reconcile(self, cart_id: str) -> tuple[Cart, ReconcileReport]:
        """Drop discontinued items and clamp lines to available stock."""
        cart = self.get(cart_id)
        if cart is None:
            raise KeyError(f"unknown cart {cart_id}")
        report = ReconcileReport()
        with self.db.transaction() as conn:
            for item in cart.items:
                if not self.catalog.is_sellable(item.sku):
                    conn.execute(
                        "DELETE FROM cart_items WHERE cart_id = ? AND sku = ?",
                        (cart_id, item.sku),
                    )
                    report.removed_discontinued.append(item.sku)
                    continue
                available = self.inventory.available(item.sku)
                if available <= 0:
                    conn.execute(
                        "DELETE FROM cart_items WHERE cart_id = ? AND sku = ?",
                        (cart_id, item.sku),
                    )
                    report.removed_out_of_stock.append(item.sku)
                elif item.quantity > available:
                    conn.execute(
                        "UPDATE cart_items SET quantity = ? WHERE cart_id = ? AND sku = ?",
                        (available, cart_id, item.sku),
                    )
                    report.clamped[item.sku] = available
            if report.changed:
                self._touch(conn, cart_id)
        return self.get(cart_id), report  # type: ignore[return-value]

    def _touch(self, conn, cart_id: str) -> None:
        conn.execute(
            "UPDATE carts SET version = version + 1, updated_at = ? WHERE id = ?",
            (self._clock(), cart_id),
        )
