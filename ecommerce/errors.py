"""Typed errors for the e-commerce domain.

Each error carries enough context for an API layer to translate it into a
clear customer- or operator-facing message. Grouping them here keeps the
control flow in the domain modules explicit about what can go wrong.
"""

from __future__ import annotations


class EcommerceError(Exception):
    """Base class for all domain errors."""


# ---- inventory -----------------------------------------------------------
class OutOfStock(EcommerceError):
    """Not enough on-hand, uncommitted stock to satisfy the request."""

    def __init__(self, sku: str, requested: int, available: int):
        self.sku = sku
        self.requested = requested
        self.available = available
        super().__init__(
            f"{sku}: requested {requested} but only {available} available"
        )


class ConcurrencyConflict(EcommerceError):
    """An optimistic version check failed; the row was modified concurrently."""

    def __init__(self, entity: str, key: str):
        self.entity = entity
        self.key = key
        super().__init__(f"{entity} {key} was modified concurrently; retry")


class ReservationExpired(EcommerceError):
    """A soft hold was released (its TTL elapsed) before it was committed."""


# ---- cart / catalog ------------------------------------------------------
class ItemDiscontinued(EcommerceError):
    """A variant is no longer sellable and was removed from the cart."""


class PriceChanged(EcommerceError):
    """A line's price moved between two snapshots (e.g. cart -> checkout)."""

    def __init__(self, sku: str, was, now):
        self.sku = sku
        self.was = was
        self.now = now
        super().__init__(f"{sku}: price changed from {was} to {now}")


# ---- pricing / coupons ---------------------------------------------------
class CouponError(EcommerceError):
    """Base class for coupon validation failures."""


class CouponNotFound(CouponError):
    pass


class CouponExpired(CouponError):
    pass


class CouponUsageExceeded(CouponError):
    pass


class CouponMinimumNotMet(CouponError):
    pass


class CouponNotApplicable(CouponError):
    """No eligible line in the cart for this coupon's product scope."""


# ---- orders --------------------------------------------------------------
class IllegalTransition(EcommerceError):
    """An order state transition is not permitted by the state machine."""

    def __init__(self, frm: str, to: str):
        self.frm = frm
        self.to = to
        super().__init__(f"illegal order transition: {frm} -> {to}")


# ---- checkout ------------------------------------------------------------
class AddressInvalid(EcommerceError):
    pass


class ShippingUnavailable(EcommerceError):
    pass


class PaymentDeclined(EcommerceError):
    pass
