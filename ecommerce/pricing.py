"""Pricing engine: tiered quantity pricing, coupons, bundles, promotions.

The engine takes a set of cart lines and returns a fully itemised
:class:`Quote`: per-line subtotals, every discount that applied (with a human
reason), and the order total — all in integer minor units. Percentage
discounts multiply in integer space and round once; order-level discounts are
split back across lines with :meth:`Money.allocate` so the parts always sum
to the whole with no lost cent.

Coupon validation (expiry, usage cap, minimum order, product eligibility) is
gathered in :meth:`Coupon.validate` and is designed to be called *inside the
order transaction*: the usage-count check reads the same ``coupon_usage``
table the order write appends to, so two concurrent redemptions of a
single-use code cannot both pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence

from .db import Database
from .errors import (
    CouponExpired,
    CouponMinimumNotMet,
    CouponNotApplicable,
    CouponNotFound,
    CouponUsageExceeded,
)
from .money import Money


@dataclass
class LineInput:
    """A single cart line handed to the pricing engine."""

    sku: str
    unit_price: Money
    quantity: int
    product_id: str = ""
    category: str = ""


@dataclass
class Adjustment:
    """A discount (negative amount) applied to the order or a line."""

    code: str
    label: str
    amount: Money  # negative
    line_sku: Optional[str] = None  # None => order-level


@dataclass
class QuoteLine:
    sku: str
    quantity: int
    unit_price: Money
    subtotal: Money  # unit_price * quantity, before discounts
    line_discount: Money  # <= 0


@dataclass
class Quote:
    currency: str
    lines: List[QuoteLine]
    adjustments: List[Adjustment]
    subtotal: Money
    discount_total: Money  # <= 0
    total: Money  # subtotal + discount_total, floored at zero

    def line(self, sku: str) -> Optional[QuoteLine]:
        return next((l for l in self.lines if l.sku == sku), None)


# ---- discount value objects ---------------------------------------------
class DiscountType(str, Enum):
    PERCENT = "percent"
    FIXED = "fixed"


@dataclass
class TierRule:
    """Quantity-break pricing: at ``min_qty`` units, charge ``unit_price``."""

    min_qty: int
    unit_price: Money


@dataclass
class TieredPrice:
    sku: str
    tiers: List[TierRule]  # sorted ascending by min_qty at apply time

    def unit_price_for(self, quantity: int, default: Money) -> Money:
        chosen = default
        for tier in sorted(self.tiers, key=lambda t: t.min_qty):
            if quantity >= tier.min_qty:
                chosen = tier.unit_price
        return chosen


@dataclass
class Bundle:
    """Buy every SKU in ``skus`` (each at least once) -> flat ``price``."""

    id: str
    skus: Sequence[str]
    price: Money
    label: str = ""


@dataclass
class PromotionRule:
    """An automatic, code-free promotion evaluated on every quote.

    ``predicate`` decides whether the promo fires for a given list of lines;
    ``discount`` returns the (negative) order-level adjustment to apply.
    """

    id: str
    label: str
    predicate: Callable[[Sequence[LineInput]], bool]
    discount: Callable[[Sequence[LineInput], Money], Money]


@dataclass
class Coupon:
    code: str
    discount_type: DiscountType
    value: int  # percent points (e.g. 15) or fixed minor units
    currency: Optional[str] = None  # required for FIXED
    expires_at: Optional[float] = None  # epoch seconds
    usage_limit: Optional[int] = None  # max total redemptions
    per_customer_limit: Optional[int] = None
    min_order_minor: int = 0
    eligible_products: Optional[set] = None  # None => whole order
    eligible_categories: Optional[set] = None

    def eligible_lines(self, lines: Sequence[LineInput]) -> List[LineInput]:
        if self.eligible_products is None and self.eligible_categories is None:
            return list(lines)
        out = []
        for l in lines:
            if self.eligible_products and l.product_id in self.eligible_products:
                out.append(l)
            elif self.eligible_categories and l.category in self.eligible_categories:
                out.append(l)
        return out

    def validate(
        self,
        lines: Sequence[LineInput],
        subtotal: Money,
        *,
        now: float,
        redemptions: int,
    ) -> List[LineInput]:
        """Run all coupon checks atomically; return the eligible lines.

        Raises the specific :class:`CouponError` subclass for the first failed
        check. ``redemptions`` is the current usage count read from the same
        transaction that will record this redemption, so the limit check is
        race-free.
        """
        if self.expires_at is not None and now > self.expires_at:
            raise CouponExpired(self.code)
        if self.usage_limit is not None and redemptions >= self.usage_limit:
            raise CouponUsageExceeded(self.code)
        if subtotal.amount < self.min_order_minor:
            raise CouponMinimumNotMet(
                f"{self.code}: order {subtotal.amount} below minimum "
                f"{self.min_order_minor}"
            )
        eligible = self.eligible_lines(lines)
        if not eligible:
            raise CouponNotApplicable(f"{self.code}: no eligible items in cart")
        return eligible


class PricingEngine:
    def __init__(
        self,
        db: Optional[Database] = None,
        *,
        clock: Optional[Callable[[], float]] = None,
    ):
        self.db = db
        self._coupons: Dict[str, Coupon] = {}
        self._tiered: Dict[str, TieredPrice] = {}
        self._bundles: List[Bundle] = []
        self._promotions: List[PromotionRule] = []
        if clock is None:
            import time

            clock = time.time
        self._clock = clock

    # ---- registration ----------------------------------------------------
    def register_coupon(self, coupon: Coupon) -> None:
        self._coupons[coupon.code.upper()] = coupon

    def register_tiered_price(self, tiered: TieredPrice) -> None:
        self._tiered[tiered.sku] = tiered

    def register_bundle(self, bundle: Bundle) -> None:
        self._bundles.append(bundle)

    def register_promotion(self, rule: PromotionRule) -> None:
        self._promotions.append(rule)

    def coupon(self, code: str) -> Optional[Coupon]:
        return self._coupons.get(code.upper())

    # ---- quoting ---------------------------------------------------------
    def quote(
        self,
        lines: Sequence[LineInput],
        *,
        currency: str,
        coupon_code: Optional[str] = None,
        customer_redemptions: int = 0,
    ) -> Quote:
        if not lines:
            zero = Money.zero(currency)
            return Quote(currency, [], [], zero, zero, zero)

        # 1) Apply quantity-break (tiered) unit prices.
        priced: List[LineInput] = []
        for l in lines:
            tiered = self._tiered.get(l.sku)
            unit = tiered.unit_price_for(l.quantity, l.unit_price) if tiered else l.unit_price
            priced.append(
                LineInput(l.sku, unit, l.quantity, l.product_id, l.category)
            )

        quote_lines = [
            QuoteLine(
                l.sku,
                l.quantity,
                l.unit_price,
                l.unit_price * l.quantity,
                Money.zero(currency),
            )
            for l in priced
        ]
        subtotal = Money(sum(ql.subtotal.amount for ql in quote_lines), currency)
        adjustments: List[Adjustment] = []

        # 2) Bundle pricing: if all bundle SKUs are present, charge the flat
        #    bundle price for one set and credit the difference as a discount.
        for bundle in self._bundles:
            if bundle.price.currency != currency:
                continue
            sets = self._bundle_sets(bundle, quote_lines)
            if sets <= 0:
                continue
            component_cost = Money.zero(currency)
            for sku in bundle.skus:
                ql = next(q for q in quote_lines if q.sku == sku)
                component_cost = component_cost + ql.unit_price
            saving = (bundle.price - component_cost) * sets  # negative if cheaper
            if saving.amount < 0:
                adjustments.append(
                    Adjustment(
                        code=f"bundle:{bundle.id}",
                        label=bundle.label or f"Bundle {bundle.id}",
                        amount=saving,
                    )
                )

        # 3) Coupon (validated + applied).
        if coupon_code:
            coupon = self._coupons.get(coupon_code.upper())
            if coupon is None:
                raise CouponNotFound(coupon_code)
            redemptions = self._redemption_count(coupon.code)
            eligible = coupon.validate(
                priced,
                subtotal,
                now=self._clock(),
                redemptions=redemptions,
            )
            adjustments.append(self._coupon_adjustment(coupon, eligible, currency))

        # 4) Automatic promotions (code-free).
        for rule in self._promotions:
            if rule.predicate(priced):
                amount = rule.discount(priced, subtotal)
                if amount.amount < 0:
                    adjustments.append(
                        Adjustment(code=f"promo:{rule.id}", label=rule.label, amount=amount)
                    )

        # 5) Attribute order-level discounts back onto lines proportionally so
        #    every line carries its share (needed for tax and partial refunds).
        self._distribute(quote_lines, adjustments, currency)

        discount_total = Money(sum(a.amount.amount for a in adjustments), currency)
        total = (subtotal + discount_total).clamp_non_negative()
        return Quote(currency, quote_lines, adjustments, subtotal, discount_total, total)

    # ---- internals -------------------------------------------------------
    def _coupon_adjustment(
        self, coupon: Coupon, eligible: Sequence[LineInput], currency: str
    ) -> Adjustment:
        eligible_subtotal = Money(
            sum((l.unit_price * l.quantity).amount for l in eligible), currency
        )
        if coupon.discount_type is DiscountType.PERCENT:
            amount = eligible_subtotal.apply_rate(coupon.value, 100)
            amount = Money(-amount.amount, currency)
        else:
            fixed = min(coupon.value, eligible_subtotal.amount)
            amount = Money(-fixed, currency)
        return Adjustment(
            code=coupon.code,
            label=f"Coupon {coupon.code}",
            amount=amount,
        )

    def _distribute(
        self,
        lines: List[QuoteLine],
        adjustments: List[Adjustment],
        currency: str,
    ) -> None:
        order_level = [a for a in adjustments if a.line_sku is None]
        if not order_level or not lines:
            return
        weights = [max(0, ql.subtotal.amount) for ql in lines]
        if sum(weights) == 0:
            return
        for adj in order_level:
            shares = adj.amount.allocate(weights)
            for ql, share in zip(lines, shares):
                ql.line_discount = ql.line_discount + share

    def _bundle_sets(self, bundle: Bundle, lines: List[QuoteLine]) -> int:
        present = {ql.sku: ql.quantity for ql in lines}
        if any(sku not in present for sku in bundle.skus):
            return 0
        return min(present[sku] for sku in bundle.skus)

    def _redemption_count(self, code: str) -> int:
        if self.db is None:
            return 0
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM coupon_usage WHERE code = ?", (code,)
            ).fetchone()
        return row["n"]

    def record_redemption(self, code: str, order_id: str) -> None:
        if self.db is None:
            return
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO coupon_usage (code, order_id) VALUES (?, ?)",
                (code, order_id),
            )
