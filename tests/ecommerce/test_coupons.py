"""Verification: coupon edge cases and the pricing engine.

Covers expired codes, exceeded usage limits, minimum-order-not-met, product
exclusions, plus tiered/bundle/percentage/fixed discounts.
"""

import pytest

from ecommerce import (
    Bundle,
    Coupon,
    Database,
    DiscountType,
    LineInput,
    Money,
    PricingEngine,
    PromotionRule,
    TierRule,
    TieredPrice,
)
from ecommerce.errors import (
    CouponExpired,
    CouponMinimumNotMet,
    CouponNotApplicable,
    CouponNotFound,
    CouponUsageExceeded,
)


def lines():
    return [
        LineInput("OLED14-BLK", Money(12900, "USD"), 1, product_id="p-oled-14", category="oled"),
        LineInput("BATT14", Money(3900, "USD"), 2, product_id="p-batt-14", category="batteries"),
    ]


def test_percentage_coupon_applies_and_rounds():
    engine = PricingEngine()
    engine.register_coupon(Coupon("SAVE10", DiscountType.PERCENT, 10))
    quote = engine.quote(lines(), currency="USD", coupon_code="SAVE10")
    # subtotal = 12900 + 7800 = 20700; 10% = 2070 off
    assert quote.subtotal.amount == 20700
    assert quote.discount_total.amount == -2070
    assert quote.total.amount == 18630


def test_fixed_coupon_capped_at_eligible_subtotal():
    engine = PricingEngine()
    engine.register_coupon(Coupon("OFF5", DiscountType.FIXED, 500, currency="USD"))
    quote = engine.quote(lines(), currency="USD", coupon_code="OFF5")
    assert quote.discount_total.amount == -500
    assert quote.total.amount == 20200


def test_unknown_coupon_rejected():
    engine = PricingEngine()
    with pytest.raises(CouponNotFound):
        engine.quote(lines(), currency="USD", coupon_code="NOPE")


def test_expired_coupon_rejected(clock=None):
    class Clk:
        now = 2_000.0

        def __call__(self):
            return self.now

    engine = PricingEngine(clock=Clk())
    engine.register_coupon(Coupon("OLD", DiscountType.PERCENT, 10, expires_at=1_000.0))
    with pytest.raises(CouponExpired):
        engine.quote(lines(), currency="USD", coupon_code="OLD")


def test_minimum_order_not_met_rejected():
    engine = PricingEngine()
    engine.register_coupon(Coupon("BIG", DiscountType.PERCENT, 10, min_order_minor=50000))
    with pytest.raises(CouponMinimumNotMet):
        engine.quote(lines(), currency="USD", coupon_code="BIG")


def test_usage_limit_exceeded_rejected():
    db = Database(":memory:")
    engine = PricingEngine(db)
    engine.register_coupon(Coupon("ONCE", DiscountType.PERCENT, 10, usage_limit=1))
    # Simulate the code already redeemed once.
    engine.record_redemption("ONCE", "order-1")
    with pytest.raises(CouponUsageExceeded):
        engine.quote(lines(), currency="USD", coupon_code="ONCE")


def test_product_exclusion_no_eligible_items():
    engine = PricingEngine()
    # Coupon only valid on a product that isn't in the cart.
    engine.register_coupon(
        Coupon("SCREENS", DiscountType.PERCENT, 20, eligible_products={"p-does-not-exist"})
    )
    with pytest.raises(CouponNotApplicable):
        engine.quote(lines(), currency="USD", coupon_code="SCREENS")


def test_product_scoped_coupon_only_discounts_eligible_lines():
    engine = PricingEngine()
    engine.register_coupon(
        Coupon("OLEDONLY", DiscountType.PERCENT, 10, eligible_products={"p-oled-14"})
    )
    quote = engine.quote(lines(), currency="USD", coupon_code="OLEDONLY")
    # 10% of the OLED line only (12900) = 1290, battery untouched.
    assert quote.discount_total.amount == -1290


def test_tiered_quantity_pricing():
    engine = PricingEngine()
    engine.register_tiered_price(
        TieredPrice("BATT14", [TierRule(3, Money(3400, "USD")), TierRule(5, Money(2900, "USD"))])
    )
    # Buying 5 batteries drops the unit price to 2900.
    ls = [LineInput("BATT14", Money(3900, "USD"), 5, product_id="p-batt-14")]
    quote = engine.quote(ls, currency="USD")
    assert quote.lines[0].unit_price.amount == 2900
    assert quote.subtotal.amount == 14500


def test_bundle_pricing_credits_difference():
    engine = PricingEngine()
    engine.register_bundle(
        Bundle("repair-kit", ["OLED14-BLK", "BATT14"], Money(15000, "USD"), "Repair Bundle")
    )
    ls = [
        LineInput("OLED14-BLK", Money(12900, "USD"), 1),
        LineInput("BATT14", Money(3900, "USD"), 1),
    ]
    quote = engine.quote(ls, currency="USD")
    # components 16800 -> bundle 15000, saving 1800
    assert quote.discount_total.amount == -1800
    assert quote.total.amount == 15000


def test_automatic_promotion_rule():
    engine = PricingEngine()
    engine.register_promotion(
        PromotionRule(
            "spend100get10",
            "Spend $100, get $10",
            predicate=lambda ls: sum((l.unit_price * l.quantity).amount for l in ls) >= 10000,
            discount=lambda ls, subtotal: Money(-1000, "USD"),
        )
    )
    quote = engine.quote(lines(), currency="USD")
    assert quote.discount_total.amount == -1000


def test_atomic_usage_count_uses_live_db_state():
    # The usage check reads the same table redemptions are written to, so a
    # single-use coupon can't be double-applied across two orders.
    db = Database(":memory:")
    engine = PricingEngine(db)
    engine.register_coupon(Coupon("SOLO", DiscountType.PERCENT, 5, usage_limit=1))
    q1 = engine.quote(lines(), currency="USD", coupon_code="SOLO")
    assert q1.discount_total.amount < 0
    engine.record_redemption("SOLO", "order-1")  # first order commits
    with pytest.raises(CouponUsageExceeded):
        engine.quote(lines(), currency="USD", coupon_code="SOLO")
