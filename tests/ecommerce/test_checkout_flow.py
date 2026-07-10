"""Verification: the complete checkout flow, cart -> payment -> order.

Exercises each supported payment method, the price-change guard between cart
and checkout, out-of-stock abort with hold release, and the multi-step
progression with address/shipping/tax.
"""

import pytest

from ecommerce import (
    Address,
    CartService,
    CheckoutService,
    FakeGateway,
    Money,
    OrderService,
    OrderState,
    PricingEngine,
    ShippingCalculator,
    ShippingMethod,
    TaxEngine,
)
from ecommerce.errors import OutOfStock, PaymentDeclined, PriceChanged


def build(db, catalog, inventory, *, decline=()):
    carts = CartService(db, catalog, inventory)
    pricing = PricingEngine(db)
    tax = TaxEngine()
    tax.set_rate("US", 725, region="CA", name="CA")
    shipping = ShippingCalculator()
    shipping.add_method(
        ShippingMethod(
            "standard", "Standard", Money(599, "USD"), Money(100, "USD"), eta_days=5,
            free_over=Money(20000, "USD"),
        )
    )
    shipping.add_method(
        ShippingMethod("express", "Express", Money(1499, "USD"), Money(300, "USD"), eta_days=2)
    )
    orders = OrderService(db)
    gateway = FakeGateway(decline_amounts=decline)
    checkout = CheckoutService(
        catalog=catalog, carts=carts, inventory=inventory, pricing=pricing,
        tax=tax, shipping=shipping, orders=orders, gateway=gateway,
        weigh=lambda cart: 400 * cart.item_count,
    )
    return carts, checkout, orders, pricing


CA = Address("Sam Fixit", "1 Market St", "San Francisco", "US", "94105", region="CA")


@pytest.mark.parametrize("method", ["card", "wallet", "cod"])
def test_full_checkout_each_payment_method(db, catalog, inventory, method):
    carts, checkout, orders, _ = build(db, catalog, inventory)
    cart = carts.create(owner="buyer", currency="USD")
    carts.add_item(cart.id, "BATT14", 2)  # 2 * 3900 = 7800

    state = checkout.begin(cart.id)
    checkout.set_address(state, CA)
    options = checkout.shipping_options(state)
    assert {o.method_id for o in options} == {"standard", "express"}
    checkout.choose_shipping(state, "standard")
    summary = checkout.compute_totals(state)

    # subtotal 7800, shipping 599 + (1kg *? ) weight=800g->1kg: 599+100=699
    assert summary.subtotal.amount == 7800
    assert summary.shipping.amount == 699
    # tax on discounted lines (7800) + shipping (699) at 7.25%
    expected_tax = Money(7800, "USD").apply_rate(725, 10000).amount + Money(699, "USD").apply_rate(725, 10000).amount
    assert summary.tax.amount == expected_tax
    assert summary.total.amount == 7800 + 699 + expected_tax

    order = checkout.place_order(state, payment_method=method, owner="buyer")
    assert order.state is OrderState.PAYMENT_CAPTURED
    assert order.total.amount == summary.total.amount
    # Stock was actually decremented (hold committed).
    assert inventory.level("BATT14").on_hand == 48


def test_unsupported_payment_method_declined(db, catalog, inventory):
    carts, checkout, _, _ = build(db, catalog, inventory)
    cart = carts.create(currency="USD")
    carts.add_item(cart.id, "BATT14", 1)
    state = checkout.begin(cart.id)
    checkout.set_address(state, CA)
    checkout.choose_shipping(state, "standard")
    checkout.compute_totals(state)
    with pytest.raises(PaymentDeclined):
        checkout.place_order(state, payment_method="bitcoin", owner=None)
    # No stock was consumed; holds were released.
    assert inventory.available("BATT14") == 50


def test_declined_payment_releases_holds(db, catalog, inventory):
    carts, checkout, _, _ = build(db, catalog, inventory)
    cart = carts.create(currency="USD")
    carts.add_item(cart.id, "BATT14", 1)
    state = checkout.begin(cart.id)
    checkout.set_address(state, CA)
    checkout.choose_shipping(state, "standard")
    summary = checkout.compute_totals(state)
    # Force the gateway to decline this exact total.
    carts2, checkout2, _, _ = build(db, catalog, inventory, decline=(summary.total.amount,))
    state2 = checkout2.begin(cart.id)
    checkout2.set_address(state2, CA)
    checkout2.choose_shipping(state2, "standard")
    checkout2.compute_totals(state2)
    with pytest.raises(PaymentDeclined):
        checkout2.place_order(state2, payment_method="card", owner=None)
    assert inventory.available("BATT14") == 50


def test_out_of_stock_aborts_before_charge(db, catalog, inventory):
    carts, checkout, _, _ = build(db, catalog, inventory)
    inventory.set_stock("BATT14", 1)
    cart = carts.create(currency="USD")
    carts.add_item(cart.id, "BATT14", 1)
    state = checkout.begin(cart.id)
    checkout.set_address(state, CA)
    checkout.choose_shipping(state, "standard")
    checkout.compute_totals(state)
    # Someone else buys the last unit between review and place.
    inventory.decrement("BATT14", 1)
    with pytest.raises(OutOfStock):
        checkout.place_order(state, payment_method="card", owner=None)


def test_price_change_between_cart_and_checkout_is_caught(db, catalog, inventory):
    carts, checkout, _, pricing = build(db, catalog, inventory)
    cart = carts.create(currency="USD")
    carts.add_item(cart.id, "BATT14", 4)
    state = checkout.begin(cart.id)
    checkout.set_address(state, CA)
    checkout.choose_shipping(state, "standard")
    checkout.compute_totals(state)
    # A tiered price kicks in AFTER the shopper reviewed -> price drops.
    from ecommerce import TierRule, TieredPrice

    pricing.register_tiered_price(
        TieredPrice("BATT14", [TierRule(4, Money(2900, "USD"))])
    )
    with pytest.raises(PriceChanged) as exc:
        checkout.place_order(state, payment_method="card", owner=None)
    assert exc.value.sku == "BATT14"


def test_free_shipping_over_threshold(db, catalog, inventory):
    carts, checkout, _, _ = build(db, catalog, inventory)
    cart = carts.create(currency="USD")
    carts.add_item(cart.id, "OLED14-BLK", 2)  # 2 * 12900 = 25800 > 20000
    state = checkout.begin(cart.id)
    checkout.set_address(state, CA)
    checkout.choose_shipping(state, "standard")
    summary = checkout.compute_totals(state)
    assert summary.shipping.amount == 0
