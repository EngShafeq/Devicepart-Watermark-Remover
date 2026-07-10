"""End-to-end walkthrough of the e-commerce backend.

Run it directly to watch a shopper move from browsing to a placed order,
through a return, with every money figure computed in integer minor units:

    python -m examples.ecommerce_demo   # or: python examples/ecommerce_demo.py

It wires the modules together the way a real application would and prints
each stage, so it doubles as a smoke test of the whole pipeline.
"""

from __future__ import annotations

import os
import sys

# Allow running as a bare script (python examples/ecommerce_demo.py) as well
# as a module (python -m examples.ecommerce_demo).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecommerce import (
    Address,
    AnalyticsService,
    CartService,
    Catalog,
    Category,
    CheckoutService,
    Coupon,
    Database,
    DiscountType,
    FakeGateway,
    Inventory,
    Money,
    NotificationService,
    OrderService,
    OrderState,
    PriceTier,
    PricingEngine,
    Product,
    RecordingTransport,
    ReturnLine,
    ReturnsService,
    SearchIndex,
    ShippingCalculator,
    ShippingMethod,
    TaxEngine,
    Variant,
)
from ecommerce.returns import RefundGateway


def build_catalog() -> Catalog:
    cat = Catalog()
    cat.add_category(Category("root", "All Parts"))
    cat.add_category(Category("screens", "Screens", parent="root"))
    cat.add_category(Category("batteries", "Batteries", parent="root"))

    screen = Product(
        "p-oled", "iPhone 14 OLED Screen Assembly", "screens",
        description="Genuine-grade OLED display with digitizer",
        attributes={"brand": "DevicePart", "grade": "premium"},
        tags=["screen", "oled", "display"], popularity=120,
    )
    sv = Variant("OLED14")
    sv.set_price(PriceTier.RETAIL, Money.from_major("129.00", "USD"))
    sv.set_price(PriceTier.MEMBER, Money.from_major("119.00", "USD"))
    screen.variants.append(sv)
    cat.add_product(screen)

    battery = Product(
        "p-batt", "iPhone 14 Replacement Battery", "batteries",
        description="High-capacity lithium battery",
        attributes={"brand": "DevicePart", "grade": "standard"},
        tags=["battery", "power"], popularity=80,
    )
    bv = Variant("BATT14")
    bv.set_price(PriceTier.RETAIL, Money.from_major("39.00", "USD"))
    battery.variants.append(bv)
    cat.add_product(battery)
    return cat


def main() -> None:
    db = Database(":memory:")
    catalog = build_catalog()

    inventory = Inventory(db)
    inventory.set_stock("OLED14", 5)
    inventory.set_stock("BATT14", 20)

    carts = CartService(db, catalog, inventory)
    pricing = PricingEngine(db)
    pricing.register_coupon(Coupon("WELCOME10", DiscountType.PERCENT, 10, min_order_minor=5000))

    tax = TaxEngine()
    tax.set_rate("US", 725, region="CA", name="CA sales tax")

    shipping = ShippingCalculator()
    shipping.add_method(
        ShippingMethod("standard", "Standard (5 days)", Money(599, "USD"),
                       Money(100, "USD"), eta_days=5, free_over=Money(20000, "USD"))
    )
    shipping.add_method(
        ShippingMethod("express", "Express (2 days)", Money(1499, "USD"),
                       Money(300, "USD"), eta_days=2)
    )

    orders = OrderService(db)
    gateway = FakeGateway()
    analytics = AnalyticsService(db)
    notify = NotificationService(RecordingTransport())

    checkout = CheckoutService(
        catalog=catalog, carts=carts, inventory=inventory, pricing=pricing,
        tax=tax, shipping=shipping, orders=orders, gateway=gateway,
        weigh=lambda cart: 400 * cart.item_count,
    )

    print("=" * 64)
    print("1. SEARCH")
    index = SearchIndex(catalog, synonyms={"display": ["screen", "display"]},
                        facet_attributes=["brand", "grade"])
    results = index.search("oled disply")  # note the typo
    print(f"   query 'oled disply' -> corrected {results.corrected_terms}")
    for hit in results.hits:
        print(f"   {hit.product_id}  score={hit.score}")

    analytics.product_view("sess-1", "p-oled")

    print("\n2. CART (anonymous, then merged on login)")
    anon = carts.create(currency="USD")
    carts.add_item(anon.id, "OLED14", 1)
    carts.add_item(anon.id, "BATT14", 2)
    analytics.add_to_cart("sess-1", "OLED14", 1)
    user_cart = carts.create(owner="dealer-42", currency="USD")
    carts.add_item(user_cart.id, "BATT14", 1)
    merged = carts.merge(anon.id, user_cart.id)
    print("   merged cart:")
    for item in merged.items:
        print(f"     {item.sku} x{item.quantity} @ {item.unit_price.format()}")
    print(f"   subtotal {merged.subtotal.format()}")

    print("\n3. CHECKOUT")
    state = checkout.begin(merged.id, coupon_code="WELCOME10")
    analytics.checkout_step("sess-1", "address")
    checkout.set_address(state, Address("Sam Fixit", "1 Market St", "San Francisco",
                                        "US", "94105", region="CA"))
    for opt in checkout.shipping_options(state):
        print(f"   ship option: {opt.label} -> {opt.amount.format()}")
    checkout.choose_shipping(state, "standard")
    summary = checkout.compute_totals(state)
    print(f"   subtotal  {summary.subtotal.format()}")
    print(f"   discount  {summary.discount.format()}")
    print(f"   shipping  {summary.shipping.format()}")
    print(f"   tax       {summary.tax.format()}")
    print(f"   TOTAL     {summary.total.format()}")

    order = checkout.place_order(state, payment_method="card", owner="dealer-42")
    analytics.purchase("sess-1", order.id, order.total.amount)
    notify.order_confirmed("dealer@example.com", name="Sam", order_id=order.id[:8],
                           total=order.total.format())
    print(f"   order {order.id[:8]} placed, state={order.state.value}")
    print(f"   OLED14 stock now {inventory.level('OLED14').on_hand}")

    print("\n4. FULFILLMENT (state machine)")
    for target in (OrderState.FULFILLMENT_PROCESSING, OrderState.SHIPPED, OrderState.DELIVERED):
        order = orders.transition(order.id, target)
        print(f"   -> {order.state.value}")
    notify.order_shipped("dealer@example.com", name="Sam", order_id=order.id[:8], tracking="TRK123")

    print("\n5. RETURN (RMA -> refund -> restock)")
    returns = ReturnsService(orders, inventory, RefundGateway())
    rma = returns.open_rma(order.id, [ReturnLine("BATT14", 1, reason="changed mind")])
    returns.issue_label(rma.id)
    returns.mark_in_transit(rma.id)
    returns.begin_inspection(rma.id)
    returns.record_inspection(rma.id, {"BATT14": True})
    refund = returns.settle(rma.id)
    print(f"   refunded {refund.format()}, RMA state={returns.get(rma.id).state.value}")
    print(f"   BATT14 stock restocked to {inventory.level('BATT14').on_hand}")

    print("\n6. ANALYTICS FUNNEL")
    for stage in analytics.funnel():
        print(f"   {stage.name:<16} sessions={stage.sessions} "
              f"conv={stage.conversion_from_previous}")
    print("=" * 64)


if __name__ == "__main__":
    main()
