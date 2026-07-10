"""Returns/RMA, notification pipeline, and analytics funnel coverage."""

from ecommerce import (
    AnalyticsService,
    Database,
    Inventory,
    Money,
    NotificationService,
    OrderLine,
    OrderService,
    OrderState,
    RecordingTransport,
    ReturnLine,
    ReturnsService,
)
from ecommerce.returns import RefundGateway, RMAState


def delivered_order(db):
    inv = Inventory(db)
    inv.set_stock("BATT14", 10)
    inv.decrement("BATT14", 2)  # 2 sold -> 8 on hand
    orders = OrderService(db)
    order = orders.create(
        owner="u",
        currency="USD",
        lines=[OrderLine("BATT14", 2, Money(3900, "USD"), Money(7800, "USD"))],
        total=Money(7800, "USD"),
        payload={"payment": {"capture_id": "cap_1"}},
    )
    for s in (
        OrderState.PAYMENT_AUTHORIZED,
        OrderState.PAYMENT_CAPTURED,
        OrderState.FULFILLMENT_PROCESSING,
        OrderState.SHIPPED,
        OrderState.DELIVERED,
    ):
        orders.transition(order.id, s)
    return inv, orders, orders.get(order.id)


def test_return_full_lifecycle_refund_and_restock(db):
    inv, orders, order = delivered_order(db)
    returns = ReturnsService(orders, inv, RefundGateway())

    rma = returns.open_rma(order.id, [ReturnLine("BATT14", 2, reason="defective")])
    assert rma.state is RMAState.REQUESTED
    tracking = returns.issue_label(rma.id)
    assert tracking
    returns.mark_in_transit(rma.id)
    returns.begin_inspection(rma.id)
    returns.record_inspection(rma.id, {"BATT14": True})
    refund = returns.settle(rma.id)

    assert refund.amount == 7800  # 2 * 3900
    assert returns.get(rma.id).state is RMAState.REFUNDED
    # Accepted, restockable units returned to inventory: 8 -> 10.
    assert inv.level("BATT14").on_hand == 10


def test_rejected_inspection_no_restock_no_refund(db):
    inv, orders, order = delivered_order(db)
    returns = ReturnsService(orders, inv, RefundGateway())
    rma = returns.open_rma(order.id, [ReturnLine("BATT14", 2)])
    returns.issue_label(rma.id)
    returns.mark_in_transit(rma.id)
    returns.begin_inspection(rma.id)
    returns.record_inspection(rma.id, {"BATT14": False})  # failed inspection
    refund = returns.settle(rma.id)
    assert refund.amount == 0
    assert inv.level("BATT14").on_hand == 8  # not restocked


def test_cannot_return_more_than_purchased(db):
    inv, orders, order = delivered_order(db)
    returns = ReturnsService(orders, inv, RefundGateway())
    try:
        returns.open_rma(order.id, [ReturnLine("BATT14", 5)])
        assert False, "should reject over-quantity return"
    except ValueError:
        pass


def test_notification_templating_and_tracking():
    transport = RecordingTransport()
    notify = NotificationService(transport)
    msg = notify.order_confirmed("a@b.com", name="Sam", order_id="o1", total="$78.00")
    assert msg.status.value == "sent"
    assert "o1" in msg.subject
    assert "Sam" in msg.body and "$78.00" in msg.body
    assert transport.sent[0].to == "a@b.com"


def test_notification_failure_marks_failed_after_retries():
    transport = RecordingTransport(fail_templates=("shipping_update",))
    notify = NotificationService(transport, max_retries=1)
    msg = notify.order_shipped("a@b.com", name="Sam", order_id="o1", tracking="TRK1")
    assert msg.status.value == "failed"
    assert msg.error
    assert notify.delivery_stats()["failed"] == 1


def test_analytics_funnel_conversion():
    db = Database(":memory:")
    analytics = AnalyticsService(db)
    # Session 1 completes the whole funnel; session 2 abandons at cart.
    analytics.product_view("s1", "p-batt-14")
    analytics.add_to_cart("s1", "BATT14", 1)
    analytics.checkout_step("s1", "address")
    analytics.purchase("s1", "o1", 7800)
    analytics.product_view("s2", "p-batt-14")
    analytics.add_to_cart("s2", "BATT14", 1)

    funnel = {s.name: s for s in analytics.funnel()}
    assert funnel["product_view"].sessions == 2
    assert funnel["add_to_cart"].sessions == 2
    assert funnel["checkout_step"].sessions == 1
    assert funnel["purchase"].sessions == 1
    # 50% of add-to-cart sessions reached checkout.
    assert funnel["checkout_step"].conversion_from_previous == 0.5
