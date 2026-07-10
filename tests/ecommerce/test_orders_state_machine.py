"""Verification: order state transitions reject invalid paths and log them."""

import pytest

from ecommerce import Money, OrderLine, OrderService, OrderState
from ecommerce.errors import IllegalTransition


def make_order(db):
    svc = OrderService(db)
    order = svc.create(
        owner="u1",
        currency="USD",
        lines=[OrderLine("BATT14", 1, Money(3900, "USD"), Money(3900, "USD"))],
        total=Money(3900, "USD"),
    )
    return svc, order


def test_happy_path_progression(db):
    svc, order = make_order(db)
    assert order.state is OrderState.PENDING
    svc.transition(order.id, OrderState.PAYMENT_AUTHORIZED)
    svc.transition(order.id, OrderState.PAYMENT_CAPTURED)
    svc.transition(order.id, OrderState.FULFILLMENT_PROCESSING)
    svc.transition(order.id, OrderState.SHIPPED)
    final = svc.transition(order.id, OrderState.DELIVERED)
    assert final.state is OrderState.DELIVERED


def test_illegal_transition_rejected_and_logged(db):
    svc, order = make_order(db)
    # pending -> shipped skips payment/fulfillment: illegal.
    with pytest.raises(IllegalTransition) as exc:
        svc.transition(order.id, OrderState.SHIPPED)
    assert exc.value.frm == "pending"
    assert exc.value.to == "shipped"
    # The attempt is recorded as a violation for monitoring.
    violations = svc.violations(order.id)
    assert len(violations) == 1
    assert violations[0].from_state == "pending"
    assert violations[0].to_state == "shipped"
    assert violations[0].ok is False
    # The order did not move.
    assert svc.get(order.id).state is OrderState.PENDING


def test_cannot_leave_terminal_state(db):
    svc, order = make_order(db)
    svc.transition(order.id, OrderState.CANCELLED)
    with pytest.raises(IllegalTransition):
        svc.transition(order.id, OrderState.PAYMENT_AUTHORIZED)


def test_delivered_is_terminal(db):
    svc, order = make_order(db)
    for state in (
        OrderState.PAYMENT_AUTHORIZED,
        OrderState.PAYMENT_CAPTURED,
        OrderState.FULFILLMENT_PROCESSING,
        OrderState.SHIPPED,
        OrderState.DELIVERED,
    ):
        svc.transition(order.id, state)
    with pytest.raises(IllegalTransition):
        svc.transition(order.id, OrderState.CANCELLED)


def test_history_records_every_applied_transition(db):
    svc, order = make_order(db)
    svc.transition(order.id, OrderState.PAYMENT_AUTHORIZED, reason="auth ok")
    svc.transition(order.id, OrderState.CANCELLED, reason="customer changed mind")
    history = [e for e in svc.history(order.id) if e.ok]
    # created + 2 applied transitions
    assert [e.to_state for e in history] == ["pending", "payment_authorized", "cancelled"]


def test_can_transition_query(db):
    svc, order = make_order(db)
    assert svc.can_transition(order.id, OrderState.PAYMENT_AUTHORIZED) is True
    assert svc.can_transition(order.id, OrderState.DELIVERED) is False
