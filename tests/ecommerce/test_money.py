"""Money must be exact integer minor units, never float."""

import pytest

from ecommerce import CurrencyMismatch, Money


def test_from_major_string_is_exact():
    # 0.1 + 0.2 is a classic float trap; in minor units it's just integers.
    assert Money.from_major("0.10", "USD").amount == 10
    assert Money.from_major("0.20", "USD").amount == 20
    total = Money.from_major("0.10", "USD") + Money.from_major("0.20", "USD")
    assert total.amount == 30
    assert total.format() == "$0.30"


def test_currency_scale_varies():
    assert Money.from_major("10", "JPY").amount == 10  # 0 decimal places
    assert Money.from_major("10.500", "KWD").amount == 10_500  # 3 decimal places
    assert Money.from_major("10.50", "USD").amount == 1_050


def test_mixed_currency_addition_rejected():
    with pytest.raises(CurrencyMismatch):
        Money(100, "USD") + Money(100, "EUR")


def test_multiplication_requires_integer_quantity():
    assert (Money(199, "USD") * 3).amount == 597
    with pytest.raises(TypeError):
        Money(199, "USD") * 1.5  # type: ignore[operator]


def test_apply_rate_rounds_half_away_from_zero():
    # 100 * 8.75% = 8.75 -> rounds to 9
    assert Money(100, "USD").apply_rate(875, 10_000).amount == 9
    # symmetric on the way down
    assert Money(-100, "USD").apply_rate(875, 10_000).amount == -9


def test_allocate_distributes_every_cent():
    # $10.00 split across weights that don't divide evenly must still sum to 1000.
    parts = Money(1000, "USD").allocate([1, 1, 1])
    assert sum(p.amount for p in parts) == 1000
    assert [p.amount for p in parts] == [334, 333, 333]  # largest-remainder


def test_clamp_non_negative_floors_a_discount():
    discounted = Money(500, "USD") - Money(800, "USD")
    assert discounted.amount == -300
    assert discounted.clamp_non_negative().amount == 0


def test_format_is_presentation_only():
    assert Money(123456, "USD").format() == "$1,234.56"
    assert Money(123456, "USD").format(symbol=False) == "1,234.56"
    assert Money(1000, "JPY").format() == "¥1,000"
