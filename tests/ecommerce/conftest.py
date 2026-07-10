"""Shared fixtures: a small but realistic device-parts store.

The catalog mirrors the repository's domain (phone/tablet spare parts) so the
tests read like a real store: hierarchical categories, variants with
size/color options, tiered prices, and multi-currency pricing.
"""

from __future__ import annotations

import os

import pytest

from ecommerce import (
    Catalog,
    Category,
    Database,
    Inventory,
    Money,
    PriceTier,
    Product,
    Variant,
)


class FakeClock:
    """A manually-advanced clock so time-based logic is deterministic."""

    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def db_file(tmp_path):
    """A real on-disk SQLite DB (needed for cross-thread concurrency tests)."""
    path = os.path.join(tmp_path, "store.db")
    return Database(path)


@pytest.fixture
def db() -> Database:
    """A fast in-memory DB for single-threaded tests."""
    return Database(":memory:")


@pytest.fixture
def catalog() -> Catalog:
    cat = Catalog()
    cat.add_category(Category("root", "All Parts"))
    cat.add_category(Category("screens", "Screens", parent="root"))
    cat.add_category(Category("oled", "OLED Screens", parent="screens"))
    cat.add_category(Category("batteries", "Batteries", parent="root"))

    # An OLED screen with size variants and tiered/multi-currency pricing.
    screen = Product(
        id="p-oled-14",
        title="iPhone 14 OLED Screen Assembly",
        category="oled",
        description="Genuine-grade OLED display assembly with digitizer",
        attributes={"brand": "DevicePart", "model": "iPhone 14", "grade": "premium"},
        tags=["screen", "oled", "display", "digitizer"],
        popularity=120,
    )
    v = Variant(sku="OLED14-BLK")
    v.set_price(PriceTier.RETAIL, Money.from_major("129.00", "USD"))
    v.set_price(PriceTier.WHOLESALE, Money.from_major("99.00", "USD"))
    v.set_price(PriceTier.MEMBER, Money.from_major("119.00", "USD"))
    v.set_price(PriceTier.RETAIL, Money.from_major("129.00", "EUR"))
    screen.variants.append(v)
    cat.add_product(screen)

    # A battery, cheaper, used for bundles and multi-line carts.
    battery = Product(
        id="p-batt-14",
        title="iPhone 14 Replacement Battery",
        category="batteries",
        description="High-capacity lithium replacement battery",
        attributes={"brand": "DevicePart", "model": "iPhone 14", "grade": "standard"},
        tags=["battery", "lithium", "power"],
        popularity=80,
    )
    b = Variant(sku="BATT14")
    b.set_price(PriceTier.RETAIL, Money.from_major("39.00", "USD"))
    b.set_price(PriceTier.WHOLESALE, Money.from_major("29.00", "USD"))
    battery.variants.append(b)
    cat.add_product(battery)

    # A tool the store may discontinue mid-test.
    tool = Product(
        id="p-tool",
        title="Precision Screwdriver Kit",
        category="root",
        description="24-piece repair toolkit",
        attributes={"brand": "DevicePart", "grade": "standard"},
        tags=["tool", "screwdriver", "kit"],
        popularity=200,
    )
    t = Variant(sku="TOOLKIT")
    t.set_price(PriceTier.RETAIL, Money.from_major("19.00", "USD"))
    tool.variants.append(t)
    cat.add_product(tool)
    return cat


@pytest.fixture
def inventory(db, clock) -> Inventory:
    inv = Inventory(db, clock=clock)
    inv.set_stock("OLED14-BLK", 10)
    inv.set_stock("BATT14", 50)
    inv.set_stock("TOOLKIT", 5)
    return inv
