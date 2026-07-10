"""Verification: concurrent purchase of a single-unit item.

Spins up many threads that all try to buy the last unit at once and asserts
exactly one succeeds while the rest receive an out-of-stock error — the core
guarantee of optimistic, version-checked inventory.
"""

import threading

import pytest

from ecommerce import Inventory, OutOfStock
from ecommerce.errors import ConcurrencyConflict


def test_single_unit_exactly_one_winner(db_file):
    inv = Inventory(db_file)
    inv.set_stock("LAST-ONE", 1)

    winners: list[bool] = []
    lock = threading.Lock()
    barrier = threading.Barrier(32)

    def buy():
        barrier.wait()  # release all threads at the same instant
        try:
            inv.decrement("LAST-ONE", 1)
            with lock:
                winners.append(True)
        except (OutOfStock, ConcurrencyConflict):
            with lock:
                winners.append(False)

    threads = [threading.Thread(target=buy) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert winners.count(True) == 1, "exactly one purchase must succeed"
    assert winners.count(False) == 31
    assert inv.available("LAST-ONE") == 0
    assert inv.level("LAST-ONE").on_hand == 0


def test_limited_stock_never_oversells(db_file):
    # 5 units, 40 concurrent buyers -> exactly 5 succeed, on_hand hits 0.
    inv = Inventory(db_file)
    inv.set_stock("FIVE", 5)

    results: list[bool] = []
    lock = threading.Lock()
    barrier = threading.Barrier(40)

    def buy():
        barrier.wait()
        try:
            inv.decrement("FIVE", 1)
            ok = True
        except (OutOfStock, ConcurrencyConflict):
            ok = False
        with lock:
            results.append(ok)

    threads = [threading.Thread(target=buy) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 5
    assert inv.available("FIVE") == 0
    assert inv.level("FIVE").on_hand == 0  # never negative -> no oversell


def test_reservation_holds_then_expires(db, clock):
    inv = Inventory(db, clock=clock)
    inv.set_stock("HOLD", 2)
    res = inv.reserve("HOLD", 2, ttl_seconds=900)
    # While held, availability is zero and a new buyer is refused.
    assert inv.available("HOLD") == 0
    with pytest.raises(OutOfStock):
        inv.reserve("HOLD", 1)
    # After the TTL passes, the hold is swept and units return.
    clock.advance(901)
    assert inv.available("HOLD") == 2
    # Committing an expired reservation must fail.
    from ecommerce.errors import ReservationExpired

    with pytest.raises(ReservationExpired):
        inv.commit(res.id)


def test_reservation_commit_decrements_on_hand(db, clock):
    inv = Inventory(db, clock=clock)
    inv.set_stock("COMMIT", 3)
    res = inv.reserve("COMMIT", 2)
    assert inv.level("COMMIT").reserved == 2
    inv.commit(res.id)
    lvl = inv.level("COMMIT")
    assert lvl.on_hand == 1
    assert lvl.reserved == 0
    assert lvl.available == 1


def test_release_returns_units(db, clock):
    inv = Inventory(db, clock=clock)
    inv.set_stock("REL", 4)
    res = inv.reserve("REL", 3)
    assert inv.available("REL") == 1
    inv.release(res.id)
    assert inv.available("REL") == 4
