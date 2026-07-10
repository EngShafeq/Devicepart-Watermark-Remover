"""Verification: cart merging and persistence.

Confirms an anonymous cart folds into the user's existing cart without
duplicating lines (shared SKUs sum), that state survives a "restart", and
that reconcile drops discontinued / out-of-stock items.
"""

from ecommerce import CartService, Database, Inventory, PriceTier


def make_cart_service(db, catalog, inventory):
    return CartService(db, catalog, inventory)


def test_merge_sums_shared_skus_without_duplicates(db, catalog, inventory):
    carts = make_cart_service(db, catalog, inventory)
    # Anonymous shopper adds a screen and a battery.
    anon = carts.create(currency="USD")
    carts.add_item(anon.id, "OLED14-BLK", 1)
    carts.add_item(anon.id, "BATT14", 2)
    # The account already had a battery in a saved cart.
    user = carts.create(owner="user-1", currency="USD")
    carts.add_item(user.id, "BATT14", 1)

    merged = carts.merge(anon.id, user.id)

    skus = {i.sku: i.quantity for i in merged.items}
    assert skus == {"OLED14-BLK": 1, "BATT14": 3}  # 2 + 1 summed, no dupe line
    assert len(merged.items) == 2
    # Anonymous cart is gone.
    assert carts.get(anon.id) is None


def test_merge_clamps_to_available_stock(db, catalog, inventory):
    carts = make_cart_service(db, catalog, inventory)
    inventory.set_stock("OLED14-BLK", 3)  # only 3 in stock
    anon = carts.create(currency="USD")
    carts.add_item(anon.id, "OLED14-BLK", 2)
    user = carts.create(owner="user-2", currency="USD")
    carts.add_item(user.id, "OLED14-BLK", 2)
    merged = carts.merge(anon.id, user.id)
    # 2 + 2 = 4 requested, clamped to 3 available.
    assert merged.item("OLED14-BLK").quantity == 3


def test_cart_survives_restart(db_file, catalog, inventory):
    # Persist a cart, then open a brand-new service/DB handle on the same file.
    inv = Inventory(db_file)
    inv.set_stock("BATT14", 10)
    carts = CartService(db_file, catalog, inv)
    cart = carts.create(owner="user-3", currency="USD")
    carts.add_item(cart.id, "BATT14", 3)

    reopened_db = Database(db_file.path)  # simulates a server restart
    reopened = CartService(reopened_db, catalog, Inventory(reopened_db))
    restored = reopened.get(cart.id)
    assert restored is not None
    assert restored.item("BATT14").quantity == 3


def test_reconcile_removes_discontinued_and_clamps(db, catalog, inventory):
    carts = make_cart_service(db, catalog, inventory)
    cart = carts.create(currency="USD")
    carts.add_item(cart.id, "OLED14-BLK", 2)
    carts.add_item(cart.id, "TOOLKIT", 1)
    carts.add_item(cart.id, "BATT14", 5)

    # Toolkit is discontinued; OLED stock drops below the cart quantity.
    catalog.discontinue("TOOLKIT")
    inventory.set_stock("OLED14-BLK", 1)

    cart, report = carts.reconcile(cart.id)
    skus = {i.sku: i.quantity for i in cart.items}
    assert "TOOLKIT" not in skus
    assert "TOOLKIT" in report.removed_discontinued
    assert skus["OLED14-BLK"] == 1  # clamped from 2 -> 1
    assert report.clamped["OLED14-BLK"] == 1
    assert skus["BATT14"] == 5  # unaffected


def test_add_item_wholesale_tier_price(db, catalog, inventory):
    carts = CartService(db, catalog, inventory, tier=PriceTier.WHOLESALE)
    cart = carts.create(owner="dealer", currency="USD")
    carts.add_item(cart.id, "OLED14-BLK", 1)
    # Wholesale price is 9900, not the 12900 retail price.
    assert cart.currency == "USD"
    assert carts.get(cart.id).item("OLED14-BLK").unit_price.amount == 9900


def test_out_of_stock_reconcile_removes_line(db, catalog, inventory):
    carts = make_cart_service(db, catalog, inventory)
    cart = carts.create(currency="USD")
    carts.add_item(cart.id, "BATT14", 2)
    inventory.set_stock("BATT14", 0)
    cart, report = carts.reconcile(cart.id)
    assert cart.item("BATT14") is None
    assert "BATT14" in report.removed_out_of_stock
