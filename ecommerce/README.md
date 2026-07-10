# E-commerce transactional backend

A dependency-free (standard-library-only) implementation of the systems that
power online retail — catalog, search, inventory, cart, pricing, tax,
shipping, checkout, orders, returns, notifications, and analytics.

It is built around two disciplines that the rest of the design serves:

1. **Money is exact integer arithmetic.** Every price, discount, tax, and
   refund is an integer count of *minor* currency units (cents, halalas,
   fils). Floating point never touches money. Display formatting is a
   separate concern (`Money.format`).
2. **State is durable and concurrency-safe.** Cart and order state live in
   SQLite so they survive a browser closing, a device switch, or a server
   restart. Inventory changes use optimistic version checks so two shoppers
   buying the last unit cannot both succeed.

## Quick start

```bash
python examples/ecommerce_demo.py      # end-to-end walkthrough
python -m pytest tests/ecommerce -q    # the verification suite
```

## Modules

| Module | Responsibility |
| --- | --- |
| `money` | Integer-minor-unit money, multi-currency, exact allocation & rounding |
| `db` | SQLite schema + WAL connections (durable, concurrency-safe state) |
| `catalog` | Hierarchical categories, variants, tiered & multi-currency prices |
| `search` | Inverted-index full-text search: facets, typo tolerance, synonyms, ranking |
| `inventory` | Optimistic stock control, time-limited soft reservations |
| `cart` | Persistent carts, anonymous→user merge, self-healing reconcile |
| `pricing` | Coupons, tiered/bundle pricing, percentage/fixed & automatic promos |
| `tax` | Jurisdiction rates in basis points, per-line half-up rounding, VAT extraction |
| `shipping` | Weight/zone rate calculation, free-shipping thresholds |
| `checkout` | Multi-step flow: address → shipping → tax → review → pay |
| `orders` | Validated state machine with audited (logged) transitions |
| `returns` | RMA lifecycle: label → inspection → refund → restock |
| `notifications` | Templated transactional email pipeline with delivery tracking |
| `analytics` | Funnel event tracking (view → cart → checkout → purchase) |

## How the hard requirements are met

**No overselling under concurrency.** `Inventory.decrement` reads the SKU's
`version`, then writes with `WHERE sku = ? AND version = ? AND on_hand -
reserved >= ?`. If a competing transaction changed the row, the update
affects zero rows and we retry against fresh state. The availability check
and the version bump are the *same* conditional `UPDATE`, so simultaneous
buyers of the last unit can't both win — verified in
`tests/ecommerce/test_inventory_concurrency.py` with 32 threads racing for
one unit (exactly one succeeds).

**Soft reservations.** Checkout takes a time-limited hold
(`Inventory.reserve`) instead of decrementing immediately; an abandoned
checkout's units return automatically once the TTL lapses. Payment capture
converts the hold to a real decrement (`commit`); a decline releases it.

**Exact money.** All calculation is in minor units. Percentage discounts and
tax multiply in integer space and round once (half away from zero); an
order-level discount is split across lines with `Money.allocate`
(largest-remainder) so the parts always sum to the whole with no lost cent.

**Prices shown == prices charged.** Before taking payment,
`CheckoutService.place_order` re-quotes the cart and compares every line
price to the reviewed quote; any drift raises `PriceChanged` so it is
surfaced *before* the card is charged.

**Atomic coupon validation.** `Coupon.validate` checks expiry, usage limit,
minimum order, and product eligibility together, reading the live
`coupon_usage` count so a single-use code can't be redeemed twice
concurrently.

**Validated order lifecycle.** `OrderService.transition` permits only the
edges declared in `TRANSITIONS`; an illegal attempt is rejected with a clear
`IllegalTransition` error *and* written to the `order_events` audit table
(`OrderService.violations`) for monitoring.

**Search within budget.** The inverted index answers a query in time
proportional to the matching documents, not the catalog. At 100,000 SKUs the
worst-case two-common-term query returns in well under the 200 ms budget
(`tests/ecommerce/test_search.py`).

## Verification map

Each requirement in the brief maps to a test:

| Requirement | Test |
| --- | --- |
| Concurrent single-unit purchase → one winner | `test_inventory_concurrency.py` |
| Full checkout across payment methods | `test_checkout_flow.py` |
| Coupon edge cases | `test_coupons.py` |
| Cart merge without duplicates | `test_cart_merge.py` |
| Tax by jurisdiction with rounding | `test_tax.py` |
| Illegal order transitions rejected + logged | `test_orders_state_machine.py` |
| Search relevance/facets/typo/latency | `test_search.py` |
| Returns, notifications, analytics | `test_returns_notifications_analytics.py` |
| Integer-money invariants | `test_money.py` |
```
