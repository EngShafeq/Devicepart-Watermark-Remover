"""A transactional e-commerce backend built on the standard library.

The package implements the core systems that power online retail — catalog,
search, inventory, cart, pricing, tax, shipping, checkout, orders, returns,
notifications, and analytics — with two non-negotiable disciplines:

* **Money is exact.** Every calculation is integer arithmetic in minor
  currency units (:mod:`ecommerce.money`); formatting is a separate concern.
* **State is durable and concurrency-safe.** Cart and order state live in
  SQLite (:mod:`ecommerce.db`); inventory decrements use optimistic
  version checks (:mod:`ecommerce.inventory`) so simultaneous buyers of the
  last unit cannot both win.

See ``ecommerce/README.md`` for an end-to-end walkthrough and
``tests/ecommerce`` for the verification suite.
"""

from .analytics import AnalyticsService, Event, FunnelStage
from .cart import Cart, CartItem, CartService, ReconcileReport
from .catalog import Catalog, Category, PriceTier, Product, Variant
from .checkout import (
    Address,
    CheckoutService,
    CheckoutState,
    FakeGateway,
    OrderSummary,
    Step,
    validate_address,
)
from .db import Database
from .errors import (
    ConcurrencyConflict,
    CouponError,
    EcommerceError,
    IllegalTransition,
    OutOfStock,
    PaymentDeclined,
    PriceChanged,
)
from .inventory import Inventory, Reservation, StockLevel
from .money import Money, CurrencyMismatch
from .notifications import NotificationService, RecordingTransport
from .orders import Order, OrderLine, OrderService, OrderState, TRANSITIONS
from .pricing import (
    Bundle,
    Coupon,
    DiscountType,
    LineInput,
    PricingEngine,
    PromotionRule,
    Quote,
    TieredPrice,
    TierRule,
)
from .returns import RMA, RMAState, ReturnLine, ReturnsService, RefundGateway
from .search import SearchIndex, SearchResults
from .shipping import Parcel, ShippingCalculator, ShippingMethod
from .tax import TaxEngine, TaxRate

__all__ = [
    "AnalyticsService", "Event", "FunnelStage",
    "Cart", "CartItem", "CartService", "ReconcileReport",
    "Catalog", "Category", "PriceTier", "Product", "Variant",
    "Address", "CheckoutService", "CheckoutState", "FakeGateway",
    "OrderSummary", "Step", "validate_address",
    "Database",
    "ConcurrencyConflict", "CouponError", "EcommerceError", "IllegalTransition",
    "OutOfStock", "PaymentDeclined", "PriceChanged",
    "Inventory", "Reservation", "StockLevel",
    "Money", "CurrencyMismatch",
    "NotificationService", "RecordingTransport",
    "Order", "OrderLine", "OrderService", "OrderState", "TRANSITIONS",
    "Bundle", "Coupon", "DiscountType", "LineInput", "PricingEngine",
    "PromotionRule", "Quote", "TieredPrice", "TierRule",
    "RMA", "RMAState", "ReturnLine", "ReturnsService", "RefundGateway",
    "SearchIndex", "SearchResults",
    "Parcel", "ShippingCalculator", "ShippingMethod",
    "TaxEngine", "TaxRate",
]
