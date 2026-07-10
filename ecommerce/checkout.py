"""Multi-step checkout: address -> shipping -> tax -> review -> pay.

:class:`CheckoutSession` walks the shopper through the ordered steps, carrying
state forward. Each step validates its input before the next unlocks:

1. **Address** — validated for completeness (:func:`validate_address`).
2. **Shipping** — live rates from :class:`ShippingCalculator`; the shopper
   picks a method.
3. **Tax** — computed for the destination jurisdiction on the *discounted*
   line totals, plus tax on shipping where applicable.
4. **Review** — a fully itemised summary; here we re-quote and assert the
   price the shopper is about to pay still matches the cart. Any drift raises
   :class:`PriceChanged` so it can be surfaced *before* payment, per the
   "prices shown must equal prices charged" rule.
5. **Place** — soft-reserve stock, authorise + capture payment, commit the
   holds, and create the order. If payment fails the holds are released so
   the units aren't stranded.

Payment is abstracted behind :class:`PaymentGateway`; a deterministic
:class:`FakeGateway` is provided for tests and each real method (card,
wallet, COD) plugs in the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Protocol

from .cart import Cart, CartService
from .catalog import Catalog
from .errors import AddressInvalid, PaymentDeclined, PriceChanged
from .inventory import Inventory, Reservation
from .money import Money
from .orders import Order, OrderLine, OrderService, OrderState
from .pricing import LineInput, PricingEngine, Quote
from .shipping import Parcel, ShippingCalculator, ShippingQuote
from .tax import TaxEngine, TaxResult


class Step(str, Enum):
    ADDRESS = "address"
    SHIPPING = "shipping"
    TAX = "tax"
    REVIEW = "review"
    PLACED = "placed"


@dataclass
class Address:
    name: str
    line1: str
    city: str
    country: str
    postal_code: str
    region: Optional[str] = None
    line2: str = ""


def validate_address(address: Address) -> None:
    """Reject an address missing any field required to ship and tax it."""
    missing = [
        f
        for f in ("name", "line1", "city", "country", "postal_code")
        if not getattr(address, f).strip()
    ]
    if missing:
        raise AddressInvalid(f"address missing required fields: {', '.join(missing)}")
    if len(address.country.strip()) != 2:
        raise AddressInvalid("country must be a 2-letter ISO code")


class PaymentGateway(Protocol):
    def authorize(self, amount: Money, method: str, ref: str) -> str: ...
    def capture(self, auth_id: str) -> str: ...
    def void(self, auth_id: str) -> None: ...


class FakeGateway:
    """Deterministic in-memory gateway for tests and demos.

    Declines any amount whose minor value is in ``decline_amounts`` (so a test
    can force a failure) and any method not in ``supported``.
    """

    def __init__(self, *, supported=("card", "wallet", "cod"), decline_amounts=()):
        self.supported = set(supported)
        self.decline_amounts = set(decline_amounts)
        self.authorizations: Dict[str, dict] = {}
        self.captures: Dict[str, dict] = {}
        self._seq = 0

    def _next(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}_{self._seq:06d}"

    def authorize(self, amount: Money, method: str, ref: str) -> str:
        if method not in self.supported:
            raise PaymentDeclined(f"unsupported payment method {method}")
        if amount.amount in self.decline_amounts:
            raise PaymentDeclined("card declined")
        auth_id = self._next("auth")
        self.authorizations[auth_id] = {"amount": amount, "method": method, "ref": ref}
        return auth_id

    def capture(self, auth_id: str) -> str:
        if auth_id not in self.authorizations:
            raise PaymentDeclined("no such authorization")
        capture_id = self._next("cap")
        self.captures[capture_id] = dict(self.authorizations[auth_id])
        return capture_id

    def void(self, auth_id: str) -> None:
        self.authorizations.pop(auth_id, None)


@dataclass
class CheckoutState:
    cart_id: str
    currency: str
    step: Step = Step.ADDRESS
    address: Optional[Address] = None
    shipping_choice: Optional[ShippingQuote] = None
    coupon_code: Optional[str] = None
    quote: Optional[Quote] = None
    tax: Optional[TaxResult] = None


@dataclass
class OrderSummary:
    subtotal: Money
    discount: Money
    shipping: Money
    tax: Money
    total: Money
    lines: List[OrderLine] = field(default_factory=list)


class CheckoutService:
    def __init__(
        self,
        *,
        catalog: Catalog,
        carts: CartService,
        inventory: Inventory,
        pricing: PricingEngine,
        tax: TaxEngine,
        shipping: ShippingCalculator,
        orders: OrderService,
        gateway: PaymentGateway,
        weigh=lambda cart: 500,  # grams per unit; override with real weights
    ):
        self.catalog = catalog
        self.carts = carts
        self.inventory = inventory
        self.pricing = pricing
        self.tax = tax
        self.shipping = shipping
        self.orders = orders
        self.gateway = gateway
        self._weigh = weigh

    # ---- step 1: address -------------------------------------------------
    def begin(self, cart_id: str, *, coupon_code: Optional[str] = None) -> CheckoutState:
        cart = self.carts.get(cart_id)
        if cart is None or not cart.items:
            raise ValueError("cannot check out an empty cart")
        # Self-heal before the shopper commits money.
        self.carts.reconcile(cart_id)
        cart = self.carts.get(cart_id)
        if cart is None or not cart.items:
            raise ValueError("cart became empty after reconciliation")
        return CheckoutState(cart_id=cart_id, currency=cart.currency, coupon_code=coupon_code)

    def set_address(self, state: CheckoutState, address: Address) -> CheckoutState:
        validate_address(address)
        state.address = address
        state.step = Step.SHIPPING
        return state

    # ---- step 2: shipping ------------------------------------------------
    def shipping_options(self, state: CheckoutState) -> List[ShippingQuote]:
        if state.address is None:
            raise ValueError("address required before shipping")
        cart = self._cart(state)
        parcel = Parcel(
            weight_grams=self._weigh(cart),
            order_value=cart.subtotal,
            item_count=cart.item_count,
        )
        return self.shipping.quotes(
            parcel, country=state.address.country, region=state.address.region
        )

    def choose_shipping(self, state: CheckoutState, method_id: str) -> CheckoutState:
        if state.address is None:
            raise ValueError("address required before shipping")
        cart = self._cart(state)
        parcel = Parcel(self._weigh(cart), cart.subtotal, cart.item_count)
        state.shipping_choice = self.shipping.quote_method(
            method_id, parcel, country=state.address.country, region=state.address.region
        )
        state.step = Step.TAX
        return state

    # ---- step 3: tax + quote --------------------------------------------
    def compute_totals(self, state: CheckoutState) -> OrderSummary:
        cart = self._cart(state)
        assert state.address is not None and state.shipping_choice is not None
        quote = self._quote(cart, state.coupon_code)
        state.quote = quote
        # Tax the post-discount line totals (subtotal + line-level discount).
        taxable = [
            (ql.sku, ql.subtotal + ql.line_discount) for ql in quote.lines
        ]
        tax = self.tax.compute(
            taxable, country=state.address.country, region=state.address.region
        )
        state.tax = tax
        shipping_cost = state.shipping_choice.amount
        # Tax on shipping, where the jurisdiction charges it.
        shipping_tax = self.tax.compute(
            [("__shipping__", shipping_cost)],
            country=state.address.country,
            region=state.address.region,
        ).total_tax
        tax_total = tax.total_tax + shipping_tax
        total = quote.total + shipping_cost + tax_total
        state.step = Step.REVIEW
        lines = [
            OrderLine(
                ql.sku,
                ql.quantity,
                ql.unit_price,
                ql.subtotal + ql.line_discount,
            )
            for ql in quote.lines
        ]
        return OrderSummary(
            subtotal=quote.subtotal,
            discount=quote.discount_total,
            shipping=shipping_cost,
            tax=tax_total,
            total=total,
            lines=lines,
        )

    # ---- step 4/5: review + place ---------------------------------------
    def place_order(
        self, state: CheckoutState, *, payment_method: str, owner: Optional[str]
    ) -> Order:
        """Reserve stock, take payment, and create the order atomically.

        Before charging, the current cart is re-quoted and every line price is
        compared to the price captured in the review step; a mismatch raises
        :class:`PriceChanged` so the shopper re-confirms rather than being
        charged a surprise amount. Stock is soft-reserved first; if payment is
        declined the holds are released.
        """
        cart = self._cart(state)
        if state.quote is None or state.tax is None or state.shipping_choice is None:
            raise ValueError("totals must be computed before placing the order")

        # Price-consistency guard: re-quote NOW and compare to the reviewed quote.
        fresh = self._quote(cart, state.coupon_code)
        for reviewed in state.quote.lines:
            current = fresh.line(reviewed.sku)
            if current is None:
                raise PriceChanged(reviewed.sku, reviewed.unit_price, None)
            if current.unit_price.amount != reviewed.unit_price.amount:
                raise PriceChanged(
                    reviewed.sku, reviewed.unit_price, current.unit_price
                )

        summary = self._summary_from(state, fresh)
        reservations: List[Reservation] = []
        try:
            # 1) Soft-reserve every line; OutOfStock aborts before any charge.
            for ql in fresh.lines:
                reservations.append(self.inventory.reserve(ql.sku, ql.quantity))
            # 2) Authorise + capture payment for the grand total.
            auth = self.gateway.authorize(summary.total, payment_method, ref=state.cart_id)
            capture = self.gateway.capture(auth)
            # 3) Commit the holds into real stock decrements.
            for res in reservations:
                self.inventory.commit(res.id)
        except Exception:
            for res in reservations:
                self.inventory.release(res.id)
            raise

        # 4) Create the order in PAYMENT_CAPTURED-ready state and record it.
        order = self.orders.create(
            owner=owner,
            currency=state.currency,
            lines=summary.lines,
            total=summary.total,
            payload={
                "shipping": {
                    "method": state.shipping_choice.method_id,
                    "amount": summary.shipping.amount,
                },
                "tax": summary.tax.amount,
                "discount": summary.discount.amount,
                "subtotal": summary.subtotal.amount,
                "payment": {"method": payment_method, "capture_id": capture},
                "address": vars(state.address) if state.address else None,
                "coupon": state.coupon_code,
            },
        )
        # Advance the state machine to reflect the captured payment.
        self.orders.transition(order.id, OrderState.PAYMENT_AUTHORIZED, reason="authorized")
        order = self.orders.transition(order.id, OrderState.PAYMENT_CAPTURED, reason="captured")
        if state.coupon_code:
            self.pricing.record_redemption(state.coupon_code, order.id)
        state.step = Step.PLACED
        return order

    # ---- helpers ---------------------------------------------------------
    def _cart(self, state: CheckoutState) -> Cart:
        cart = self.carts.get(state.cart_id)
        if cart is None:
            raise ValueError("cart no longer exists")
        return cart

    def _quote(self, cart: Cart, coupon_code: Optional[str]) -> Quote:
        line_inputs: List[LineInput] = []
        for item in cart.items:
            resolved = self.catalog.resolve(item.sku)
            product = resolved[0] if resolved else None
            line_inputs.append(
                LineInput(
                    sku=item.sku,
                    unit_price=item.unit_price,
                    quantity=item.quantity,
                    product_id=product.id if product else "",
                    category=product.category if product else "",
                )
            )
        return self.pricing.quote(
            line_inputs, currency=cart.currency, coupon_code=coupon_code
        )

    def _summary_from(self, state: CheckoutState, quote: Quote) -> OrderSummary:
        assert state.address is not None and state.shipping_choice is not None
        taxable = [(ql.sku, ql.subtotal + ql.line_discount) for ql in quote.lines]
        tax = self.tax.compute(
            taxable, country=state.address.country, region=state.address.region
        )
        shipping_cost = state.shipping_choice.amount
        shipping_tax = self.tax.compute(
            [("__shipping__", shipping_cost)],
            country=state.address.country,
            region=state.address.region,
        ).total_tax
        tax_total = tax.total_tax + shipping_tax
        total = quote.total + shipping_cost + tax_total
        lines = [
            OrderLine(ql.sku, ql.quantity, ql.unit_price, ql.subtotal + ql.line_discount)
            for ql in quote.lines
        ]
        return OrderSummary(
            quote.subtotal, quote.discount_total, shipping_cost, tax_total, total, lines
        )
