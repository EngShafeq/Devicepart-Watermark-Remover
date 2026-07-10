"""Shipping methods with real-time rate calculation.

Rates are computed from the shipment's billable weight, destination zone, and
order value, so the checkout can present live quotes ("Standard $5.99,
Express $14.99") rather than a flat guess. A free-shipping threshold and a
carrier-style weight bracket are both supported. All figures are integer
minor units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .errors import ShippingUnavailable
from .money import Money


@dataclass
class Parcel:
    """What we're shipping: total weight (grams) and declared value."""

    weight_grams: int
    order_value: Money
    item_count: int = 1


@dataclass
class ShippingQuote:
    method_id: str
    label: str
    amount: Money
    eta_days: int


@dataclass
class ShippingMethod:
    """A carrier service with a rate function over destination zones.

    ``base`` + ``per_kg`` * ceil(kg) gives the raw rate; if the order value
    meets ``free_over`` the rate is waived. ``zones`` maps a destination zone
    to a surcharge multiplier in basis points (10000 = 1.0x).
    """

    id: str
    label: str
    base: Money
    per_kg: Money
    eta_days: int
    free_over: Optional[Money] = None
    zones: Dict[str, int] = field(default_factory=dict)  # zone -> multiplier bp
    available_zones: Optional[set] = None  # None => all zones

    def quote(self, parcel: Parcel, zone: str) -> Optional[ShippingQuote]:
        if self.available_zones is not None and zone not in self.available_zones:
            return None
        kg = -(-parcel.weight_grams // 1000)  # ceil division to whole kilos
        raw = self.base + self.per_kg * kg
        multiplier = self.zones.get(zone, 10_000)
        rate = raw.apply_rate(multiplier, 10_000)
        if self.free_over is not None and parcel.order_value.amount >= self.free_over.amount:
            rate = Money.zero(rate.currency)
        return ShippingQuote(self.id, self.label, rate, self.eta_days)


class ShippingCalculator:
    def __init__(self) -> None:
        self._methods: List[ShippingMethod] = []
        # Country -> zone resolver; defaults to the country code as the zone.
        self._zone_of: Callable[[str, Optional[str]], str] = (
            lambda country, region: country.upper()
        )

    def add_method(self, method: ShippingMethod) -> None:
        self._methods.append(method)

    def set_zone_resolver(self, fn: Callable[[str, Optional[str]], str]) -> None:
        self._zone_of = fn

    def zone_for(self, country: str, region: Optional[str] = None) -> str:
        return self._zone_of(country, region)

    def quotes(
        self, parcel: Parcel, *, country: str, region: Optional[str] = None
    ) -> List[ShippingQuote]:
        """Return every available shipping option for the destination.

        Raises :class:`ShippingUnavailable` if no method serves the zone, so
        the checkout can block progression rather than ship an un-priced order.
        """
        zone = self.zone_for(country, region)
        quotes = [q for m in self._methods if (q := m.quote(parcel, zone))]
        if not quotes:
            raise ShippingUnavailable(f"no shipping methods serve zone {zone}")
        quotes.sort(key=lambda q: q.amount.amount)
        return quotes

    def quote_method(
        self,
        method_id: str,
        parcel: Parcel,
        *,
        country: str,
        region: Optional[str] = None,
    ) -> ShippingQuote:
        zone = self.zone_for(country, region)
        method = next((m for m in self._methods if m.id == method_id), None)
        if method is None:
            raise ShippingUnavailable(f"unknown shipping method {method_id}")
        quote = method.quote(parcel, zone)
        if quote is None:
            raise ShippingUnavailable(f"{method_id} does not serve zone {zone}")
        return quote
