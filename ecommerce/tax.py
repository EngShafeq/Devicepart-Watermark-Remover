"""Tax computation by jurisdiction with regulator-friendly rounding.

Rates are expressed as integer basis points (1% = 100 bp) so no float ever
touches a tax figure. Tax is computed per line and rounded per line — the
approach most jurisdictions expect — then summed, which avoids the penny
drift you get from taxing the order total in one shot. Rounding is
half-away-from-zero via :meth:`Money.apply_rate`, matching the "round half
up" rule used by most VAT/sales-tax authorities.

A jurisdiction is resolved from a destination address (country, optionally
region/state). Tax-inclusive jurisdictions (VAT-style, where the shelf price
already contains tax) are supported by :meth:`TaxEngine.extract`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .money import Money


@dataclass
class TaxRate:
    """A named rate in basis points. ``inclusive`` means price already has tax."""

    name: str
    basis_points: int
    inclusive: bool = False


@dataclass
class TaxLine:
    sku: str
    taxable: Money
    tax: Money
    rate_name: str
    basis_points: int


@dataclass
class TaxResult:
    lines: List[TaxLine]
    total_tax: Money

    def tax_for(self, sku: str) -> Money:
        line = next((l for l in self.lines if l.sku == sku), None)
        return line.tax if line else Money.zero(self.total_tax.currency)


class TaxEngine:
    """Resolves a jurisdiction's rate and applies it line-by-line."""

    def __init__(self) -> None:
        # (country, region|None) -> TaxRate. A None region is the country default.
        self._rates: Dict[Tuple[str, Optional[str]], TaxRate] = {}

    def set_rate(
        self,
        country: str,
        basis_points: int,
        *,
        region: Optional[str] = None,
        name: str = "tax",
        inclusive: bool = False,
    ) -> None:
        self._rates[(country.upper(), region.upper() if region else None)] = TaxRate(
            name, basis_points, inclusive
        )

    def rate_for(self, country: str, region: Optional[str] = None) -> Optional[TaxRate]:
        country = country.upper()
        if region:
            specific = self._rates.get((country, region.upper()))
            if specific is not None:
                return specific
        return self._rates.get((country, None))

    def compute(
        self,
        taxable_lines: List[Tuple[str, Money]],
        *,
        country: str,
        region: Optional[str] = None,
    ) -> TaxResult:
        """Return per-line tax for the destination jurisdiction.

        ``taxable_lines`` is ``[(sku, taxable_amount), ...]`` where the
        taxable amount is the post-discount line total. For an inclusive
        (VAT) jurisdiction the tax is *extracted* from that amount instead of
        added on top.
        """
        currency = taxable_lines[0][1].currency if taxable_lines else "USD"
        rate = self.rate_for(country, region)
        if rate is None or rate.basis_points == 0:
            return TaxResult(
                [TaxLine(sku, amt, Money.zero(amt.currency), "none", 0) for sku, amt in taxable_lines],
                Money.zero(currency),
            )
        lines: List[TaxLine] = []
        total = 0
        for sku, amount in taxable_lines:
            if rate.inclusive:
                tax = self._extract(amount, rate.basis_points)
            else:
                tax = amount.apply_rate(rate.basis_points, 10_000)
            lines.append(TaxLine(sku, amount, tax, rate.name, rate.basis_points))
            total += tax.amount
        return TaxResult(lines, Money(total, currency))

    def extract(self, gross: Money, *, country: str, region: Optional[str] = None) -> Money:
        rate = self.rate_for(country, region)
        if rate is None:
            return Money.zero(gross.currency)
        return self._extract(gross, rate.basis_points)

    @staticmethod
    def _extract(gross: Money, basis_points: int) -> Money:
        """Tax component already inside ``gross``: gross * bp / (10000 + bp)."""
        return gross.apply_rate(basis_points, 10_000 + basis_points)
