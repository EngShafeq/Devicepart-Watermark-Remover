"""Money as integer minor currency units.

All monetary calculations in this package use integer arithmetic in the
*minor* unit of a currency (cents for USD/EUR, halalas for SAR, and so on).
Floating point is never used for money: it cannot represent 0.10 exactly and
accumulates rounding error across a cart of many lines.

Display formatting (grouping separators, currency symbols, decimal places) is
a strictly separate presentation concern implemented in :func:`Money.format`
and never mixed into calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

# Number of minor units in one major unit, per ISO-4217 currency.
# Most currencies use 2 decimal places; a few (JPY, KWD, BHD) differ.
_EXPONENT: Dict[str, int] = {
    "USD": 2,
    "EUR": 2,
    "GBP": 2,
    "SAR": 2,
    "AED": 2,
    "JPY": 0,
    "KWD": 3,
    "BHD": 3,
}

_SYMBOL: Dict[str, str] = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "SAR": "SAR ",
    "AED": "AED ",
    "JPY": "¥",
    "KWD": "KWD ",
    "BHD": "BHD ",
}


def minor_units(currency: str) -> int:
    """Return the number of decimal places for ``currency``."""
    return _EXPONENT.get(currency.upper(), 2)


class CurrencyMismatch(ValueError):
    """Raised when two :class:`Money` values of different currencies mix."""


@dataclass(frozen=True)
class Money:
    """An immutable amount of money stored as integer minor units.

    ``Money(1050, "USD")`` is $10.50. Arithmetic is exact; addition and
    subtraction require matching currencies. Multiplication is by an integer
    quantity only — never by a float — to keep the result exact.
    """

    amount: int  # minor units, may be negative (e.g. a discount line)
    currency: str

    def __post_init__(self) -> None:
        if not isinstance(self.amount, int):
            raise TypeError("Money.amount must be an int (minor units)")
        object.__setattr__(self, "currency", self.currency.upper())

    # ---- construction helpers -------------------------------------------
    @classmethod
    def zero(cls, currency: str) -> "Money":
        return cls(0, currency)

    @classmethod
    def from_major(cls, major: str | int, currency: str) -> "Money":
        """Build from a decimal string like ``"10.50"`` without float error.

        Accepts an int (whole major units) or a string. A string is parsed by
        splitting on the decimal point so no binary-float rounding occurs.
        """
        exp = minor_units(currency)
        if isinstance(major, int):
            return cls(major * (10 ** exp), currency)
        text = major.strip()
        neg = text.startswith("-")
        if neg:
            text = text[1:]
        if "." in text:
            whole, frac = text.split(".", 1)
        else:
            whole, frac = text, ""
        frac = (frac + "0" * exp)[:exp]  # pad/truncate to the currency's scale
        value = int(whole or "0") * (10 ** exp) + int(frac or "0")
        return cls(-value if neg else value, currency)

    # ---- arithmetic ------------------------------------------------------
    def _check(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise CurrencyMismatch(
                f"cannot combine {self.currency} with {other.currency}"
            )

    def __add__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, qty: int) -> "Money":
        if not isinstance(qty, int):
            raise TypeError("Money can only be multiplied by an int quantity")
        return Money(self.amount * qty, self.currency)

    __rmul__ = __mul__

    def __neg__(self) -> "Money":
        return Money(-self.amount, self.currency)

    def is_negative(self) -> bool:
        return self.amount < 0

    def is_zero(self) -> bool:
        return self.amount == 0

    def clamp_non_negative(self) -> "Money":
        """Floor at zero — used so a discount can never make a line negative."""
        return self if self.amount >= 0 else Money.zero(self.currency)

    # ---- proportional splitting -----------------------------------------
    def apply_rate(self, numerator: int, denominator: int, *, round_half_up: bool = True) -> "Money":
        """Multiply by the exact fraction ``numerator/denominator``.

        Used for tax and percentage discounts. The multiplication happens in
        integer space; only the final division rounds. ``round_half_up`` gives
        the "round half away from zero" behaviour most tax jurisdictions and
        card networks expect; set it ``False`` for banker's-unfriendly floor.
        """
        product = self.amount * numerator
        if round_half_up:
            # Round half away from zero for symmetric behaviour on refunds.
            if product >= 0:
                rounded = (product + denominator // 2) // denominator
            else:
                rounded = -((-product + denominator // 2) // denominator)
        else:
            rounded = product // denominator
        return Money(rounded, self.currency)

    def allocate(self, weights: list[int]) -> list["Money"]:
        """Split into parts proportional to ``weights`` with no lost cents.

        Distributes any rounding remainder one minor unit at a time to the
        largest fractional parts, guaranteeing the parts sum exactly to the
        original — the standard "largest remainder" allocation used for
        splitting an order-level discount across lines.
        """
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("cannot allocate against zero total weight")
        remainder = self.amount
        parts: list[int] = []
        for w in weights:
            share = self.amount * w // total_weight
            parts.append(share)
            remainder -= share
        # Hand out the leftover units to the largest weights first.
        order = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
        i = 0
        step = 1 if remainder >= 0 else -1
        while remainder != 0:
            parts[order[i % len(order)]] += step
            remainder -= step
            i += 1
        return [Money(p, self.currency) for p in parts]

    # ---- presentation (never used in calculation) -----------------------
    def format(self, *, symbol: bool = True) -> str:
        exp = minor_units(self.currency)
        sign = "-" if self.amount < 0 else ""
        units = abs(self.amount)
        if exp == 0:
            body = f"{units:,}"
        else:
            major, minor = divmod(units, 10 ** exp)
            body = f"{major:,}.{minor:0{exp}d}"
        prefix = _SYMBOL.get(self.currency, self.currency + " ") if symbol else ""
        return f"{sign}{prefix}{body}"

    def __str__(self) -> str:
        return self.format()
