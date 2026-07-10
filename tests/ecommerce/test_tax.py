"""Verification: tax against known rates for multiple jurisdictions.

Confirms per-line computation, half-up rounding, and VAT-inclusive
extraction match regulatory expectations.
"""

from ecommerce import Money, TaxEngine


def make_engine() -> TaxEngine:
    engine = TaxEngine()
    engine.set_rate("US", 0, name="none")  # no federal sales tax
    engine.set_rate("US", 725, region="CA", name="CA sales tax")  # 7.25%
    engine.set_rate("US", 400, region="NY", name="NY sales tax")  # 4.00%
    engine.set_rate("SA", 1500, name="VAT", inclusive=False)  # 15% KSA VAT
    engine.set_rate("GB", 2000, name="VAT", inclusive=True)  # 20% inclusive
    return engine


def test_california_rate_rounds_half_up():
    engine = make_engine()
    # $12.99 @ 7.25% = 0.941775 -> $0.94
    result = engine.compute([("A", Money(1299, "USD"))], country="US", region="CA")
    assert result.total_tax.amount == 94


def test_new_york_rate():
    engine = make_engine()
    # $100.00 @ 4% = $4.00 exactly
    result = engine.compute([("A", Money(10000, "USD"))], country="US", region="NY")
    assert result.total_tax.amount == 400


def test_no_tax_jurisdiction():
    engine = make_engine()
    result = engine.compute([("A", Money(5000, "USD"))], country="US")
    assert result.total_tax.amount == 0


def test_per_line_rounding_beats_whole_order_rounding():
    engine = make_engine()
    # Three lines at $3.33 each. Per-line: round(3.33*7.25%)=round(24.14c)=24c
    # each -> 72c. This is the regulator-expected per-line method.
    lines = [("A", Money(333, "USD")), ("B", Money(333, "USD")), ("C", Money(333, "USD"))]
    result = engine.compute(lines, country="US", region="CA")
    assert [l.tax.amount for l in result.lines] == [24, 24, 24]
    assert result.total_tax.amount == 72


def test_ksa_vat_added_on_top():
    engine = make_engine()
    result = engine.compute([("A", Money(10000, "SAR"))], country="SA")
    # 100.00 SAR @ 15% = 15.00 SAR
    assert result.total_tax.amount == 1500


def test_uk_vat_extracted_from_inclusive_price():
    engine = make_engine()
    # £120.00 inclusive of 20% VAT -> VAT component = 120 * 20/120 = £20.00
    result = engine.compute([("A", Money(12000, "GBP"))], country="GB")
    assert result.total_tax.amount == 2000
    # And the extract() helper agrees.
    assert engine.extract(Money(12000, "GBP"), country="GB").amount == 2000


def test_region_falls_back_to_country_default():
    engine = make_engine()
    # An unconfigured US region gets the country default (0%).
    result = engine.compute([("A", Money(10000, "USD"))], country="US", region="TX")
    assert result.total_tax.amount == 0
